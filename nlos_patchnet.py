""" Code for NLOS Patch Network
"""
from functools import partial
from typing import List, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def always(val):
    return lambda *args: val


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def divisible_by(numer, denom):
    return (numer % denom) == 0


# auto grouping images


def group_images_by_max_seq_len(
    images: List[Tensor], patch_size: int, calc_token_dropout=None, max_seq_len=2048
) -> List[List[Tensor]]:
    """
    Takes a list of images, a patch size, a function or value for calculating token dropout,
    and a maximum sequence length as input. It returns a list of groups of images,
    where each group's total sequence length does not exceed the maximum sequence length.
    """

    calc_token_dropout = default(calc_token_dropout, always(0.0))

    groups = []
    group = []
    seq_len = 0  # total sequence length

    if isinstance(calc_token_dropout, (float, int)):
        """
        If calc_token_dropout is a float or an integer
        it converts it to a function that always returns this value.
        """
        calc_token_dropout = always(calc_token_dropout)

    for image in images:
        assert isinstance(image, Tensor)

        image_dims = image.shape[-2:]
        ph, pw = map(lambda t: t // patch_size, image_dims)

        image_seq_len = ph * pw
        image_seq_len = int(image_seq_len * (1 - calc_token_dropout(*image_dims)))

        assert (
            image_seq_len <= max_seq_len
        ), f"image with dimensions {image_dims} exceeds maximum sequence length"

        if (seq_len + image_seq_len) > max_seq_len:
            groups.append(group)
            group = []
            seq_len = 0

        group.append(image)
        seq_len += image_seq_len

    if len(group) > 0:
        groups.append(group)

    return groups

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        normed = F.normalize(x, dim=-1)
        return normed * self.scale * self.gamma

def FeedForward(dim, hidden_dim, dropout=0.0):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.norm = LayerNorm(dim)

        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, attn_mask=None):
        x = self.norm(x)
        kv_input = default(context, x)

        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        q = self.q_norm(q)
        k = self.k_norm(k)

        dots = torch.matmul(q, k.transpose(-1, -2))

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

        self.norm = LayerNorm(dim)

    def forward(self, x, mask=None, attn_mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask, attn_mask=attn_mask) + x
            x = ff(x) + x

        return self.norm(x)


class NaViT(nn.Module):
    def __init__(
        self,
        image_size=256,
        patch_size=32,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        channels=3,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1,
        token_dropout_prob=0.1,
        **kwargs,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)

        # what percent of tokens to dropout
        # if int or float given, then assume constant dropout prob
        # otherwise accept a callback that in turn calculates dropout prob from height and width

        self.calc_token_dropout = None

        if callable(token_dropout_prob):
            self.calc_token_dropout = token_dropout_prob

        elif isinstance(token_dropout_prob, (float, int)):
            assert 0.0 < token_dropout_prob < 1.0
            token_dropout_prob = float(token_dropout_prob)
            self.calc_token_dropout = lambda height, width: token_dropout_prob

        # calculate patching related stuff

        assert divisible_by(image_height, patch_size) and divisible_by(
            image_width, patch_size
        ), "Image dimensions must be divisible by the patch size."

        patch_height_dim, patch_width_dim = (image_height // patch_size), (
            image_width // patch_size
        )
        patch_dim = channels * (patch_size**2)

        self.channels = channels
        self.patch_size = patch_size

        # Used to convert patches into embeddings.
        # This layer consists of a layer normalization layer, a linear layer, and another layer normalization layer.
        # The input and output dimensions of these layers match the dimensions of the patches
        # and the specified dimension dim, respectively.
        self.to_patch_embedding_raw = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim),
        )
        self.to_patch_embedding_delta = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim),
        )

        # Positional embeddings for the height and width of the patches, respectively.
        # They are initialized as learnable parameters with random values.
        self.pos_embed_height_raw = nn.Parameter(torch.randn(patch_height_dim, dim))
        self.pos_embed_width_raw = nn.Parameter(torch.randn(patch_width_dim, dim))
        self.pos_embed_height_delta = nn.Parameter(torch.randn(patch_height_dim, dim))
        self.pos_embed_width_delta = nn.Parameter(torch.randn(patch_width_dim, dim))

        # This is a dropout layer, which randomly sets a fraction of input units to 0 at each update during training time to prevent overfitting. The fraction of zeroing out is determined by emb_dropout.
        self.dropout_raw = nn.Dropout(emb_dropout)
        self.dropout_delta = nn.Dropout(emb_dropout)

        self.transformer_raw = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer_delta  = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # final attention pooling queries is a learnable parameter with random inital values

        self.attn_pool_queries_raw = nn.Parameter(torch.randn(dim))
        self.attn_pool_raw = Attention(dim=dim, dim_head=dim_head, heads=heads)
        self.attn_pool_queries_delta = nn.Parameter(torch.randn(dim))
        self.attn_pool_delta = Attention(dim=dim, dim_head=dim_head, heads=heads)
        
        # output to logits

        self.to_latent_raw = nn.Identity()
        self.to_latent_delta = nn.Identity()

        self.mlp_head_raw = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, 2), nn.Sigmoid()
        )
        self.mlp_head_delta = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, 1), nn.Sigmoid()
        )

        self.criterion = nn.MSELoss(reduction="mean")

    def compute_v_loss(self, x_pred: Tensor, x_gt: Tensor):
        v_pred = torch.sub(x_pred[:, 1:], x_pred[:, :-1])
        v_gt = torch.sub(x_gt[:, 1:], x_gt[:, :-1])

        return self.criterion(v_pred, v_gt)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        Ix,
        group_images=False,
        group_max_seq_len=2048,
    ):
        batched_images, delta_images, x_gt, v_gt, planeids = Ix        
        B = x_gt.shape[0]
        p, c, device, has_token_dropout = (
            self.patch_size,
            self.channels,
            self.device,
            exists(self.calc_token_dropout),
        )
        #Processing batched_images
        arange = partial(torch.arange, device=device)
        pad_sequence = partial(orig_pad_sequence, batch_first=True)
        # process images into variable lengthed sequences with attention mask

        num_images = []
        batched_sequences = []
        batched_positions = []
        batched_image_ids = []
        batched_delta_sequences = []
        batched_delta_positions = []
        batched_delta_image_ids = []

        for images in batched_images:
            num_images.append(len(images))

            sequences = []
            positions = []
            image_ids = torch.empty((0,), device=device, dtype=torch.long)
            for image_id, image in enumerate(images):                
                assert image.ndim == 3 and image.shape[0] == c
                image_dims = image.shape[-2:]
                assert all(
                    [divisible_by(dim, p) for dim in image_dims]
                ), f"height and width {image_dims} of images must be divisible by patch size {p}"

                ph, pw = map(lambda dim: dim // p, image_dims)

                pos = torch.stack(
                    torch.meshgrid((arange(ph), arange(pw)), indexing="ij"), dim=-1
                )

                pos = rearrange(pos, "h w c -> (h w) c")
                seq = rearrange(image, "c (h p1) (w p2) -> (h w) (c p1 p2)", p1=p, p2=p)

                seq_len = seq.shape[-2]

                if has_token_dropout:
                    token_dropout = self.calc_token_dropout(*image_dims)
                    num_keep = max(1, int(seq_len * (1 - token_dropout)))
                    keep_indices = (
                        torch.randn((seq_len,), device=device)
                        .topk(num_keep, dim=-1)
                        .indices
                    )

                    seq = seq[keep_indices]
                    pos = pos[keep_indices]

                image_ids = F.pad(image_ids, (0, seq.shape[-2]), value=image_id)
                sequences.append(seq)
                positions.append(pos)

            batched_image_ids.append(image_ids)
            batched_sequences.append(torch.cat(sequences, dim=0))
            batched_positions.append(torch.cat(positions, dim=0))

        # derive key padding mask

        lengths = torch.tensor(
            [seq.shape[-2] for seq in batched_sequences],
            device=device,
            dtype=torch.long,
        )
        max_length = arange(lengths.amax().item())
        key_pad_mask = rearrange(lengths, "b -> b 1") <= rearrange(
            max_length, "n -> 1 n"
        )

        # derive attention mask, and combine with key padding mask from above

        batched_image_ids = pad_sequence(batched_image_ids)
        attn_mask = rearrange(batched_image_ids, "b i -> b 1 i 1") == rearrange(
            batched_image_ids, "b j -> b 1 1 j"
        )
        attn_mask = attn_mask & rearrange(key_pad_mask, "b j -> b 1 1 j")

        # combine patched images as well as the patched width / height positions for 2d positional embedding

        patches = pad_sequence(batched_sequences)
        patch_positions = pad_sequence(batched_positions)

        # need to know how many images for final attention pooling

        num_images = torch.tensor(num_images, device=device, dtype=torch.long)
        
        num_delta_images = []
        #Processing delta_images
        # Start processing delta_images starting with the second image
        for images in delta_images:
            num_delta_images.append(len(images))

            sequences = []
            positions = []
            image_ids = torch.empty((0,), device=device, dtype=torch.long)

            for image_id, image in enumerate(images):
                assert image.ndim == 3 and image.shape[0] == c
                image_dims = image.shape[-2:]
                assert all(
                    [divisible_by(dim, p) for dim in image_dims]
                ), f"height and width {image_dims} of images must be divisible by patch size {p}"

                ph, pw = map(lambda dim: dim // p, image_dims)

                pos = torch.stack(
                    torch.meshgrid((arange(ph), arange(pw)), indexing="ij"), dim=-1
                )

                pos = rearrange(pos, "h w c -> (h w) c")
                seq = rearrange(image, "c (h p1) (w p2) -> (h w) (c p1 p2)", p1=p, p2=p)

                seq_len = seq.shape[-2]

                if has_token_dropout:
                    token_dropout = self.calc_token_dropout(*image_dims)
                    num_keep = max(1, int(seq_len * (1 - token_dropout)))
                    keep_indices = (
                        torch.randn((seq_len,), device=device)
                        .topk(num_keep, dim=-1)
                        .indices
                    )

                    seq = seq[keep_indices]
                    pos = pos[keep_indices]

                image_ids = F.pad(image_ids, (0, seq.shape[-2]), value=image_id)
                sequences.append(seq)
                positions.append(pos)

            batched_delta_image_ids.append(image_ids)
            batched_delta_sequences.append(torch.cat(sequences, dim=0))
            batched_delta_positions.append(torch.cat(positions, dim=0))
        
        lengths_delta = torch.tensor(
            [seq.shape[-2] for seq in batched_delta_sequences],
            device=device,
            dtype=torch.long,
        )
        max_length_delta = arange(lengths_delta.amax().item())
        key_pad_mask_delta = rearrange(lengths_delta, "b -> b 1") <= rearrange(
            max_length_delta, "n -> 1 n"
        )
        batched_delta_image_ids = pad_sequence(batched_delta_image_ids)
        attn_mask_delta = rearrange(batched_delta_image_ids, "b i -> b 1 i 1") == rearrange(
            batched_delta_image_ids, "b j -> b 1 1 j"
        )
        attn_mask_delta = attn_mask_delta & rearrange(key_pad_mask_delta, "b j -> b 1 1 j")
        patches_delta = pad_sequence(batched_delta_sequences)
        patch_positions_delta = pad_sequence(batched_delta_positions)
        num_delta_images = torch.tensor(num_delta_images, device=device, dtype=torch.long)
        

        # to patches

        x_raw = self.to_patch_embedding_raw(patches)
        x_delta = self.to_patch_embedding_delta(patches_delta)

        # factorized 2d absolute positional embedding

        h_indices, w_indices = patch_positions.unbind(dim=-1)
        h_indices_delta, w_indices_delta = patch_positions_delta.unbind(dim=-1)

        h_pos = self.pos_embed_height_raw[h_indices]
        w_pos = self.pos_embed_width_raw[w_indices]
        h_pos_delta = self.pos_embed_height_delta[h_indices_delta]
        w_pos_delta = self.pos_embed_width_delta[w_indices_delta]

        x_raw = x_raw + h_pos + w_pos
        x_delta = x_delta + h_pos_delta + w_pos_delta

        # embed dropout

        x_raw = self.dropout_raw(x_raw)
        x_delta = self.dropout_delta(x_delta)
        

        # attention

        x_raw = self.transformer_raw(x_raw, attn_mask=attn_mask)
        x_delta = self.transformer_delta(x_delta, attn_mask=attn_mask_delta)

        # do attention pooling at the end

        max_queries_raw = num_images.amax().item()
        max_queries_delta = num_delta_images.amax().item()

        queries_raw = repeat(
            self.attn_pool_queries_raw, "d -> b n d", n=max_queries_raw, b=x_raw.shape[0]
        )
        queries_delta = repeat(
            self.attn_pool_queries_delta, "d -> b n d", n=max_queries_delta, b=x_delta.shape[0]
        )
        

        # attention pool mask

        image_id_arange_raw = arange(max_queries_raw)
        image_id_arange_delta = arange(max_queries_delta)

        attn_pool_mask_raw = rearrange(image_id_arange_raw, "i -> i 1") == rearrange(
            batched_image_ids, "b j -> b 1 j"
        )
        attn_pool_mask_delta = rearrange(image_id_arange_delta, "i -> i 1") == rearrange(
            batched_delta_image_ids, "b j -> b 1 j"
        )
        
        attn_pool_mask_raw = attn_pool_mask_raw & rearrange(key_pad_mask, "b j -> b 1 j")
        attn_pool_mask_delta = attn_pool_mask_delta & rearrange(key_pad_mask_delta, "b j -> b 1 j")

        attn_pool_mask_raw = rearrange(attn_pool_mask_raw, "b i j -> b 1 i j")
        attn_pool_mask_delta = rearrange(attn_pool_mask_delta, "b i j -> b 1 i j")

        # attention pool

        x_raw = self.attn_pool_raw(queries_raw, context=x_raw, attn_mask=attn_pool_mask_raw) + queries_raw
        x_delta = self.attn_pool_delta(queries_delta, context=x_delta, attn_mask=attn_pool_mask_delta) + queries_delta

        x_raw = rearrange(x_raw, "b n d -> (b n) d")
        x_delta  = rearrange(x_delta , "b n d -> (b n) d")

        # each batch element may not have same amount of images

        is_images = image_id_arange_raw < rearrange(num_images, "b -> b 1")
        is_images = rearrange(is_images, "b n -> (b n)")
        is_images_delta = image_id_arange_delta < rearrange(num_delta_images, "b -> b 1")
        is_images_delta = rearrange(is_images_delta, "b n -> (b n)")

        x_raw = x_raw[is_images]
        x_delta = x_delta[is_images_delta]

        # project out to logits

        x_raw = self.to_latent_raw(x_raw)
        x_delta = self.to_latent_delta(x_delta)

        x_raw = rearrange(x_raw, "(b t) d -> b t d", b=B)
        x_delta = rearrange(x_delta, "(b t) d -> b t d", b=B)
        x_pred = self.mlp_head_raw(x_raw)
        v_pred = self.mlp_head_delta(x_delta)
        
        loss_x = self.criterion(x_pred, x_gt)
        loss_v = self.criterion(v_pred, v_gt)
        
        
        return (loss_x, loss_v), x_pred

    def vis_forward(self, Ix: tuple, **kwargs):
        return self.forward(Ix)[1].detach().cpu()


