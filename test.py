import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from data.dataset import split_dataset
from utils.vis import *
from utils.tools import load_model
from utils.trainermppt import _set_seed
import warnings
from PIL import Image
plt.rcParams['figure.figsize'] = (10.0, 8.0) 

# Set cuda requirements
print(f"torch version: {torch.__version__}")
use_cuda = 0# torch.cuda.is_available()
if use_cuda:
    GPU_nums = torch.cuda.device_count()
    GPU = torch.cuda.get_device_properties(0)
    print(f"There are {GPU_nums} GPUs in total.\nThe first GPU: {GPU}")
    print(f"CUDA version: {torch.version.cuda}")
device = torch.device(f"cuda:0" if use_cuda else "cpu")
print(f"Using {device} now!")

# Load Model
path = 'DynamicNLOS/NLOSmodel'
model = load_model(path).to(device)

# Load Dataset
_set_seed(seed=0, deterministic=True)
test_d = split_dataset(
    dataset_root="testdata_real/1",  phase='test'
    )

loader_kwargs = {
    'batch_size' : 1,
    'num_workers': 4,
    'pin_memory': True,
    'prefetch_factor': 4,
    'persistent_workers': True
}
test_loader = DataLoader(test_d, shuffle=False,**loader_kwargs)
iter_test_loader = iter(test_loader)

# Test
import time

#print inference time
t = time.time()
inputs, diffimgs, planeids, labels, v_gt, map_sizes = next(iter_test_loader)
t = time.time() - t
print(f"Time taken to load data: {t:.3f}s")

inputs = [
            [torch.unbind(input_tensor, dim=0) for input_tensor in input_list]
            for input_list in inputs
        ]
frames = [
    [input_tensor for input_tensor in input_list[0]]
    for input_list in inputs
]
diffimgs = [
    [torch.unbind(input_tensor, dim=0) for input_tensor in diffimg_list]
    for diffimg_list in diffimgs
]
diffimgs = [
    [input_tensor for input_tensor in diffimg_list[0]]
    for diffimg_list in diffimgs
]

planeids = [
    [torch.unbind(input_tensor, dim=0) for input_tensor in planeid_list]
    for planeid_list in planeids
]
planeids = [
    [input_tensor for input_tensor in planeid_list[0]]
    for planeid_list in planeids
]
batch_size = 1

gt_routes = labels.squeeze(0)
v_gt = v_gt.squeeze(0)
map_sizes = map_sizes.squeeze(0)
with torch.no_grad():
    with autocast():
        pred_routes = model.vis_forward((frames, diffimgs, gt_routes, v_gt, planeids))
array1 = gt_routes[:, 0, :].expand(1, -1, -1)
pred1 = pred_routes[:, 0, :].expand(1, -1, -1)

# Plot the trajectory
for idx, (gt, pred) in enumerate(zip(array1.numpy(), pred1.numpy())):
    image_array = draw_routes((gt, pred),'fig_array')
image_array = Image.fromarray(image_array).convert("RGB")
image_array.save('predicted_trajectory.png')
print("Inference completed!")

