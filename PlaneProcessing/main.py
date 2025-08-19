import csv
from PIL import Image
import os
import numpy as np
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
import pyransac3d as pyrsc
from models.functions.funcs import PCA_svd
from models.functions.funcs import calc_size_preserve_ar, pad_even_divided
from planerecnet import PlaneRecNet
from planerecnet_config.augmentations import FastBaseTransform
from planerecnet_config.config import set_cfg, cfg, COLORS
from collections import defaultdict, namedtuple
from utils_pr import timer
from data.augmentations import FastBaseTransform
from data.config import set_cfg, cfg, COLORS

timer.disable_all()
set_cfg("PlaneRecNet_101_config")

net = PlaneRecNet(cfg)
net.load_weights("DynamicNLOS/PlaneRecNetmodel/PlaneRecNet.pth")
net.train(mode=False)
net = net.cuda()
torch.set_default_tensor_type("torch.cuda.FloatTensor")

color_cache = defaultdict(lambda: {})

def read_camera_positions(csv_file):
    timestamps = []
    positions = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            timestamp = row[0]
            x, y, z = float(row[1]), float(row[2]), float(row[3])
            timestamps.append(timestamp)
            positions.append(np.array([x, y, z]))

    return timestamps, positions

def read_images_and_positions(image_directory, csv_file):
    timestamps, positions = read_camera_positions(csv_file)

    png_files = [f for f in os.listdir(image_directory) if f.endswith('.png')]
    png_files.sort()
    timestamps.sort()

    data_triplets = []

    prev_timestamp = None
    prev_image = None
    prev_position = None

    for png_file, timestamp, position in zip(png_files, timestamps, positions):
        input_path = os.path.join(image_directory, png_file)
        image = Image.open(input_path)

        data_triplets.append((timestamp, image, position, prev_timestamp, prev_image, prev_position))

        prev_timestamp = timestamp
        prev_image = image
        prev_position = position

    return data_triplets

def custom_segmentation_function(img, diff_img, Hom_mat, current_timestamp, net, cfg, prev_planes=None,n=3, similarity_threshold=0.5):
    if img is None:
        return

    frame_np = np.array(img)

    try:
        frame_np = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    except:
        pass

    H, W, _ = frame_np.shape

    if frame_np is None:
        return

    frame_np = cv2.resize(frame_np, calc_size_preserve_ar(W, H, cfg.max_size), interpolation=cv2.INTER_LINEAR)
    frame_np = pad_even_divided(frame_np)

    frame = torch.from_numpy(frame_np).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    results = net(batch)

    frame_gpu = frame / 255.0
    h, w, _ = frame.shape

    # Resize the difference image 
    diff_img = cv2.resize(diff_img, (w, h))

    cv2.imshow("Diff", diff_img)

    pred_scores = results[0]["pred_scores"]
    pred_depth = results[0]["pred_depth"]
    pred_masks = results[0]["pred_masks"]

    if pred_masks is not None:
        k_matrix = np.array([[W, 0, 0], [0, H, 0], [0, 0, 1]])
        k_matrix = torch.from_numpy(k_matrix).double().cuda()
        intrinsic_inv = torch.inverse(k_matrix).double().cuda()

        B, C, H, W = pred_depth.shape

        cx = k_matrix[0][2]
        cy = k_matrix[1][2]
        fx = k_matrix[0][0]
        fy = k_matrix[1][1]

        v, u = torch.meshgrid(torch.arange(H), torch.arange(W))
        Z = pred_depth.squeeze(dim=0)
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        point_cloud = torch.cat((X, Y, Z), dim=0).permute(1, 2, 0)

        N = pred_masks.shape[0]
        mask_areas = []
        for idx in range(N):
            mask = pred_masks[idx].bool()
            area = torch.sum(mask).item()
            mask_areas.append((idx, area))

        mask_areas.sort(key=lambda x: x[1], reverse=True)
        top_n = min(n, len(mask_areas))

        for plane_id, (idx, _) in enumerate(mask_areas[:top_n]):  # Ensure indices are within bounds
            mask = pred_masks[idx].bool()
            point_cloud_seg = point_cloud[mask, :].squeeze(dim=0)

            plane1 = pyrsc.Plane()
            pts = point_cloud_seg.cpu().detach().numpy()
            rand_pts = pts[np.random.choice(pts.shape[0], size=100, replace=False)]
            best_eq, best_inliers = plane1.fit(rand_pts, 0.01)

            mask_cpu = mask.cpu().numpy()
            masked_image = frame_np.copy()
            masked_image[~mask_cpu] = 0

            best_eq = np.array(best_eq)  # Convert to NumPy array


            # Check for overlap with the previous image mask
            if prev_planes is not None:
                assigned_plane_id = None
                prev_overlap_percentage = 0.0

                for prev_plane_id, prev_mask in prev_planes.items():
                    overlap = np.logical_and(mask_cpu, prev_mask)
                    overlap_percentage = np.sum(overlap) / np.sum(prev_mask)

                    if overlap_percentage > prev_overlap_percentage and overlap_percentage >= similarity_threshold:
                        assigned_plane_id = prev_plane_id
                        prev_overlap_percentage = overlap_percentage

                if assigned_plane_id is not None:
                    # print(f"Overlap percentage with previous mask for Plane {assigned_plane_id}: {prev_overlap_percentage}")

                    # Assign the plane ID
                    plane_id = assigned_plane_id
                else:
                    # Assign a new plane ID
                    plane_id = len(prev_planes)

            # Update the previous plane mask
            prev_planes[plane_id] = mask_cpu

            # Convert the previous plane mask to the current mask space using the homography matrix
            if Hom_mat is not None:
                # Assuming prev_mask is the binary mask from the previous frame
                prev_mask_warped = cv2.warpPerspective(prev_planes[plane_id].astype(np.uint8), Hom_mat, (w, h), flags=cv2.INTER_NEAREST)

                # Use the warped mask as the current mask
                mask_cpu = prev_mask_warped.astype(bool)

                # Obtain the overlap region between the current mask and the warped previous mask
                overlap_mask = np.logical_and(mask_cpu, prev_mask_warped)

                # Update the current mask to be the overlap region
                mask_cpu = overlap_mask

                # Apply the overlap mask on the original image to get the masked image
                masked_image = frame_np.copy()
                masked_image[~mask_cpu] = 0 
                masked_image = masked_image.astype(np.uint8)

                 # Apply the overlap mask on the diff image to get the masked diff image
                masked_diff_image = diff_img.copy()
                masked_diff_image[~mask_cpu] = 0 
                masked_diff_image = masked_diff_image.astype(np.uint8)

            print(f"Plane {plane_id} equation: {best_eq}")

            # # Convert the masked image from NumPy array to PIL Image
            pil_image = Image.fromarray(masked_image.astype(np.uint8))
            pil_image.save(os.path.join(processed_folder_raw, f"{current_timestamp}_{plane_id}.png"))

            # # Convert the masked image from NumPy array to PIL Image
            pil_diff_image = Image.fromarray(masked_diff_image.astype(np.uint8))
            pil_diff_image.save(os.path.join(processed_folder_diff_img, f"{current_timestamp}_{plane_id}.png"))

          
        cv2.waitKey(2)




def diff_images(prev_image, current_image):
    # Convert images to NumPy arrays
    prev_np = np.array(prev_image)
    current_np = np.array(current_image)

    # Create ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(prev_np, None)
    kp2, des2 = orb.detectAndCompute(current_np, None)

    # Use the BFMatcher to find the best matches between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort them in ascending order of distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Take the top N matches
    N = 50
    good_matches = matches[:N]

    # Get corresponding points in both images
    prev_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    current_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calculate homography
    H, _ = cv2.findHomography(prev_pts, current_pts, cv2.RANSAC, 1.0)

    # Warp the previous image to the current image space
    warped_img = cv2.warpPerspective(prev_np, H, (prev_np.shape[1], prev_np.shape[0]))

    # Find absolute difference between the images
    diff_img = cv2.absdiff(current_np, warped_img)

    return diff_img, H




if __name__ == "__main__":
    image_directory = "out_dir/raw" #sample img path
    cam_pose_csv = "out_dir/mocap/camera_pose.csv" #sample csv path

    processed_folder_raw = os.path.join("out_dir", "processed", "raw")
    processed_folder_diff_img = os.path.join("out_dir", "processed", "diff_img")
    os.makedirs(processed_folder_raw, exist_ok=True)
    os.makedirs(processed_folder_diff_img, exist_ok=True)

    triplets = read_images_and_positions(image_directory, cam_pose_csv)
    
    prev_planes = {}

    for current_timestamp, current_image, current_position, prev_timestamp, prev_image, prev_position in triplets:
        print("\n")
        print(f"<<<< ==== Current Timestamp: {current_timestamp} ==== >>>>")
        print(f"Current Camera Position:- {current_position}")
        if prev_image is not None:
            diff_img, H = diff_images(prev_image, current_image)
            custom_segmentation_function(current_image, diff_img, H, current_timestamp, net, cfg, prev_planes)


    cv2.destroyAllWindows()
