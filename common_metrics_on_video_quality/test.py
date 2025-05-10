import torch
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips

import json
import os
import cv2 
import numpy as np
from tqdm import tqdm

# --- 配置参数 ---
original_video_dir = "/root/xDiT/sample_videos_old"
generated_video_dir = "/root/xDiT/sample_videos_binary"
output_filename = "evaluation_results_full_length.json"

##  如果设为0或负数，则评估所有找到的视频对
NUMBER_OF_VIDEOS = 200
VIDEO_LENGTH = 49
CHANNEL = 3                  
TARGET_SIZE_HW = (480, 720) 
device = torch.device("cuda")

def load_and_preprocess_videos(video_paths):
    final_batch_tensor = torch.zeros((NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, TARGET_SIZE_HW[0], TARGET_SIZE_HW[1]), requires_grad=False)
    for i, video_path in enumerate(tqdm(video_paths, desc=f"Loading videos ({VIDEO_LENGTH})")):
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}. Creating placeholder.")
            continue
        cap = cv2.VideoCapture(video_path)
        count = 0
        while cap.isOpened() and count < VIDEO_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (TARGET_SIZE_HW[0], TARGET_SIZE_HW[1]), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            frame_chw = np.transpose(frame_normalized, (2, 1, 0))
            final_batch_tensor[i, count] = torch.from_numpy(frame_chw)
            count += 1
        cap.release()

        if count == 0:  # If no frames were read
            print(f"Warning: Could not read any frames from {video_path}. Using placeholder.")
            continue
            
        if count < VIDEO_LENGTH:
            print(f"Info: Video {video_path} had {count} frames, "
                  f"padding to {VIDEO_LENGTH} frames with the last frame.")
            last_frame = final_batch_tensor[i, count-1]
            final_batch_tensor[i, count:] = last_frame.unsqueeze(0).expand(VIDEO_LENGTH - count, -1, -1, -1)
    return final_batch_tensor

all_original_files = sorted([os.path.join(original_video_dir, f) 
                             for f in os.listdir(original_video_dir) 
                             if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))])
all_generated_files = sorted([os.path.join(generated_video_dir, f) 
                              for f in os.listdir(generated_video_dir) 
                              if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))])

num_available_pairs = min(len(all_original_files), len(all_generated_files))
if NUMBER_OF_VIDEOS <= 0:
    actual_num_to_evaluate = num_available_pairs
else:
    actual_num_to_evaluate = min(NUMBER_OF_VIDEOS, num_available_pairs)
if actual_num_to_evaluate == 0:
    raise ValueError("No video pairs available or selected for evaluation.")

original_video_files = all_original_files[:actual_num_to_evaluate]
generated_video_files = all_generated_files[:actual_num_to_evaluate]

print(f"Selected {actual_num_to_evaluate} video pairs for evaluation.")

print(f"Loading original videos from: {original_video_dir}")
videos1_list = load_and_preprocess_videos(original_video_files)
print(f"Loaded {videos1_list.shape[0]} original videos. Shape: {videos1_list.shape}")

print(f"Loading generated videos from: {generated_video_dir}")
videos2_list = load_and_preprocess_videos(generated_video_files)
print(f"Loaded {videos2_list.shape[0]} generated videos. Shape: {videos2_list.shape}")

if videos1_list.shape[0] != actual_num_to_evaluate or videos2_list.shape[0] != actual_num_to_evaluate:
    print(f"Warning: Expected to load {actual_num_to_evaluate} pairs, "
          f"but got videos1_list.shape[0]={videos1_list.shape[0]} and videos2_list.shape[0]={videos2_list.shape[0]}. "
          "This might happen if some videos failed to load and were replaced by placeholders.")

videos1_list = videos1_list.requires_grad_(False)
videos2_list = videos2_list.requires_grad_(False)

print(f"Using device: {device}")
if videos1_list.numel() > 0 :
    print(f"Pixel value range for videos1_list: min={videos1_list.min().item()}, max={videos1_list.max().item()}")
    assert videos1_list.min() >= 0.0 and videos1_list.max() <= 1.0, "videos1_list pixel values out of [0,1] range!"
if videos2_list.numel() > 0 :
    print(f"Pixel value range for videos2_list: min={videos2_list.min().item()}, max={videos2_list.max().item()}")
    assert videos2_list.min() >= 0.0 and videos2_list.max() <= 1.0, "videos2_list pixel values out of [0,1] range!"

result = {}
only_final = True 

print("\nCalculating metrics...")
print(f"Calculating FVD for {VIDEO_LENGTH} frames...")
result['fvd'] = calculate_fvd(videos1_list, videos2_list, device, method='styleganv', only_final=only_final)
print(f"Calculating SSIM for {VIDEO_LENGTH} frames...")
# result['ssim'] = calculate_ssim(videos1_list, videos2_list, only_final=only_final)
print(f"Calculating PSNR for {VIDEO_LENGTH} frames...")
result['psnr'] = calculate_psnr(videos1_list, videos2_list, only_final=only_final)
print(f"Calculating LPIPS for {VIDEO_LENGTH} frames...")
result['lpips'] = calculate_lpips(videos1_list, videos2_list, device, only_final=only_final)

print("\n--- Results ---")
for key, value in result.items():
    print(f"{key}: {value}")

with open(output_filename, "w") as f:
    json.dump(result, f, indent=4)
print(f"\nResults saved to {output_filename}")