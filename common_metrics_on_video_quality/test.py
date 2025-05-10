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
original_video_dir = "/workspace/common_metrics_on_video_quality/video/original"
generated_video_dir = "/workspace/common_metrics_on_video_quality/video/test"

##  如果设为0或负数，则评估所有找到的视频对
num_videos_to_evaluate = 1 
# 你希望的视频输出分辨率 (H, W)                        
TARGET_SIZE_HW = (480, 720) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------

def get_video_actual_length(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path} to get length.")
        return 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length

def load_and_preprocess_videos(video_paths, target_length_for_all, target_size_hw, device):
    """
    加载并预处理一批视频，确保所有视频输出为 target_length_for_all 帧。
    """
    batch_videos = []
    
    if isinstance(target_size_hw, int):
        target_height, target_width = target_size_hw, target_size_hw
    else:
        target_height, target_width = target_size_hw

    for video_path in tqdm(video_paths, desc=f"Loading videos (target length: {target_length_for_all})"):
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}. Creating placeholder.")
            placeholder_frame = np.zeros((3, target_height, target_width), dtype=np.float32)
            video_frames = [placeholder_frame] * target_length_for_all
            video_tensor_np = np.stack(video_frames, axis=0)
            batch_videos.append(video_tensor_np)
            continue

        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        actual_frames_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened() and count < target_length_for_all:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            frame_chw = np.transpose(frame_normalized, (2, 0, 1))
            frames.append(frame_chw)
            count += 1
        cap.release()

        if not frames: # 如果一帧都读不到
            print(f"Warning: Could not read any frames from {video_path}. Using placeholder.")
            placeholder_frame = np.zeros((3, target_height, target_width), dtype=np.float32)
            video_frames = [placeholder_frame] * target_length_for_all
            video_tensor_np = np.stack(video_frames, axis=0)
            batch_videos.append(video_tensor_np)
            continue
         
        num_read_frames = len(frames)
        if num_read_frames < target_length_for_all:
            print(f"Info: Video {video_path} had {num_read_frames} frames, "
                  f"padding to {target_length_for_all} frames with the last frame.")
            last_frame = frames[-1] # 取最后一帧进行填充
            padding = [last_frame] * (target_length_for_all - num_read_frames)
            frames.extend(padding)
        elif num_read_frames > target_length_for_all:
            frames = frames[:target_length_for_all]
            
        video_tensor_np = np.stack(frames, axis=0)
        batch_videos.append(video_tensor_np)

    if not batch_videos: 
        if video_paths: # 只有当有尝试加载的视频时
            print("Warning: All videos failed to load. Returning a batch of placeholders.")
            placeholder_frame = np.zeros((3, target_height, target_width), dtype=np.float32)
            video_frames_placeholder = [placeholder_frame] * target_length_for_all
            video_tensor_placeholder = np.stack(video_frames_placeholder, axis=0)
            batch_videos = [video_tensor_placeholder] * len(video_paths)
            final_batch_tensor_np = np.stack(batch_videos, axis=0)
            return torch.from_numpy(final_batch_tensor_np).to(device)
        else: # 如果 video_paths 本身就是空的
            return torch.empty((0, target_length_for_all, 3, target_height, target_width), device=device)


    final_batch_tensor_np = np.stack(batch_videos, axis=0)
    final_batch_tensor = torch.from_numpy(final_batch_tensor_np).to(device)
    
    return final_batch_tensor

# 获取视频文件列表
all_original_files = sorted([os.path.join(original_video_dir, f) 
                             for f in os.listdir(original_video_dir) 
                             if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))])
all_generated_files = sorted([os.path.join(generated_video_dir, f) 
                              for f in os.listdir(generated_video_dir) 
                              if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))])

# 确定评估视频对
num_available_pairs = min(len(all_original_files), len(all_generated_files))

if num_videos_to_evaluate <= 0:
    actual_num_to_evaluate = num_available_pairs
else:
    actual_num_to_evaluate = min(num_videos_to_evaluate, num_available_pairs)

if actual_num_to_evaluate == 0:
    raise ValueError("No video pairs available or selected for evaluation.")

original_video_files = all_original_files[:actual_num_to_evaluate]
generated_video_files = all_generated_files[:actual_num_to_evaluate]

print(f"Selected {actual_num_to_evaluate} video pairs for evaluation.")

# 确定帧数量
VIDEO_LENGTH_ACTUAL = 0
if original_video_files: 
    first_video_path = original_video_files[0]
    VIDEO_LENGTH_ACTUAL = get_video_actual_length(first_video_path)
    if VIDEO_LENGTH_ACTUAL == 0:
        raise ValueError(f"Could not determine the length of the first video: {first_video_path}. "
                         "Please ensure it's a valid video file.")
    print(f"Determined video length from first original video: {VIDEO_LENGTH_ACTUAL} frames.")
else:
    raise ValueError("No original video files found to determine video length.")

# 加载
print(f"Loading original videos from: {original_video_dir}")
videos1_list = load_and_preprocess_videos(original_video_files, VIDEO_LENGTH_ACTUAL, TARGET_SIZE_HW, device)
print(f"Loaded {videos1_list.shape[0]} original videos. Shape: {videos1_list.shape}")

print(f"Loading generated videos from: {generated_video_dir}")
videos2_list = load_and_preprocess_videos(generated_video_files, VIDEO_LENGTH_ACTUAL, TARGET_SIZE_HW, device)
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
# 因为我们要评估整个视频，所以用 True
only_final = True 

print("\nCalculating metrics...")
if VIDEO_LENGTH_ACTUAL > 10:
    print(f"Calculating FVD for {VIDEO_LENGTH_ACTUAL} frames...")
    result['fvd'] = calculate_fvd(videos1_list, videos2_list, device, method='styleganv', only_final=only_final)
else:
    print(f"Skipping FVD as video length ({VIDEO_LENGTH_ACTUAL}) is not > 10.")
    result['fvd'] = {"value": [f"N/A (video length {VIDEO_LENGTH_ACTUAL} too short)"]}

print(f"Calculating SSIM for {VIDEO_LENGTH_ACTUAL} frames...")
result['ssim'] = calculate_ssim(videos1_list, videos2_list, only_final=only_final)
print(f"Calculating PSNR for {VIDEO_LENGTH_ACTUAL} frames...")
result['psnr'] = calculate_psnr(videos1_list, videos2_list, only_final=only_final)
print(f"Calculating LPIPS for {VIDEO_LENGTH_ACTUAL} frames...")
result['lpips'] = calculate_lpips(videos1_list, videos2_list, device, only_final=only_final)

print("\n--- Results ---")
print(json.dumps(result, indent=4))

output_filename = "evaluation_results_full_length.json"
with open(output_filename, "w") as f:
    json.dump(result, f, indent=4)
print(f"\nResults saved to {output_filename}")