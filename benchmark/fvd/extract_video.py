import cv2
import os

def extract_keyframes(video_path, output_dir, num_keyframes=6):
    """
    从视频中提取指定数量的关键帧并保存为图片。

    Args:
        video_path (str): 视频文件的路径。
        output_dir (str): 保存关键帧图片的目录。
        num_keyframes (int): 要提取的关键帧数量。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        print("视频帧数为 0 或无法获取帧数。")
        cap.release()
        return

    # 计算提取关键帧的帧索引
    indices = [int(i * frame_count / (num_keyframes + 1)) for i in range(1, num_keyframes + 1)]

    for i, frame_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if ret:
            output_path = os.path.join(output_dir, f"keyframe_{i+1}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"已保存关键帧: {output_path}")
        else:
            print(f"无法读取帧索引: {frame_index}")

    cap.release()
    print("关键帧提取完成。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Extract keyframes from video')
    parser.add_argument('--video', type=str, default="video/df.mp4", help='Path to input video file')
    parser.add_argument('--output', type=str, default="keyframes", help='Directory to save extracted keyframes')
    args = parser.parse_args()
    
    video_file = args.video
    output_directory = args.output
    num_key_frames_to_extract = 6

    extract_keyframes(video_file, output_directory, num_key_frames_to_extract)