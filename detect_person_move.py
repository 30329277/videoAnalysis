import torch
from torchvision import models, transforms
import cv2
import time
import concurrent.futures
from queue import Queue
from threading import Lock
import numpy as np
import subprocess
import os

# 加载预训练的Faster R-CNN模型
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# COCO数据集中类别的名称列表
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 定义图像转换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

def process_frame(frame, model, transform, target_label_id, threshold=0.8):
    frame_tensor = transform(frame).unsqueeze(0)
    with torch.no_grad():
        predictions = model(frame_tensor)

    people_boxes = []
    for pred in range(len(predictions[0]['labels'])):
        label_id = predictions[0]['labels'][pred].item()
        if label_id == target_label_id and predictions[0]['scores'][pred] > threshold:
            people_boxes.append(predictions[0]['boxes'][pred].cpu().numpy())
    
    return people_boxes

def frame_generator(video_path, target_width=640, target_height=480, interval=30):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_per_interval = int(fps * interval)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_width, target_height))
        yield int(cap.get(cv2.CAP_PROP_POS_FRAMES)), frame
        
        # 跳到下一个间隔
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frames_per_interval - 1)

    cap.release()
def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}m {seconds}s"    

def estimate_remaining_time(start_time, processed_frames, total_frames, fps):
    elapsed_time = time.time() - start_time
    avg_time_per_frame = elapsed_time / processed_frames if processed_frames > 0 else 0
    remaining_frames = total_frames - processed_frames
    remaining_time = remaining_frames * avg_time_per_frame
    return remaining_time

def calculate_motion(previous_boxes, current_boxes, movement_threshold=10):
    def box_center(box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2

    if not previous_boxes or not current_boxes:
        return False

    for prev_box in previous_boxes:
        prev_center = box_center(prev_box)
        for curr_box in current_boxes:
            curr_center = box_center(curr_box)
            distance = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
            if distance > movement_threshold:
                return True
    
    return False

def convert_to_mp4(video_path):
    """Converts video to MP4 format using FFmpeg."""
    base, ext = os.path.splitext(video_path)
    output_path = base + ".mp4"
    ffmpeg_path = "ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe"  # 使用相对路径

    try:
        subprocess.run([ffmpeg_path, '-i', video_path, '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', output_path, '-y'], check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {e}")
        return None

def detect_people_in_video(video_path, model, transform, target_label_id, interval=30, num_threads=4, threshold=0.8):
    # Check if the video is MTS and convert if necessary
    if video_path.lower().endswith(".mts"):
        print("MTS file detected. Converting to MP4...")
        converted_path = convert_to_mp4(video_path)
        if converted_path:
            video_path = converted_path
        else:
            print("Failed to convert MTS file. Exiting.")
            return -1, -1, -1


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    detected_times = 0
    active_times = 0
    lock = Lock()
    processed_frames = 0
    previous_boxes = None
    progress_lock = Lock()
    start_time = time.time()

    def worker(frame_queue):
        nonlocal processed_frames, detected_times, active_times, previous_boxes
        while True:
            try:
                frame_num, frame = frame_queue.get(timeout=1)
                current_boxes = process_frame(frame, model, transform, target_label_id, threshold)
                
                # 如果检测到“人”，累加检测时长
                if current_boxes:
                    with lock:
                        detected_times += interval  # 累加有“人”的时长

                    # 检查人是否在活动
                    if previous_boxes and calculate_motion(previous_boxes, current_boxes):
                        with lock:
                            active_times += interval  # 累加活动的时长
                    
                    previous_boxes = current_boxes

                with progress_lock:
                    processed_frames += 1
                frame_queue.task_done()
            except Queue.Empty:
                break

    frame_queue = Queue(maxsize=num_threads * 10)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for _ in range(num_threads):
            executor.submit(worker, frame_queue)

        try:
            for frame_num, frame in frame_generator(video_path, interval=interval):
                frame_queue.put((frame_num, frame))

                with progress_lock:
                    current_progress = (processed_frames / total_frames) * 100
                    remaining_time = estimate_remaining_time(start_time, processed_frames, total_frames, fps)
                    remaining_time_formatted = format_time(remaining_time)
                    print(f"Processing frames: {processed_frames}/{total_frames} ({current_progress:.2f}%), Estimated time remaining: {remaining_time:.2f}s ({remaining_time_formatted})", end='\r')
        except Exception as e:
            print(f"An error occurred during processing: {e}")
            return -1, -1, -1

    frame_queue.join()


    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nProcessing frames: {processed_frames}/{total_frames} (100.00%)")
    print(f"Video analysis completed. Time taken: {total_time:.2f} seconds ({format_time(total_time)})")

    video_duration_formatted = format_time(video_duration)
    detected_percentage = (detected_times / video_duration) * 100 if video_duration > 0 else 0
    active_percentage = (active_times / video_duration) * 100 if video_duration > 0 else 0

    print(f"Total time with people detected: {detected_times} seconds ({format_time(detected_times)}), ({detected_percentage:.2f}% of video duration {video_duration_formatted}).")
    print(f"Total time with active people detected: {active_times} seconds ({format_time(active_times)}), ({active_percentage:.2f}% of video duration {video_duration_formatted}).")

    return detected_times, active_times, fps

if __name__ == "__main__":
    video_path = "data/02978.MTS"
    interval = 30  # 每隔30秒取一帧进行分析
    target_label_id = 1  # 目标类别ID（1代表'person'）
    detected_times, active_times, video_fps = detect_people_in_video(video_path, model, transform, target_label_id, interval=interval, num_threads=4)
