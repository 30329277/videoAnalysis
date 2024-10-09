import torch
from torchvision import models, transforms
import cv2
import time
import concurrent.futures
from queue import Queue
from threading import Lock

# 加载预训练的Faster R-CNN模型
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 定义图像转换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

def process_frame(frame, model, transform, threshold=0.8):
    frame_tensor = transform(frame).unsqueeze(0)
    with torch.no_grad():
        predictions = model(frame_tensor)
    
    for pred in range(len(predictions[0]['labels'])):
        # if predictions[0]['labels'][pred] == 17 and predictions[0]['scores'][pred] > threshold:  # COCO 数据集中 'cat' 的类别 ID 是 17
        if predictions[0]['labels'][pred] == 3 and predictions[0]['scores'][pred] > threshold:  # COCO 数据集中 'cat' 的类别 ID 是 17
            return True
    return False

def frame_generator(video_path, target_width=640, target_height=480):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_width, target_height))
        yield int(cap.get(cv2.CAP_PROP_POS_FRAMES)), frame
    cap.release()

def estimate_remaining_time(start_time, processed_frames, total_frames, fps):
    elapsed_time = time.time() - start_time
    avg_time_per_frame = elapsed_time / processed_frames if processed_frames > 0 else 0
    remaining_frames = total_frames - processed_frames
    remaining_time = remaining_frames * avg_time_per_frame
    return remaining_time

def detect_cat_in_video(video_path, model, transform, num_threads=4, threshold=0.8):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cat_detected_frames = []
    lock = Lock()
    processed_frames = 0
    progress_lock = Lock()
    start_time = time.time()

    def worker(frame_queue):
        nonlocal processed_frames
        while True:
            try:
                frame_num, frame = frame_queue.get(timeout=1)
                if process_frame(frame, model, transform, threshold):
                    with lock:
                        cat_detected_frames.append(frame_num)
                with progress_lock:
                    processed_frames += 1
                frame_queue.task_done()
            except Queue.Empty:
                break

    frame_queue = Queue(maxsize=num_threads * 10)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for _ in range(num_threads):
            executor.submit(worker, frame_queue)

        for frame_num, frame in frame_generator(video_path):
            frame_queue.put((frame_num, frame))

            with progress_lock:
                current_progress = (processed_frames / total_frames) * 100
                remaining_time = estimate_remaining_time(start_time, processed_frames, total_frames, fps)
                print(f"Processing frames: {processed_frames}/{total_frames} ({current_progress:.2f}%), Estimated time remaining: {remaining_time:.2f}s", end='\r')

    frame_queue.join()

    with progress_lock:
        current_progress = (processed_frames / total_frames) * 100
    print(f"\nProcessing frames: {processed_frames}/{total_frames} ({current_progress:.2f}%)")
    end_time = time.time()
    print(f"Video analysis completed. Time taken: {end_time - start_time:.2f} seconds")

    cat_percentage = (len(cat_detected_frames) / total_frames) * 100
    print(f"Cat detected in {len(cat_detected_frames)} frames, which is {cat_percentage:.2f}% of the total video duration.")

    return sorted(cat_detected_frames), fps

def write_results_to_file(cat_detected_frames, fps, output_file='output.txt'):
    with open(output_file, 'w') as f:
        for i, frame_num in enumerate(cat_detected_frames):
            start_time = frame_num / fps
            end_time = (frame_num + 1) / fps
            duration = end_time - start_time
            f.write(f"Cat detected at {start_time:.2f}s, duration: {duration:.2f}s\n")

if __name__ == "__main__":
    video_path = "data\\Media1.mp4"
    cat_frames, video_fps = detect_cat_in_video(video_path, model, transform, num_threads=4)
    write_results_to_file(cat_frames, video_fps)