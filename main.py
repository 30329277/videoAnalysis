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
        if predictions[0]['labels'][pred] == 17 and predictions[0]['scores'][pred] > threshold:  # COCO 数据集中 'cat' 的类别 ID 是 17
            return True
    return False

def frame_generator(video_path, target_width=640, target_height=480):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 调整帧的大小
        frame = cv2.resize(frame, (target_width, target_height))
        yield int(cap.get(cv2.CAP_PROP_POS_FRAMES)), frame
    cap.release()

def detect_cat_in_video(video_path, model, transform, num_threads=4, threshold=0.8):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cat_detected_frames = []
    lock = Lock()
    processed_frames = 0
    progress_lock = Lock()

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

    start_time = time.time()  # 记录开始时间

    frame_queue = Queue(maxsize=num_threads * 10)  # 设置队列的最大大小

    # 创建和启动线程
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for _ in range(num_threads):
            executor.submit(worker, frame_queue)

        # 逐帧读取并放入队列
        for frame_num, frame in frame_generator(video_path):
            frame_queue.put((frame_num, frame))

            # 显示进度
            with progress_lock:
                current_progress = (processed_frames / total_frames) * 100
            print(f"Processing frames: {processed_frames}/{total_frames} ({current_progress:.2f}%)", end='\r')

    # 确保所有任务完成
    frame_queue.join()

    # 确保最后一次更新进度
    with progress_lock:
        current_progress = (processed_frames / total_frames) * 100
    print(f"Processing frames: {processed_frames}/{total_frames} ({current_progress:.2f}%)")

    end_time = time.time()  # 记录结束时间
    print(f"\nVideo analysis completed. Time taken: {end_time - start_time:.2f} seconds")

    return sorted(cat_detected_frames), fps

def write_results_to_file(cat_detected_frames, fps, output_file='output.txt'):
    with open(output_file, 'w') as f:
        for i, frame_num in enumerate(cat_detected_frames):
            start_time = frame_num / fps
            end_time = (frame_num + 1) / fps  # 假设猫只在这一帧出现
            duration = end_time - start_time
            f.write(f"Cat detected at {start_time:.2f}s, duration: {duration:.2f}s\n")

if __name__ == "__main__":
    video_path = "C:\\Users\\1\\Videos\\movie01.mp4"
    cat_frames, video_fps = detect_cat_in_video(video_path, model, transform, num_threads=4)
    write_results_to_file(cat_frames, video_fps)