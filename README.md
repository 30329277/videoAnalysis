# Project README

## Overview
This project aims to detect frames within a video that contain cats using a pre-trained Faster R-CNN model. It leverages the `torchvision` models provided by the PyTorch library and employs multi-threading for faster video analysis.

## Features
- **Cat Detection**: The program identifies frames in a video where a cat is present.
- **Multi-Threading**: Utilizes Python's `concurrent.futures` to process video frames in parallel, improving the speed of the detection process.
- **Progress Reporting**: Displays the progress of the video analysis, including the estimated remaining time.
- **Output File Generation**: Writes the timestamps of detected cat appearances into an output file.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- OpenCV (cv2)
- NumPy (used implicitly by other libraries)

## Installation
To install the required packages, you can use pip:
```bash
pip install torch torchvision opencv-python
```

## Code Explanation
### Imports
The script starts with importing necessary libraries, including PyTorch for the model, torchvision for pre-trained models and transformations, and OpenCV for handling video input.

### Model Initialization
- A pre-trained Faster R-CNN model is loaded from `torchvision.models.detection`.
- The model is set to evaluation mode.

### Frame Processing
- A function `process_frame` is defined to handle the transformation of each frame into a tensor and then passing it through the model to get predictions.
- The function specifically looks for predictions with a high confidence score (above a threshold) and checks if the predicted class corresponds to a cat (COCO dataset class ID 17).

### Video Handling
- `frame_generator` is a generator function that reads frames from the video, resizes them, and yields them one by one.
- `detect_cat_in_video` is the main function that initializes the video capture, sets up threading, and processes the frames.
- It also prints out the processing progress and estimates the remaining time.

### Multi-Threading
- A thread pool is created to distribute the frame processing across multiple threads.
- Each thread takes frames from a queue, processes them, and updates the shared data structures in a thread-safe manner.

### Output
- After all frames have been processed, the script calculates the percentage of frames where a cat was detected.
- The results, including the start times and durations of the detected cat appearances, are written to an output file.

## Usage
To run the script, simply execute it with Python, ensuring that the path to the video file is correctly specified. An example of running the script is as follows:
```python
if __name__ == "__main__":
    video_path = "data\\Media1.mp4"
    cat_frames, video_fps = detect_cat_in_video(video_path, model, transform, num_threads=4)
    write_results_to_file(cat_frames, video_fps)
```
Make sure the video file exists at the specified location, and adjust the number of threads based on your system's capabilities.

## Output
The script will generate an `output.txt` file containing the timestamps of when a cat was detected in the video, along with the duration of each appearance. Additionally, it will print out the total number of frames analyzed, the percentage of frames with a cat, and the total time taken to complete the analysis.
