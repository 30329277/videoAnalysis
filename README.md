# Project README

## Project Overview
This project utilizes PyTorch and a pre-trained Faster R-CNN model to detect people in video footage and analyze the time during which people are present and active. The project employs multithreading to process video frames, enhancing processing speed, and can automatically convert MTS video files to MP4 format.

## Features
- **Person Detection**: Uses a pre-trained Faster R-CNN model to identify people in the video.
- **Activity Detection**: Calculates the movement of people between frames to determine if they are active.
- **Time Statistics**: Computes the percentage of video duration where people are detected and the percentage of time where people are active.
- **Video Format Conversion**: Automatically converts MTS video files to MP4 format.

## Environment Requirements
- Python 3.7 or higher
- PyTorch
- OpenCV
- torchvision
- numpy
- FFmpeg (for video conversion)

## Installation
Install the required dependencies using pip:
```bash
pip install torch torchvision opencv-python numpy
```

Ensure that FFmpeg is installed and `ffmpeg.exe` is available in your system's PATH, or specify the correct path to the FFmpeg executable in the code.

## Usage
1. Place your video file in the `data/` directory, or modify the `video_path` variable to point to the correct video file.
2. Run the main script:
   ```bash
   python your_script_name.py
   ```

### Configuration Parameters
- `video_path`: Path to the video file.
- `interval`: Detection interval in seconds, default is 30 seconds.
- `target_label_id`: Target class ID, corresponding to the COCO dataset class index (e.g., 1 for "person").
- `num_threads`: Number of threads to use for processing video frames, default is 4.

## Output
Upon completion, the program will output the following information:
- Total video duration
- Total time with people detected and the percentage of the total video duration
- Total time with active people detected and the percentage of the total video duration
- Total time taken to process the video

## Notes
- Ensure your machine has sufficient memory to handle large video files.
- If the video is in MTS format, the program will attempt to convert it to MP4 before processing.
- You can balance processing speed and accuracy by adjusting the `interval` parameter.

## Code Structure
- `process_frame`: Processes a single frame and returns detected person bounding boxes.
- `frame_generator`: Generator function that reads video frames at specified intervals.
- `format_time`: Formats time output.
- `estimate_remaining_time`: Estimates the remaining processing time.
- `calculate_motion`: Calculates the motion between two sets of bounding boxes.
- `convert_to_mp4`: Converts the video format to MP4.
- `detect_people_in_video`: Main processing function that orchestrates other helper functions and aggregates results.

## Contributing
Contributions are welcome! If you find any bugs or have suggestions for improvements, please submit an issue or a pull request.

---

This README provides a basic overview of the project, including how to set up the environment, run the code, and what to expect as output. Depending on your specific needs, you may want to add more details, such as more configuration options, example outputs, and additional documentation.
