# Real-Time Face Detection, Tracking, and Auto-Zoom

A lightweight, real-time computer vision application that detects a user’s face from a webcam, tracks it continuously, and dynamically crops and zooms the video feed to keep the face centered. Designed for efficiency on standard consumer laptops.

## Features

-   **Real-Time Face Detection**: Uses **MediaPipe** for robust and accurate initial face detection.
-   **Efficient Tracking**: Utilizes **KCF (Kernelized Correlation Filters)** from OpenCV for high-speed tracking with minimal CPU overhead.
-   **Auto-Zoom & Center**: Dynamically crops the frame to center the user's face with a smooth digital zoom effect.
-   **Stabilization**: Implements **Exponential Moving Average (EMA)** smoothing to prevent jitter and provide a cinematic feel.
-   **Performance Optimized**:
    -   Processes detection and tracking on a downscaled frame (640px) to conserve resources.
    -   Applies the final high-quality crop to the original full-resolution video feed.
    -   Mirror effect for natural interaction.

## Requirements

-   Python 3.7+
-   Webcam

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Amish-03/centered_face_tracking.git
    cd centered_face_tracking
    ```

2.  **Install dependencies**:
    ```bash
    pip install opencv-python opencv-contrib-python mediapipe numpy
    ```
    *Note: `opencv-contrib-python` is recommended for the KCF tracker, though the standard `opencv-python` often includes it in newer versions.*

## Usage

Run the main script to start the application:

```bash
python main.py
```

### Controls
-   **q**: Quit the application.

## How It Works

1.  **Detection**: The system periodically scans for a face using MediaPipe.
2.  **Tracking**: Once a face is found, it initializes the KCF tracker to follow the face frame-by-frame. This is much faster than running detection on every frame.
3.  **Smoothing**: The bounding box coordinates are smoothed over time to remove jitter.
4.  **Zooming**: The application calculates a crop region centered on the face, ensuring a safe margin, and resizes it to fill the window.

## Customization

You can adjust parameters in `main.py` and `zoom.py`:

-   **`PROCESS_WIDTH`** (in `main.py`): Lower this value (e.g., 480) for even better performance on very old hardware.
-   **`smoothing_factor`** (in `main.py` -> `ZoomController`): Adjust between 0.1 (very smooth, more lag) and 0.9 (snappy, more jitter).
-   **`margin_factor`** (in `main.py` -> `ZoomController`): Controls how much space is left around the face (default 0.6 = 60%).