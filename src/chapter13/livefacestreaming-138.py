import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os

# ---------------------------------------------------------
# 1. Initialize MediaPipe
# ---------------------------------------------------------
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ---------------------------------------------------------
# 2. Try to open webcam or fallback
# ---------------------------------------------------------
video_source = None

# Try ffmpeg backend
try:
    iio.immeta("ffmpeg://0")  # probe for webcam
    video_source = "ffmpeg://0"
    print("✅ Webcam detected via ffmpeg.")
except Exception:
    # Fallback to local file if webcam not available
    if os.path.exists("sample_video.mp4"):
        video_source = "sample_video.mp4"
        print("🎞️ Using fallback video: sample_video.mp4")
    else:
        raise RuntimeError("❌ No webcam or sample video found.")

# ---------------------------------------------------------
# 3. Process video frames
# ---------------------------------------------------------
print("Press Ctrl+C to stop...")
try:
    for frame in iio.imiter(video_source):
        img_rgb = frame[:, :, :3]
        results = detector.process(img_rgb)

        if results.detections:
            for det in results.detections:
                mp_draw.draw_detection(frame, det)

        plt.imshow(frame)
        plt.axis("off")
        plt.title("Live Face Detection (No OpenCV)")
        plt.pause(0.01)
        plt.clf()
except KeyboardInterrupt:
    print("\n👋 Stream stopped by user.")
