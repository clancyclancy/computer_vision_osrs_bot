import cv2
import os

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(BASE_DIR, "VIDEO.mp4")
output_folder = os.path.join(BASE_DIR, "FRAMES")
dataset_dir = os.path.join(BASE_DIR, "allDataYOLO")
frame_interval = 1                  # Save every frame (set to >1 to skip frames)

# --- Create output folder if it doesn't exist ---
os.makedirs(output_folder, exist_ok=True)

# --- Load video ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

frame_count = 0
saved_count = 0

# --- Frame extraction loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    if frame_count % frame_interval == 0:
        filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
        #cv2.imwrite(filename.replace(".jpg", ".png"), frame)
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"✅ Extracted {saved_count} frames to '{output_folder}'")