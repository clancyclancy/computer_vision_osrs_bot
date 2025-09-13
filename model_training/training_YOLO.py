from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(BASE_DIR, "model_training\data.yaml")


def main():
    # Load a pretrained YOLOv8 model (nano version is fastest to train)
    model = YOLO("yolov8n.pt")  # or yolov8s.pt / yolov8m.pt for more accuracy

    # Train
    model.train(
        data=base_dir,
        epochs=400,  # quick tests are 20
        imgsz=256,   # trained at 320 trained at 416  usually 640
        batch=16,
        workers=6,   # lower if issues
        device=0     # 0 for first GPU
    )

if __name__ == "__main__":
    main()
