import argparse
import os
from datetime import datetime
from ultralytics import YOLO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"runs/train/{os.path.splitext(args.data)[0]}-{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO("yolov8s.pt")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=output_dir,
    )
