from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/detect/train/weights/best.pt")
    results = model.predict(source="./testData/zebra.jpg", show=True, save=True, name="output")

