import argparse
import os
from datetime import datetime

import cv2
from ultralytics import YOLO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    # ----------------------------- #
    if args.save:
        if args.save_path is not None:
            base_folder = args.save_path
        else:
            base_folder = os.path.join("runs", "detect", "output")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join(base_folder, timestamp)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None

    # ----------------------------- #
    model = YOLO(args.model)

    ext = os.path.splitext(args.source)[1].lower()
    is_video = False
    if ext == ".mp4" or ext == ".avi" or ext == ".mov" or ext == ".mkv":
        is_video = True
    # ----------------------------- #
    if not is_video:

        original_image = cv2.imread(args.source)
        output_image = model.predict(original_image)[0].plot()

        height = output_image.shape[0]
        width = output_image.shape[1]
        scale = 480 / height

        display_image = cv2.resize(output_image, (int(width * scale), 480))
        cv2.imshow("Detection", display_image)

        if args.save:
            save_filename = timestamp + ".jpg"
            out_path = os.path.join(save_dir, save_filename)
            cv2.imwrite(out_path, output_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # ----------------------------- #
    else:

        cap = cv2.VideoCapture(args.source)
        writer = None

        if args.save:

            fps_value = cap.get(cv2.CAP_PROP_FPS)
            if fps_value <= 0:
                fps_value = 24

            success, first_frame = cap.read()
            frame_height = first_frame.shape[0]
            frame_width = first_frame.shape[1]

            save_filename = timestamp + ".mp4"
            out_path = os.path.join(save_dir, save_filename)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                out_path, fourcc, fps_value, (frame_width, frame_height)
            )

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            success, frame = cap.read()
            if not success:
                break

            detection = model.predict(frame)[0]
            output_image = detection.plot()

            height = output_image.shape[0]
            width = output_image.shape[1]
            scale = 480 / height

            display_image = cv2.resize(output_image, (int(width * scale), 480))
            cv2.imshow("Detection", display_image)

            if writer is not None:
                writer.write(output_image)

            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
