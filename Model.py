# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import argparse

from ultralytics import YOLO
import supervision as sv

def parser_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument("--webcam-resolution", default=[1080, 1920], nargs=2, type=int)

    args = parser.parse_args()
    return args

def main():
    args = parser_arguments()
    frame_w, frame_h = args.webcam_resolution

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

    model = YOLO("yolov8n.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )


    while True:
        ret, frame = cap.read()

        result = model.track(frame, persist=True)
        frame_p = result[0].plot()
        #detections = sv.Detections.from_yolov8(result)
        #detections =sv.Detections.from_ultralytics(result)
        #frame = box_annotator.annotate(scene=frame, detections=detections)

        cv2.imshow("yolov8", frame_p)

        #print(frame.shape)
        #break

        if (cv2.waitKey(30) == 27):
            break

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/