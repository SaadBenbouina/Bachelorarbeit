import cv2
import numpy as np
from ultralytics import YOLO
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch


def draw_yolo_detections(frame, yolo_result, yolo_model, detection_labels):
    detected = False
    detections = yolo_result[0]

    for i, box in enumerate(detections.boxes):
        class_id = int(box.cls[0])
        label = yolo_model.names[class_id]
        if label in detection_labels:
            detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            label_text = f'{label} {confidence:.2f}'

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put the label and confidence score above the bounding box
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10

            # Draw the text background for better visibility
            cv2.rectangle(frame, (text_x, text_y - text_size[1]),
                          (text_x + text_size[0], text_y), (0, 255, 0), -1)

            # Put the text (label and confidence score)
            cv2.putText(frame, label_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Ausgabe der Koordinaten im Terminal
            print(f"Detected {label} with confidence {confidence:.2f} at coordinates: ({x1}, {y1}), ({x2}, {y2})")


    return detected


def apply_panoptic_segmentation(frame, panoptic_result):
    panoptic_seg = panoptic_result["panoptic_seg"][0].cpu().numpy()
    segments_info = panoptic_result["panoptic_seg"][1]

    for segment in segments_info:
        mask = panoptic_seg == segment['id']
        color = np.random.randint(0, 255, (1, 3), dtype=np.uint8).tolist()[0]
        frame[mask] = frame[mask] * 0.5 + np.array(color) * 0.5


def setup_panoptic_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    panoptic_predictor = DefaultPredictor(cfg)
    return panoptic_predictor


def process_media(media_path, yolo_model, panoptic_model, detection_labels=None):
    if detection_labels is None:
        detection_labels = []

    is_video = media_path.endswith(('.mp4', '.avi', '.mov', '.mkv'))

    detected = False

    if is_video:
        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {media_path}")
            return detected

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video loaded: {frame_width}x{frame_height} at {fps} FPS with {total_frames} frames")

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            yolo_result = yolo_model.predict(frame)
            panoptic_result = panoptic_model(frame)

            detected |= draw_yolo_detections(frame, yolo_result, yolo_model, detection_labels)
            apply_panoptic_segmentation(frame, panoptic_result)

            cv2.imshow('Processed Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(f"Processing frame {frame_count}/{total_frames}")
            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

    else:
        image = cv2.imread(media_path)
        if image is None:
            print(f"Error: Could not load image {media_path}")
            return detected
        print(f"Image loaded: {image.shape}")

        yolo_result = yolo_model.predict(image)
        panoptic_result = panoptic_model(image)

        detected = draw_yolo_detections(image, yolo_result, yolo_model, detection_labels)
        apply_panoptic_segmentation(image, panoptic_result)

        cv2.imshow('Processed Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detected


def main():
    media_path = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/TrainDataYolo/train/image_1335717_0_jpg.rf.d44d2d834be0eca702ff7655d000d661.jpg"
    yolo_model = YOLO("yolov8n.pt")  # build a new model from scratch

    panoptic_model = setup_panoptic_model()  # Set up the panoptic segmentation model

    detection_labels = ["boat"]  # Labels for detection

    process_media(media_path, yolo_model, panoptic_model, detection_labels)


if __name__ == '__main__':
    main()
