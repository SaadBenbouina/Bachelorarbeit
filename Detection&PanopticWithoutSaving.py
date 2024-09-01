import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch

def draw_detectron2_detections(frame, detectron2_result, detection_labels):
    detected_boats = []  # To store coordinates of detected boats

    instances = detectron2_result["instances"]
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    scores = instances.scores if instances.has("scores") else None
    classes = instances.pred_classes if instances.has("pred_classes") else None

    # Debug: Print the number of detections
    print(f"Total detections: {len(boxes)}")

    for i in range(len(boxes)):
        class_id = int(classes[i])
        label = detection_labels[class_id] if class_id < len(detection_labels) else f"Class {class_id}"
        if label in detection_labels:
            x1, y1, x2, y2 = boxes[i].tensor[0].numpy().astype(int)
            confidence = scores[i].item()
            label_text = f'{label} {confidence:.2f}'

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Append boat coordinates to the list
            detected_boats.append(((x1, y1), (x2, y2)))

    return detected_boats

def setup_detection_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the threshold for this model
    detection_predictor = DefaultPredictor(cfg)
    return detection_predictor

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

def process_media(media_path, detection_model, panoptic_model, detection_labels=None):
    if detection_labels is None:
        detection_labels = []

    is_video = media_path.endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if is_video:
        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {media_path}")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detectron2_result = detection_model(frame)
            panoptic_result = panoptic_model(frame)

            detected_boats = draw_detectron2_detections(frame, detectron2_result, detection_labels)
            apply_panoptic_segmentation(frame, panoptic_result)

            # Print detected boats and their coordinates
            print(f"Detected boats: {len(detected_boats)}")
            for i, coords in enumerate(detected_boats):
                print(f"Boat {i+1}: {coords}")

            cv2.imshow('Processed Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        image = cv2.imread(media_path)
        if image is None:
            print(f"Error: Could not load image {media_path}")
            return

        detectron2_result = detection_model(image)
        panoptic_result = panoptic_model(image)

        detected_boats = draw_detectron2_detections(image, detectron2_result, detection_labels)
        apply_panoptic_segmentation(image, panoptic_result)

        # Print detected boats and their coordinates
        print(f"Detected boats: {len(detected_boats)}")
        for i, coords in enumerate(detected_boats):
            print(f"Boat {i+1}: {coords}")

        cv2.imshow('Processed Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    media_path = "/Users/saadbenboujina/Downloads/1/2/image_1234562_0.jpg"

    detection_model = setup_detection_model()  # Set up the Detectron2 detection model

    panoptic_model = setup_panoptic_model()  # Set up the panoptic segmentation model

    detection_labels = ["boat"]  # Labels for detection

    process_media(media_path, detection_model, panoptic_model, detection_labels)

if __name__ == '__main__':
    main()
