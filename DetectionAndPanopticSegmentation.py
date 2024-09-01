import os
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch  # Corrected the import statement

def draw_detectron_detections(frame, detectron_result, detection_labels):
    detected = False
    instances = detectron_result["instances"]

    for i in range(len(instances)):
        class_id = int(instances.pred_classes[i])
        score = instances.scores[i].item()
        if score >= 0.5:  # Added a threshold for detection score
            if class_id < len(detection_labels):
                label = detection_labels[class_id]
                detected = True
                box = instances.pred_boxes[i].tensor.cpu().numpy().astype(int)[0]
                label_text = f'{label} {score:.2f}'
                # Draw the rectangle and label
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
                cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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

def setup_detectron_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for detection
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    detectron_predictor = DefaultPredictor(cfg)
    return detectron_predictor

def visualize_detections(frame, detectron_result):
    instances = detectron_result["instances"]
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()

    for i in range(len(boxes)):
        box = boxes[i].astype(int)
        label_text = f'Class: {classes[i]}, Score: {scores[i]:.2f}'
        frame = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        frame = cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detections", frame)

def process_media(media_path, detectron_model, panoptic_model, output_dir='output', detection_labels=None):
    if detection_labels is None:
        detection_labels = []

    is_video = media_path.endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    detected = False  # To track if the object is detected in the media

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

        output_video_path = os.path.join(output_dir, os.path.basename(media_path))
        fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detectron_result = detectron_model(frame)
            panoptic_result = panoptic_model(frame)

            detected |= draw_detectron_detections(frame, detectron_result, detection_labels)
            apply_panoptic_segmentation(frame, panoptic_result)

            visualize_detections(frame, detectron_result)  # Visualize detections for debugging

            print(f"Processing frame {frame_count}/{total_frames}")
            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processed and saved video: {output_video_path}")

    else:
        image = cv2.imread(media_path)
        if image is None:
            print(f"Error: Could not load image {media_path}")
            return detected
        print(f"Image loaded: {image.shape}")

        detectron_result = detectron_model(image)
        panoptic_result = panoptic_model(image)

        detected = draw_detectron_detections(image, detectron_result, detection_labels)
        apply_panoptic_segmentation(image, panoptic_result)

        visualize_detections(image, detectron_result)  # Visualize detections for debugging

        # Keep the window open until 'q' is pressed
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        if detected:
            output_image_path = os.path.join(output_dir, os.path.basename(media_path))
            cv2.imwrite(output_image_path, image)
            print(f"Processed and saved image: {output_image_path}")

    return detected

def main():
    media_path = "/Users/saadbenboujina/Downloads/1/2/image_1234563_0.jpg"
    output_folder = "/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/JustInputWithBoat"
    detectron_model = setup_detectron_model()  # Load the Detectron2 model for detection
    panoptic_model = setup_panoptic_model()  # Set up the panoptic segmentation model

    detection_labels = ["boat"]  # Labels for detectionq

    process_media(media_path, detectron_model, panoptic_model, output_folder, detection_labels)

if __name__ == '__main__':
    main()
