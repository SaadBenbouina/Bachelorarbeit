import cv2
import os
import numpy as np
from ultralytics import YOLO

def process_media(media_path, model, output_dir='output', detection_labels=None, segmentation_labels=None):
    if detection_labels is None:
        detection_labels = []
    if segmentation_labels is None:
        segmentation_labels = []

    is_video = media_path.endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    detected = False  # To track if the object is detected in the media

    if is_video:
        # Process video
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
        fourcc = cv2.VideoWriter.fourcc('M', 'P', '4', 'V')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = model.predict(frame)
            detections = result[0]

            for box in detections.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                if label in detection_labels:
                    detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    label_text = f'{label} {confidence:.2f}'

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Apply segmentation if the label is also in segmentation_labels
                    if label in segmentation_labels and detections.masks:
                        mask = detections.masks.data[int(box.cls[0])]
                        mask = mask.cpu().numpy()

                        # Resize mask to match the frame dimensions
                        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                        violet_color = (238, 130, 238)
                        colored_mask = cv2.merge([mask_resized * violet_color[0],
                                                  mask_resized * violet_color[1],
                                                  mask_resized * violet_color[2]])
                        colored_mask = colored_mask.astype('uint8')

                        # Blend the frame with the colored mask
                        frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

            cv2.imshow('Processed Video', frame)  # Display the video frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(f"Processing frame {frame_count}/{total_frames}")
            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processed and saved video: {output_video_path}")

    else:
        # Process image
        image = cv2.imread(media_path)
        if image is None:
            print(f"Error: Could not load image {media_path}")
            return detected
        print(f"Image loaded: {image.shape}")

        result = model.predict(image)
        detections = result[0]

        for i, box in enumerate(detections.boxes):
            class_id = int(box.cls[0])
            label = model.names[class_id]
            if label in detection_labels:
                detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                label_text = f'{label} {confidence:.2f}'

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Apply segmentation if the label is also in segmentation_labels
                if label in segmentation_labels and detections.masks:
                    try:
                        mask = detections.masks.data[i]
                        mask = mask.cpu().numpy()

                        # Resize mask to match the image dimensions
                        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

                        # Ensure mask is binary
                        mask_resized = np.where(mask_resized > 0.5, 1, 0)

                        violet_color = (238, 130, 238)
                        colored_mask = np.stack([mask_resized * violet_color[0],
                                                 mask_resized * violet_color[1],
                                                 mask_resized * violet_color[2]], axis=-1)
                        colored_mask = colored_mask.astype('uint8')

                        # Blend the image with the colored mask
                        image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)
                    except IndexError:
                        print(f"Warning: No corresponding mask found for detection {i} with label '{label}'.")

        cv2.imshow('Processed Image', image)  # Display the processed image
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()

        if detected:
            output_image_path = os.path.join(output_dir, os.path.basename(media_path))
            cv2.imwrite(output_image_path, image)
            print(f"Processed and saved image: {output_image_path}")

    return detected

def main():
    # Provide a single image or video path
    media_path = "/Users/saadbenboujina/Downloads/1/IMG_8861.mp4"
    output_folder = "/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/JustInputWithBoat"
    model = YOLO("yolov8s-seg.pt")  # Load the segmentation variant of YOLO

    detection_labels = ["boat"]  # Labels for detection
    segmentation_labels = ["boat","person"]  # Labels for segmentation

    # Process the single image or video
    process_media(media_path, model, output_folder, detection_labels, segmentation_labels)

if __name__ == '__main__':
    main()
