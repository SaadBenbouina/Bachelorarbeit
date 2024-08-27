import cv2
import os
from ultralytics import YOLO

def process_media(media_path, model, output_dir='output', labels=None):
    if labels is None:
        labels = []

    is_video = media_path.endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if is_video:
        # Process video
        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {media_path}")
            return

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video loaded: {frame_width}x{frame_height} at {fps} FPS with {total_frames} frames")

        # Define the codec and create a VideoWriter object to save the output video
        output_video_path = os.path.join(output_dir, os.path.basename(media_path))
        fourcc = cv2.VideoWriter.fourcc('M', 'P', '4', 'V')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Inference with YOLO model
            result = model.predict(frame)
            detections = result[0]

            detected_labels = []
            for box in detections.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                if label in labels:
                    detected_labels.append(label)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    label_text = f'{label} {confidence:.2f}'

                    # Draw the bounding box on the frame only for specified labels
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print(f"Processing frame {frame_count}/{total_frames}: Detected objects: {detected_labels}")

            # Show the frame with annotations
            cv2.imshow("Annotated Video", frame)

            # Write the frame into the output video
            out.write(frame)
            frame_count += 1

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processed and saved video: {output_video_path}")

    else:
        image = cv2.imread(media_path)
        if image is None:
            print(f"Error: Could not load image {media_path}")
            return
        print(f"Image loaded: {image.shape}")

        result = model.predict(image)
        detections = result[0]

        detected_labels = []
        for box in detections.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            if label in labels:
                detected_labels.append(label)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                label_text = f'{label} {confidence:.2f}'

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"Detected objects: {detected_labels}")

        if detected_labels:
            cv2.imshow("Annotated Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            output_image_path = os.path.join(output_dir, os.path.basename(media_path))
            cv2.imwrite(output_image_path, image)
            print(f"Processed and saved image: {output_image_path}")
        else:
            print("No specified objects found in the image.")

def main():
    media_path = "/Users/saadbenboujina/Downloads/1/IMG_8861.mp4"
    model = YOLO("yolov8n.pt")
    labels = [ "boat"]

    process_media(media_path, model, labels=labels)

if __name__ == '__main__':
    main()
