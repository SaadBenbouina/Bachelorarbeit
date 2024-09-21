import cv2
import os

# Function to convert bounding boxes in YOLO format to xmin, ymin, xmax, ymax.
def yolo2bbox(bboxes):
    xmin = bboxes[0] - bboxes[2] / 2
    ymin = bboxes[1] - bboxes[3] / 2
    xmax = bboxes[0] + bboxes[2] / 2
    ymax = bboxes[1] + bboxes[3] / 2
    return xmin, ymin, xmax, ymax

# Function to plot the bounding boxes on the image.
def plot_box(image, bboxes, labels):
    h, w, _ = image.shape  # Image dimensions

    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)

        # Denormalizing the coordinates.
        xmin = int(x1 * w)
        ymin = int(y1 * h)
        xmax = int(x2 * w)
        ymax = int(y2 * h)

        thickness = max(2, int(w / 275))

        # Draw the bounding box on the image.
        cv2.rectangle(
            image,
            (xmin, ymin), (xmax, ymax),
            color=(0, 0, 255), thickness=thickness
        )

        # Optionally, add label text above the bounding box
        if labels:
            label = labels[box_num]
            cv2.putText(
                image,
                label,
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

    return image

# Function to process all images in a folder and save the plotted images
def plot_folder_images(image_folder, label_folder, output_folder):
    print("Starting processing...")

    # Create the output folder if it doesn't exist
    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Output folder is set to: {output_folder}")
    except Exception as e:
        print(f"Failed to create output folder: {e}")
        return

    # Supported image extensions
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    # List all image files in the image_folder
    try:
        image_files = [
            f for f in os.listdir(image_folder)
            if os.path.splitext(f)[1].lower() in supported_extensions
        ]
        print(f"Found {len(image_files)} image(s) in the folder.")
    except Exception as e:
        print(f"Failed to list files in image folder: {e}")
        return

    if not image_files:
        print("No images found in the specified image folder.")
        return

    processed_count = 0
    skipped_no_label = 0
    skipped_bad_label = 0

    for idx, image_file in enumerate(image_files, start=1):
        print(f"\nProcessing {idx}/{len(image_files)}: {image_file}")
        image_path = os.path.join(image_folder, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(label_folder, label_file)

        # Check if label file exists
        if not os.path.exists(label_path):
            print(f"  [SKIP] Label file not found: {label_file}")
            skipped_no_label += 1
            continue

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  [ERROR] Failed to read image: {image_file}")
            skipped_bad_label += 1
            continue

        # Read the label file
        bboxes = []
        labels = []
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    print(f"  [WARNING] Label file is empty: {label_file}")
                for label_line_num, label_line in enumerate(lines, start=1):
                    if not label_line.strip():
                        print(f"    [INFO] Skipping empty line {label_line_num} in {label_file}.")
                        continue  # Skip empty lines
                    parts = label_line.strip().split()
                    if len(parts) < 5:
                        print(f"    [WARNING] Invalid label format on line {label_line_num} in {label_file}, skipping this box.")
                        skipped_bad_label += 1
                        continue
                    label = parts[0]
                    try:
                        x_c, y_c, w, h = map(float, parts[1:5])
                    except ValueError:
                        print(f"    [WARNING] Non-numeric values on line {label_line_num} in {label_file}, skipping this box.")
                        skipped_bad_label += 1
                        continue
                    bboxes.append([x_c, y_c, w, h])
                    labels.append(label)
        except Exception as e:
            print(f"  [ERROR] Failed to read label file {label_file}: {e}")
            skipped_bad_label += 1
            continue

        if not bboxes:
            print(f"  [INFO] No valid bounding boxes found in {label_file}, skipping image.")
            skipped_bad_label += 1
            continue

        # Plotting the bounding boxes on the image
        try:
            result_image = plot_box(image.copy(), bboxes, labels)
        except Exception as e:
            print(f"  [ERROR] Failed to plot boxes on image {image_file}: {e}")
            skipped_bad_label += 1
            continue

        # Define the output path
        output_path = os.path.join(output_folder, image_file)

        # Save the result image to the output folder
        try:
            success = cv2.imwrite(output_path, result_image)
            if success:
                print(f"  [SUCCESS] Processed and saved: {output_path}")
                processed_count += 1
            else:
                print(f"  [ERROR] Failed to save image: {output_path}")
                skipped_bad_label += 1
        except Exception as e:
            print(f"  [ERROR] Exception occurred while saving image {output_path}: {e}")
            skipped_bad_label += 1

    # Summary
    print("\nProcessing Completed.")
    print(f"  Total Images Found: {len(image_files)}")
    print(f"  Successfully Processed and Saved: {processed_count}")
    print(f"  Skipped (No Label Files): {skipped_no_label}")
    print(f"  Skipped Due to Errors or Bad Labels: {skipped_bad_label}")

# Example usage:
if __name__ == "__main__":
    # Define your directories
    image_folder = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/TrainDataYolo/train"
    label_folder = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/TrainDataYolo/train"
    output_folder = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/WithPlotBox"

    # Process the folder
    plot_folder_images(image_folder, label_folder, output_folder)
