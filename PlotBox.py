import cv2
import matplotlib.pyplot as plt


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

    return image

# Function to process one image and its bounding box labels.
def plot_single_image(image_path, label_path):
    image_name = image_path.split('/')[-1].split('.')[0]
    image = cv2.imread(image_path)

    with open(label_path, 'r') as f:
        bboxes = []
        labels = []
        lines = f.readlines()
        for label_line in lines:
            label = label_line[0]
            bbox_string = label_line[2:]
            x_c, y_c, w, h = bbox_string.split(' ')
            x_c = float(x_c)
            y_c = float(y_c)
            w = float(w)
            h = float(h)
            bboxes.append([x_c, y_c, w, h])
            labels.append(label)

    # Plotting the bounding boxes on the image
    result_image = plot_box(image, bboxes, labels)

    # Display the result
    plt.imshow(result_image[:, :, ::-1])
    plt.axis('off')
    plt.show()

# Visualization for one specific image and its corresponding label file
plot_single_image(
    image_path="/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/TrainDataYolo/train/image_1335717_0_jpg.rf.d44d2d834be0eca702ff7655d000d661.jpg",
    label_path="/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/TrainDataYolo/train/image_1335717_0_jpg.rf.d44d2d834be0eca702ff7655d000d661.txt"
)
