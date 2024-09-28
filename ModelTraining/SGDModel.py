import torch
import torchvision
from matplotlib import patches, pyplot as plt
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import os
from PIL import Image

# Definiere den Dataset-Klasse
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, target_size=(800, 800)):
        self.root = root
        self.transforms = transforms
        self.target_size = target_size  # Neue Bildgröße nach Transformation
        all_images = [f for f in sorted(os.listdir(root)) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        all_annotations = [f for f in sorted(os.listdir(root)) if f.lower().endswith('.txt')]

        image_basenames = set(os.path.splitext(f)[0] for f in all_images)
        annotation_basenames = set(os.path.splitext(f)[0] for f in all_annotations)
        common_basenames = image_basenames.intersection(annotation_basenames)

        self.basenames = sorted(common_basenames)

    def visualize_sample(self, idx, class_names=None):
        img, target = self[idx]
        img = img.permute(1, 2, 0).numpy()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        for box, label in zip(target['boxes'], target['labels']):
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            if class_names and label < len(class_names):
                ax.text(xmin, ymin, class_names[label], fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))
        plt.show()

    def __getitem__(self, idx):
        basename = self.basenames[idx]
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            potential_path = os.path.join(self.root, basename + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        img = Image.open(img_path).convert("RGB")
        original_width, original_height = img.size

        annot_path = os.path.join(self.root, basename + '.txt')
        boxes = []
        labels = []
        with open(annot_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_label, x_center, y_center, box_width, box_height = parts
                class_label = int(class_label[1:]) if class_label.startswith('i') else int(float(class_label))
                class_label += 1  # Hintergrund ist 0

                x_center = float(x_center) * original_width
                y_center = float(y_center) * original_height
                box_width = float(box_width) * original_width
                box_height = float(box_height) * original_height

                xmin = x_center - (box_width / 2)
                ymin = y_center - (box_height / 2)
                xmax = x_center + (box_width / 2)
                ymax = y_center + (box_height / 2)

                # Clip Bounding Boxes auf Bildgrenzen
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(original_width, xmax)
                ymax = min(original_height, ymax)

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.numel() > 0 else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            # Berechne Skalierungsfaktoren
            new_width, new_height = self.target_size
            scale_x = new_width / original_width
            scale_y = new_height / original_height

            # Wende die Transformationen an (Bild wird skaliert)
            img = self.transforms(img)

            # Skaliere die Bounding Boxes entsprechend
            target["boxes"] = target["boxes"].clone()
            target["boxes"][:, [0, 2]] = target["boxes"][:, [0, 2]] * scale_x
            target["boxes"][:, [1, 3]] = target["boxes"][:, [1, 3]] * scale_y

        return img, target

    def __len__(self):
        return len(self.basenames)

# Funktion zum Anpassen des Modells
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

# Visualisierungsfunktion außerhalb der Klasse definieren
def visualize_sample(dataset, idx, class_names=None):
    img, target = dataset[idx]
    img = img.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for box, label in zip(target['boxes'], target['labels']):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if class_names and label < len(class_names):
            ax.text(xmin, ymin, class_names[label], fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))
    plt.show()

def main():
    dataset_root = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/TrainDataYolo/train2"

    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(dataset_root, transforms=transform)

    # Optional: Definiere Klassen-Namen
    class_names = ['Hintergrund', 'Klasse1']  # Passe dies an deine tatsächlichen Klassen an

    # Visualisierung aufrufen
    visualize_sample(dataset, 4, class_names)
    # Optional: Weitere Beispiele visualisieren
    # visualize_sample(dataset, 1, class_names)
    # visualize_sample(dataset, 2, class_names)

    total_samples = len(dataset)

    num_test = min(50, total_samples // 5)
    num_train = total_samples - num_test

    train_dataset, _ = torch.utils.data.random_split(dataset, [num_train, num_test])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)

    num_classes = 2  # Hintergrund + eine Klasse
    model = get_model(num_classes)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for batch_idx, (images, targets) in enumerate(train_loader, 1):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        lr_scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        print(f"Finished Epoch {epoch + 1}/{num_epochs}")
        print(f"Epoch: {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

    torch.save(model.state_dict(), "fasterrcnn_model.pth")
    print("Modell gespeichert als 'fasterrcnn_model.pth'")

if __name__ == "__main__":
    main()
