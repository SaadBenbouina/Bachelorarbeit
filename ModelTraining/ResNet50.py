import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import copy
import time
import logging
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Konfigurieren Sie das Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Datenvorbereitung
data_dir = '/path/to/organized_dataset'  # Pfad zu Ihrem organisierten Datensatz
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [
            0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [
            0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(
    image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
    for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

logger.info(f"Klassen: {class_names}")
logger.info(f"Trainingsgröße: {dataset_sizes['train']}")
logger.info(f"Validierungsgröße: {dataset_sizes['val']}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Verwendes Gerät: {device}")

# Berechnung der Klassen-Gewichte zur Handhabung von Klassenungleichgewichten
train_labels = []
for _, label in image_datasets['train']:
    train_labels.append(label)
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
logger.info(f"Klassen-Gewichte: {class_weights}")

# Modell laden und anpassen
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 16)  # Anzahl der Schiffskategorien

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# Nur die letzten Schichten trainieren
optimizer_ft = torch.optim.Adam(model_ft.fc.parameters(), lr=0.001)

# Scheduler für Lernrate
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Training und Validierung
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        logger.info('-' * 10)

        # Jede Epoche hat ein Training und eine Validierung
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Trainingsmodus
            else:
                model.eval()   # Evaluationsmodus

            running_loss = 0.0
            running_corrects = 0

            # Iterieren über Daten
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Vorwärts-Pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Rückwärts-Pass und Optimierung nur im Training
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Statistiken sammeln
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Beste Genauigkeit speichern
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        logger.info()

    time_elapsed = time.time() - since
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best val Acc: {best_acc:.4f}')

    # Beste Modellgewichte laden
    model.load_state_dict(best_model_wts)
    return model

# Starten des Trainings
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

# Modell speichern
model_save_path = 'ship_classification_resnet50.pth'
torch.save(model_ft.state_dict(), model_save_path)
logger.info(f"Modell gespeichert als {model_save_path}")
