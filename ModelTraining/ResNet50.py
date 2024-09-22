import torch
import torch.nn as nn
from future.moves import multiprocessing
from torchvision import datasets, models, transforms
import os
import copy
import time
import logging
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm  # Für Fortschrittsbalken

def main():
    # Konfigurieren Sie das Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Datenvorbereitung
    data_dir = '/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/ForCategory'  # Ersetzen Sie dies durch Ihren tatsächlichen Pfad
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

    # Überprüfen Sie, ob die Unterverzeichnisse 'train' und 'val' existieren
    for phase in ['train', 'val']:
        path = os.path.join(data_dir, phase)
        if not os.path.isdir(path):
            logger.error(f"Verzeichnis für Phase '{phase}' nicht gefunden: {path}")
            raise FileNotFoundError(f"Verzeichnis für Phase '{phase}' nicht gefunden: {path}")

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    num_classes = len(image_datasets['train'].classes)
    class_names = image_datasets['train'].classes

    num_workers = multiprocessing.cpu_count() if os.name != 'nt' else 0
    dataloaders = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True)
        for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    logger.info(f"Klassen: {class_names}")
    logger.info(f"Trainingsgröße: {dataset_sizes['train']}")
    logger.info(f"Validierungsgröße: {dataset_sizes['val']}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Verwendetes Gerät: {device}")

    # Berechnung der Klassen-Gewichte zur Handhabung von Klassenungleichgewichten
    train_labels = image_datasets['train'].targets
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    logger.info(f"Klassen-Gewichte: {class_weights}")

    # Modell laden und anpassen
    model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Aktualisiert auf 'weights'
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)  # Anzahl der Schiffskategorien

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Nur die letzten Schichten trainieren
    optimizer_ft = torch.optim.Adam(model_ft.fc.parameters(), lr=0.001)

    # Scheduler für Lernrate
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Training und Validierung
    def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
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

                # Iterieren über Daten mit Fortschrittsbalken
                for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()}-Phase"):
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

                    # **Assert-Anweisungen **
                    assert isinstance(preds, torch.Tensor), "preds ist kein Tensor"
                    assert isinstance(labels, torch.Tensor), "labels ist kein Tensor"
                    # Statistiken sammeln
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += (preds == labels).sum().item()

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                logger.info(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Beste Genauigkeit speichern
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            logger.info("")  # Leere Zeile zur Trennung der Epochen

        time_elapsed = time.time() - since
        logger.info(f'Training abgeschlossen in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        logger.info(f'Beste Validierungsgenauigkeit: {best_acc:.4f}')

        # Beste Modellgewichte laden
        model.load_state_dict(best_model_wts)
        return model

    # Starten des Trainings
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)

    # Modell speichern
    model_save_path = 'ship_classification_resnet50.pth'
    torch.save(model_ft.state_dict(), model_save_path)
    logger.info(f"Modell gespeichert als {model_save_path}")

if __name__ == '__main__':
    main()
