import torch
import torch.nn as nn
import multiprocessing
from torchvision import datasets, models, transforms
import os
import copy
import time
import logging
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm  # Für Fortschrittsbalken
from sklearn.metrics import classification_report, confusion_matrix

# Setze den Zufallssamen für Reproduzierbarkeit
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Early Stopping Klasse
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): Wie viele Epochen ohne Verbesserung gewartet werden soll.
            verbose (bool): Wenn True, gibt es Meldungen bei Verbesserungen.
            delta (float): Mindestverbesserung, um als Verbesserung zu gelten.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
            if self.verbose:
                print(f'Initiales Bestes Verlust: {self.best_loss:.4f}')
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f'Verbesserter Verlust: {self.best_loss:.4f}')

def main():
    # Setze den Zufallssamen
    set_seed(42)

    # Konfigurieren Sie das Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Datenverzeichnis
    data_dir = '/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/ForCategory'

    # Datenvorbereitung mit Datenaugmentation für das Training
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

    # Laden der Datensätze
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    num_classes = len(image_datasets['train'].classes)
    class_names = image_datasets['train'].classes

    # Bestimmen der Anzahl der Arbeiter für DataLoader
    num_workers = multiprocessing.cpu_count() if os.name != 'nt' else 0
    dataloaders = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=32, shuffle=True if x == 'train' else False, num_workers=num_workers, pin_memory=True)
        for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    # Logging der Dateninformationen
    logger.info(f"Klassen: {class_names}")
    logger.info(f"Trainingsgröße: {dataset_sizes['train']}")
    logger.info(f"Validierungsgröße: {dataset_sizes['val']}")
    logger.info(f"Testgröße: {dataset_sizes['test']}")

    # Bestimmen des Geräts
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

    # Feinabstimmung: Nur bestimmte Schichten trainieren
    for param in model_ft.parameters():
        param.requires_grad = False  # Alle Schichten einfrieren

    # Letzte Blockschichten freigeben (zum Beispiel layer4)
    for param in model_ft.layer4.parameters():
        param.requires_grad = True

    # Letzte vollständig verbundene Schicht trainieren
    for param in model_ft.fc.parameters():
        param.requires_grad = True

    model_ft = model_ft.to(device)

    # Verlustfunktion mit Klassen-Gewichten
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimierer: Nur die Parameter mit requires_grad=True optimieren
    optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.001)

    # Scheduler für Lernrate
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Training und Validierung
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        # Early Stopping initialisieren
        early_stopping = EarlyStopping(patience=10, verbose=True)

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

                # Validierungsphase: Early Stopping überwachen
                if phase == 'val':
                    early_stopping(epoch_loss)

                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

                    if early_stopping.early_stop:
                        logger.info("Early stopping aktiviert")
                        model.load_state_dict(best_model_wts)
                        return model

            logger.info("")  # Leere Zeile zur Trennung der Epochen

        time_elapsed = time.time() - since
        logger.info(f'Training abgeschlossen in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        logger.info(f'Beste Validierungsgenauigkeit: {best_acc:.4f}')

        # Beste Modellgewichte laden
        model.load_state_dict(best_model_wts)
        return model

    # Starten des Trainings
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

    # Modell speichern (nur die Gewichte)
    model_save_path = 'ship_classification_resnet50.pth'
    torch.save(model_ft.state_dict(), model_save_path)
    logger.info(f"Modell gespeichert als {model_save_path}")

if __name__ == '__main__':
    main()
