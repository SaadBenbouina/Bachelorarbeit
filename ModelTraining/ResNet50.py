import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os
import copy
import logging
import optuna
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setze den Zufallssamen für Reproduzierbarkeit
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Early Stopping Klasse
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
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
                logger.info(f'Initiales Bestes Verlust: {self.best_loss:.4f}')
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                logger.info(f'Verbesserter Verlust: {self.best_loss:.4f}')

def objective(trial):
    set_seed(42)

    # Hyperparameter optimieren
    batch_size = trial.suggest_int("batch_size", 16, 64, step=16)
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    patience = trial.suggest_int("patience", 5, 15)

    # Start Logging für diesen Trial
    logger.info(f"Starte Trial {trial.number} mit batch_size={batch_size}, lr={learning_rate}, patience={patience}")

    # Datenverzeichnis
    data_dir = '/content/drive/MyDrive/crypto/ForCategory'

    # Datenvorbereitung
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Daten laden
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'))
                   for x in ['train', 'val']}

    num_classes = len(image_datasets['train'].classes)
    logger.info(f"Anzahl Klassen: {num_classes}, Klassen: {image_datasets['train'].classes}")
    logger.info(f"Trainingsgröße: {dataset_sizes['train']}, Validierungsgröße: {dataset_sizes['val']}")

    # Gerät festlegen
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Verwendetes Gerät: {device}")

    # Klassen-Gewichte
    train_labels = image_datasets['train'].targets
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    logger.info(f"Klassen-Gewichte: {class_weights}")

    # Modell laden und konfigurieren
    model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    # Schichten einfrieren/auftauen
    for param in model_ft.parameters():
        param.requires_grad = False
    for param in model_ft.layer4.parameters():
        param.requires_grad = True
    for param in model_ft.fc.parameters():
        param.requires_grad = True

    model_ft = model_ft.to(device)

    # Verlustfunktion, Optimierer, Scheduler
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=3)

    # Early Stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Training
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model_ft.state_dict())
    num_epochs = 25
    logger.info(f"Starte Training für {num_epochs} Epochen")

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        logger.info('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model_ft.train()
                logger.info("Train-Phase gestartet")
            else:
                model_ft.eval()
                logger.info("Val-Phase gestartet")

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()}-Phase"):
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)

                    if phase == 'train':
                        optimizer_ft.zero_grad()
                        loss.backward()
                        optimizer_ft.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            logger.info(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)
                # Bestes Modell aktualisieren
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model_ft.state_dict())
                    # Verzeichnis für das beste Modell anlegen, falls nicht vorhanden
                    best_model_dir = '/content/drive/MyDrive/optuna_results_category'
                    os.makedirs(best_model_dir, exist_ok=True)

                    model_save_path = os.path.join(best_model_dir, f'best_model_trial_{trial.number}.pth')
                    torch.save(best_model_wts, model_save_path)
                    logger.info(f"Bestes Modell gespeichert: {model_save_path}")

                # Early Stopping prüfen
                early_stopping(epoch_loss)
                if early_stopping.early_stop:
                    logger.info("Early stopping aktiviert")
                    model_ft.load_state_dict(best_model_wts)
                    logger.info(f"Beende Training in Epoche {epoch+1} mit bester Accuracy: {best_acc:.4f}")
                    return best_acc

    # Endgültiges Modell speichern
    final_model_dir = '/content/drive/MyDrive/optuna_results'
    os.makedirs(final_model_dir, exist_ok=True)
    final_model_path = os.path.join(final_model_dir, f'final_model_trial_{trial.number}.pth')
    torch.save(best_model_wts, final_model_path)
    logger.info(f"Finales Modell gespeichert: {final_model_path}")

    model_ft.load_state_dict(best_model_wts)
    logger.info(f"Training abgeschlossen. Beste Accuracy: {best_acc:.4f}")
    return best_acc

# Optuna-Studie starten
logger.info("Starte Optuna-Studie")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

best_trial_number = study.best_trial.number
logger.info(f"Beste Parameter: {study.best_params}")
logger.info(f"Beste Validierungsgenauigkeit: {study.best_value}")
logger.info(f"Beste Trial-Nummer: {best_trial_number}")
