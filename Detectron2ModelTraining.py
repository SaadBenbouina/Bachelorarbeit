import os

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo

# Register your dataset in COCO format
def register_datasets():
    register_coco_instances("my_dataset_train", {}, "path/to/annotations/instances_train.json", "path/to/train/images")
    register_coco_instances("my_dataset_val", {}, "path/to/annotations/instances_val.json", "path/to/val/images")

# Set up the configuration
def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))

    # Load pre-trained weights if available or use COCO weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

    # Set the number of classes in your dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # If you have only one class, e.g., "boat"

    # Set the dataset paths
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)

    # Set the batch size, learning rate, and other hyperparameters
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 10000  # Number of iterations

    # Save checkpoints during training
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    # Set output directory
    cfg.OUTPUT_DIR = "/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/segmentation_Model"

    return cfg

# Start training
def train_model():
    register_datasets()
    cfg = setup_cfg()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Trainer
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    train_model()
