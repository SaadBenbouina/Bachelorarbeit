import os
import torch
import detectron2.utils.logger as d2_logger
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo

d2_logger.setup_logger()

# Register your dataset in COCO format
def register_datasets():
    print("Registering datasets...")
    register_coco_instances("my_dataset_train", {}, '/private/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/export_coco_panoptic_corrected.json', "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/TrainDataSegmentation/notdone")
    register_coco_instances("my_dataset_val", {}, '/private/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/export_coco_panoptic_corrected.json', "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/TrainDataSegmentation/notdone")
    print("Datasets registered.")

# Set up the configuration
def setup_cfg():
    print("Setting up configuration...")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))

    # Load pre-trained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

    # Set the number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    # Set the dataset paths
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)

    # Set hyperparameters
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    # Set output directory
    cfg.OUTPUT_DIR = "/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/DetectronModel"

    # Check if CUDA is available
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        print("CUDA is available. Using GPU.")
    else:
        cfg.MODEL.DEVICE = "cpu"
        print("CUDA is not available. Using CPU.")

    return cfg

# Start training
def train_model():
    print("Starting model training...")
    register_datasets()
    cfg = setup_cfg()

    print(f"Creating output directory: {cfg.OUTPUT_DIR}")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("Initializing trainer...")
    trainer = DefaultTrainer(cfg)

    print("Resuming or starting new training...")
    trainer.resume_or_load(resume=False)
    trainer.train()

    print("Training completed.")

if __name__ == "__main__":
    train_model()
