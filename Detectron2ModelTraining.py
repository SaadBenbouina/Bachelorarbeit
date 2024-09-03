from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_panoptic_separated
from fiftyone.utils.transformers import torch


def setup_panoptic_model():
    cfg = get_cfg()

    # Load a base config file for Panoptic Segmentation
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))

    # Load pre-trained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

    # Use GPU if available
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset registration (replace with your dataset paths)
    register_coco_panoptic_separated(
        "my_panoptic_train",
        {},
        image_root="path/to/your/train/images",
        panoptic_root="path/to/your/panoptic/annotations",
        panoptic_json="path/to/your/panoptic/train_annotations.json",
        instances_json="path/to/your/instances/train_annotations.json"
    )

    register_coco_panoptic_separated(
        "my_panoptic_val",
        {},
        image_root="path/to/your/val/images",
        panoptic_root="path/to/your/panoptic/annotations",
        panoptic_json="path/to/your/panoptic/val_annotations.json",
        instances_json="path/to/your/instances/val_annotations.json"
    )

    # Dataset parameters
    cfg.DATASETS.TRAIN = ("my_panoptic_train",)
    cfg.DATASETS.TEST = ("my_panoptic_val",)

    # Number of classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 80  # Adjust according to your dataset

    # Batch size and learning rate
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # Adjust according to your needs
    cfg.SOLVER.MAX_ITER = 300000  # Adjust number of iterations

    # Set the output directory
    cfg.OUTPUT_DIR = "./output_panoptic"

    # Create the trainer and start training
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # Start training
    trainer.train()

# Call the function to train the model
setup_panoptic_model()
