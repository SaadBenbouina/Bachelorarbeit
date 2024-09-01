import os
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.checkpoint import Checkpointer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine.hooks import PeriodicCheckpointer

class CustomTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.checkpointer = PeriodicCheckpointer(self.checkpointer, period=1, max_iter=cfg.SOLVER.MAX_ITER)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, self.checkpointer)
        return hooks

# Register your dataset if it's not in COCO format
# Assuming "my_dataset_train" and "my_dataset_test" are registered correctly

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MAX_ITER = 500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class: boat

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=True)  # Resume from the last checkpoint if it exists
trainer.train()

# Save the final model after all iterations
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
trainer.checkpointer.save("model_final")
