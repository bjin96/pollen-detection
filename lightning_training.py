import subprocess

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from data_loading.load_augsburg15 import Augsburg15DetectionDataset
from models.object_detector import ObjectDetector
from models.timm_adapter import Network


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


if __name__ == '__main__':
    model = ObjectDetector(
        num_classes=Augsburg15DetectionDataset.NUM_CLASSES,
        batch_size=4,
        timm_model=Network.RESNET_50,
        min_image_size=800,
        max_image_size=1066,
        freeze_backbone=False
    )
    logger = TensorBoardLogger('logs', f'faster_rcnn#{get_git_revision_short_hash()}')
    trainer = Trainer(max_epochs=40, logger=logger)
    trainer.fit(model, train_dataloaders=model.train_dataloader(), val_dataloaders=model.val_dataloader())
