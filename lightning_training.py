import subprocess

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data_loading.load_augsburg15 import Augsburg15DetectionDataset
from models.object_detector import ObjectDetector, ClassificationLoss
from models.timm_adapter import Network


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


if __name__ == '__main__':
    model = ObjectDetector(
        num_classes=Augsburg15DetectionDataset.NUM_CLASSES,
        batch_size=2,
        timm_model=Network.RESNET_50,
        min_image_size=800,
        max_image_size=1066,
        freeze_backbone=False,
        classification_loss_function=ClassificationLoss.CROSS_ENTROPY
    )
    log_directory = 'logs'
    experiment_name = f'faster_rcnn#{get_git_revision_short_hash()}'
    logger = TensorBoardLogger(log_directory, experiment_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{log_directory}/{experiment_name}',
        save_last=True,
        save_top_k=1,
        monitor='validation_map_50',
        mode='max',
    )
    trainer = Trainer(max_epochs=40, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=model.train_dataloader(), val_dataloaders=model.val_dataloader())
