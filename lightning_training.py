import subprocess

import click
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data_loading.load_augsburg15 import Augsburg15DetectionDataset
from models.object_detector import ObjectDetector, ClassificationLoss, Augmentation
from models.timm_adapter import Network


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


NETWORKS = {
    'resnet50': Network.RESNET_50,
    'efficient_net_v2': Network.EFFICIENT_NET_V2,
    'mobile_net_v3': Network.MOBILE_NET_V3,
}
CLASSIFICATION_LOSS_FUNCTIONS = {
    'cross_entropy': ClassificationLoss.CROSS_ENTROPY,
    'focal_loss': ClassificationLoss.FOCAL,
}


@click.command()
@click.option(
    '--experiment_name',
    help='Experiment name used for identifying the logs and checkpoints.',
    required=True,
)
@click.option(
    '--checkpoint_path',
    default=None,
    help='Path to the checkpoint from which to resume training.',
)
@click.option(
    '--batch_size',
    default=2,
    help='Batch size for the training.'
)
@click.option(
    '--backbone',
    default='resnet50',
    help='Which pre-trained model to use as a backbone.'
)
@click.option(
    '--min_image_size',
    default=800,
    help='Minimum size of the resized image that is fed into the model.'
)
@click.option(
    '--max_image_size',
    default=1066,
    help='Maximum size of the resized image that is fed into the model.'
)
@click.option(
    '--freeze_backbone',
    default=False,
    help='Whether to freeze the backbone for the training.'
)
@click.option(
    '--classification_loss_function',
    default='cross_entropy',
    help='Which loss function to use for the classification.'
)
@click.option(
    '--use_weights',
    default=False,
    help='Whether to use weights to counteract class imbalance.'
)
def start_experiment(
        experiment_name: str,
        checkpoint_path: str,
        batch_size: int,
        backbone: str,
        min_image_size: int,
        max_image_size: int,
        freeze_backbone: bool,
        classification_loss_function: str,
        use_weights: bool,
):
    print(
        f'Starting experiment {experiment_name} with: batch_size = {batch_size}, backbone = {backbone}, '
        f'min_image_size = {min_image_size}, max_image_size = {max_image_size}, freeze_backbone = {freeze_backbone}, '
        f'classification_loss_function = {classification_loss_function}, class_weights = {use_weights}'
    )

    backbone = NETWORKS[backbone]
    classification_loss_function = CLASSIFICATION_LOSS_FUNCTIONS[classification_loss_function]
    class_weights = Augsburg15DetectionDataset.CLASS_WEIGHTS if use_weights else None

    model = ObjectDetector(
        num_classes=Augsburg15DetectionDataset.NUM_CLASSES,
        batch_size=batch_size,
        learning_rate=0.0001,
        timm_model=backbone,
        min_image_size=min_image_size,
        max_image_size=max_image_size,
        augmentations=[Augmentation.VERTICAL_FLIP, Augmentation.HORIZONTAL_FLIP],
        freeze_backbone=freeze_backbone,
        classification_loss_function=classification_loss_function,
        class_weights=class_weights
    )
    log_directory = 'logs'
    experiment_name = f'{experiment_name}#{get_git_revision_short_hash()}'
    logger = TensorBoardLogger(log_directory, experiment_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{log_directory}/{experiment_name}',
        save_last=True,
        save_top_k=1,
        monitor='validation_map_50',
        mode='max',
    )
    trainer = Trainer(
        max_epochs=40,
        logger=logger,
        callbacks=[checkpoint_callback],
        gpus=1,
        precision=16,
        progress_bar_refresh_rate=1000,
    )
    trainer.fit(
        model,
        train_dataloaders=model.train_dataloader(),
        val_dataloaders=model.val_dataloader(),
        ckpt_path=checkpoint_path,
    )


if __name__ == '__main__':
    start_experiment()
