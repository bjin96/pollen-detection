from enum import Enum
from typing import List

import timm
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import MultiScaleRoIAlign
from torch.nn.functional import cross_entropy

from data_loading.load_augsburg15 import collate_augsburg15_detection, Augsburg15Dataset
from loss.focal_loss import calculate_focal_loss
from model_definition.anchor_utils import AnchorGenerator
from model_definition.faster_rcnn import FasterRCNN
from models.timm_adapter import Network, TimmBackboneWithFPN


class ClassificationLoss(Enum):
    CROSS_ENTROPY = cross_entropy
    FOCAL = calculate_focal_loss


class ObjectDetector(LightningModule):

    def __init__(
            self,
            num_classes: int,
            batch_size: int,
            learning_rate: float,
            timm_model: Network,
            min_image_size: int,
            max_image_size: int,
            train_dataset: Augsburg15Dataset,
            validation_dataset: Augsburg15Dataset,
            test_dataset: Augsburg15Dataset,
            freeze_backbone: bool = False,
            classification_loss_function: ClassificationLoss = ClassificationLoss.CROSS_ENTROPY,
            class_weights: List[float] = None,
    ):
        """
        Creates a Faster R-CNN model with a pre-trained backbone from timm
        (https://github.com/rwightman/pytorch-image-models) and feature pyramid network.

        Args:
            num_classes: Number of classes to classify objects in the image. Includes an additional background class.
            batch_size: Size of the batch per training step.
            learning_rate: Learning rate for the optimizer.
            timm_model: Identifier for a pre-trained timm backbone.
            min_image_size: Minimum size to which the image is scaled.
            max_image_size: Maximum size to which the image is scaled.
            train_dataset: Dataset to use for training.
            validation_dataset: Dataset to use for validation.
            test_dataset: Dataset to use for testing.
            freeze_backbone: Whether to freeze the backbone for the training.
            classification_loss_function: Loss function to apply for the classification loss part of the ROI heads.
            """
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.timm_model = timm_model
        self.freeze_backbone = freeze_backbone
        self.model = self.define_model(min_image_size, max_image_size, classification_loss_function, class_weights)
        self.validation_mean_average_precision = MeanAveragePrecision(class_metrics=True, compute_on_step=False)
        self.test_mean_average_precision = MeanAveragePrecision(class_metrics=True, compute_on_step=False)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

    def define_model(self, min_image_size, max_image_size, classification_loss_function, class_weights):
        feature_extractor = timm.create_model(
            self.timm_model.value,
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )
        out_indices = feature_extractor.feature_info.out_indices
        out_channels = 256
        in_channels = [i['num_chs'] for i in feature_extractor.feature_info.info[1:]]

        if self.freeze_backbone:
            # Freeze similarly to pytorch model.
            for child in list(feature_extractor.children())[:-1]:
                for p in child.parameters():
                    p.requires_grad_(False)

            for p in list(feature_extractor.children())[-1][:3].parameters():
                p.requires_grad_(False)

        backbone = TimmBackboneWithFPN(
            backbone=feature_extractor,
            in_channels_list=in_channels,
            out_channels=out_channels
        )

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),) * (len(out_indices) + 1),
            aspect_ratios=((0.5, 1.0, 2.0),) * (len(out_indices) + 1)
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=list(range(len(out_indices))),
            # featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, device=self.device)
        return FasterRCNN(
            backbone,
            num_classes=self.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=min_image_size,
            max_size=max_image_size,
            classification_loss_function=classification_loss_function,
            class_weights=class_weights
        )

    def forward(self, x, y=None):
        return self.model(x, y)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, targets = batch

        loss_dict = self(images, targets)

        total_loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', total_loss, on_step=True, batch_size=self.batch_size)
        return total_loss

    def validation_step(self, batch, batch_idx) -> None:
        images, targets = batch
        predictions = self(images, targets)
        self.validation_mean_average_precision(predictions, targets)

    def on_validation_epoch_end(self) -> None:
        metrics = self.validation_mean_average_precision.compute()
        self._log_metrics(metrics, mode='validation')
        self.validation_mean_average_precision.reset()

    def test_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self(images, targets)
        self.test_mean_average_precision(predictions, targets)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_mean_average_precision.compute()
        self._log_metrics(metrics, mode='test')
        self.test_mean_average_precision.reset()

    def _log_metrics(self, mean_average_precision, mode):
        for index, value in enumerate(mean_average_precision['map_per_class']):
            mean_average_precision[f'map_per_class_{index}'] = value
        for index, value in enumerate(mean_average_precision['mar_100_per_class']):
            mean_average_precision[f'mar_100_per_class_{index}'] = value
        del mean_average_precision['map_per_class']
        del mean_average_precision['mar_100_per_class']
        for name, metric in mean_average_precision.items():
            self.log(f'{mode}_{name}', metric, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer, factor=0.5, patience=4),
                'monitor': 'validation_map_50',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int):
        optimizer.zero_grad(set_to_none=True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_augsburg15_detection,
            shuffle=True,
            num_workers=2
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_augsburg15_detection,
            num_workers=2
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_augsburg15_detection,
            num_workers=2
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
