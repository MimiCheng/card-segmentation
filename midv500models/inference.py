import argparse
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import yaml
import pytorch_lightning as pl
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from albumentations.core.serialization import from_dict
from iglovikov_helper_functions.utils.image_utils import unpad
from iglovikov_helper_functions.utils.mask_utils import remove_small_connected_binary
from pytorch_toolbelt.inference import tta

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from midv500models.dataloaders import SegmentationDatasetTest
# from midv500models.train import SegmentDocs
from midv500models.utils import load_checkpoint, mask_overlay
from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss
from pytorch_toolbelt.inference.tta import TTAWrapper, d4_image2mask, fliplr_image2mask, MultiscaleTTA

from albumentations.core.serialization import from_dict
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from pytorch_lightning.loggers import NeptuneLogger

from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from midv500models.dataloaders import SegmentationDataset
from midv500models.metrics import binary_mean_iou
from midv500models.utils import get_samples, load_checkpoint


def fill_small_holes(mask: np.ndarray, min_area: int) -> np.ndarray:
    inverted_mask = 1 - mask
    inverted_mask = remove_small_connected_binary(inverted_mask, min_area=min_area)
    return 1 - inverted_mask


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    arg("-i", "--image_path", type=Path, help="Path to images.", required=True)
    arg("-o", "--output_path", type=Path, help="Path to images.", required=True)
    arg("-w", "--checkpoint_path", type=Path, help="Path to weights.", required=True)
    return parser.parse_args()

class SegmentDocs(pl.LightningModule):
    def __init__(self, hparams):
        super(SegmentDocs, self).__init__()
        self.hparams = hparams
        self.model = object_from_dict(self.hparams["model"])

        if "resume_from_checkpoint" in self.hparams:
            corrections: Dict[str, str] = {"model.": ""}

            checkpoint = load_checkpoint(
                file_path=self.hparams["resume_from_checkpoint"],
                rename_in_layers=corrections,
            )
            self.model.load_state_dict(checkpoint["state_dict"])

        self.losses = [
            ("jaccard", 0.1, JaccardLoss(mode="binary", from_logits=True)),
            ("focal", 0.9, BinaryFocalLoss()),
        ]

    @property
    def hparams(self):
        return self._hparams

    @hparams.setter
    def hparams(self, hparams):
        self._hparams = hparams

    def forward(self, batch):
        return self.model(batch)

    def prepare_data(self):
        self.train_samples = get_samples(
            Path(self.hparams["data_path"]) / "images",
            Path(self.hparams["data_path"]) / "masks",
        )

    def train_dataloader(self):
        train_aug = from_dict(self.hparams["train_aug"])

        result = DataLoader(
            SegmentationDataset(self.train_samples, train_aug),
            batch_size=self.hparams["train_parameters"]["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        return result

    def val_dataloader(self):
        val_aug = from_dict(self.hparams["val_aug"])

        result = DataLoader(
            SegmentationDataset(self.train_samples, val_aug),
            batch_size=self.hparams["val_parameters"]["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        return result

    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams["optimizer"],
            params=filter(lambda x: x.requires_grad, self.model.parameters()),
        )

        scheduler = object_from_dict(self.hparams["scheduler"], optimizer=optimizer)
        self.optimizers = [optimizer]  # skipcq: PYL-W0201

        return self.optimizers, [scheduler]

    def training_step(self, batch, batch_idx):
        features = batch["features"]
        masks = batch["masks"]

        logits = self.forward(features)

        total_loss = 0

        for loss_name, weight, loss in self.losses:
            ls_mask = loss(logits, masks)
            total_loss += weight * ls_mask
            self.log(
                f"train_mask_{loss_name}",
                ls_mask,
                on_epoch=True,
                on_step=True,
                logger=True,
                prog_bar=True,
            )

        self.log(
            "total_loss",
            total_loss,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "lr",
            self._get_current_lr(),
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )

        return total_loss

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]
        return torch.Tensor([lr])[0].cuda()

    def validation_step(self, batch, batch_idx):
        features = batch["features"]
        masks = batch["masks"]

        logits = self.forward(features)

        for loss_name, _, loss in self.losses:
            self.log(
                f"val_mask_{loss_name}",
                loss(logits, masks),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        self.log(
            "val_iou",
            binary_mean_iou(logits, masks),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

def main():
    args = get_args()

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    model = SegmentDocs(hparams)

    test_file_names = sorted(Path(args.image_path).glob("*.jpg"))

    test_mask_path = args.output_path / "masks"
    test_vis_path = args.output_path / "vis"

    test_mask_path.mkdir(exist_ok=True, parents=True)
    test_vis_path.mkdir(exist_ok=True, parents=True)

    test_aug = from_dict(hparams["test_aug"])

    dataloader = DataLoader(
        SegmentationDatasetTest(test_file_names, test_aug),
        batch_size=hparams["test_parameters"]["batch_size"],
        num_workers=hparams["num_workers"],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    corrections: Dict[str, str] = {}

    checkpoint = load_checkpoint(file_path=args.checkpoint_path, rename_in_layers=corrections)  # type: ignore

    model.load_state_dict(checkpoint["state_dict"])
    model = nn.Sequential(model, nn.Sigmoid())

    # model = tta.MultiscaleTTA(model, [0.5, 2])
    # model = tta.MultiscaleTTAWrapper(model, [0.5, 2])
    model.eval()
    model = model.half()
    model.cuda()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            features = batch["features"]
            image_ids = batch["image_id"]

            preds = tta.fliplr_image2mask(model, features.half().cuda())
            # preds = fliplr_image2mask(model, features.half().cuda())

            for batch_id in range(features.shape[0]):
                image_id = image_ids[batch_id]
                mask = (preds[batch_id][0] > 0.5).cpu().numpy().astype(np.uint8)

                height = batch["height"][batch_id].item()
                width = batch["width"][batch_id].item()
                pads = batch["pads"][batch_id].cpu().numpy()

                mask = unpad(mask, pads)

                mask = remove_small_connected_binary(mask, min_area=100)
                mask = fill_small_holes(mask, min_area=100)

                mask = cv2.resize(
                    mask, (width, height), interpolation=cv2.INTER_NEAREST
                )

                cv2.imwrite(str(test_mask_path / f"{image_id}.png"), mask * 255)

                image = cv2.imread(str(args.image_path / f"{image_id}.jpg"))

                mask_image = mask_overlay(image, mask)

                cv2.imwrite(
                    str(test_vis_path / f"{image_id}.jpg"),
                    np.hstack(
                        [mask_image, cv2.cvtColor((mask * 255), cv2.COLOR_GRAY2BGR)]
                    ),
                )

if __name__ == "__main__":
    main()
