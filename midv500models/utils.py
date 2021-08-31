import re
from pathlib import Path
from typing import Optional, Any
from typing import Union, Dict, List, Tuple

import cv2
import numpy as np
import torch


def get_id2_file_paths(path: Union[str, Path]) -> Dict[str, Path]:
    return {x.stem: x for x in Path(path).glob("*.*")}


def get_samples(image_path: Path, mask_path: Path) -> List[Tuple[Path, Path]]:
    """Couple masks and images.

    Args:
        image_path:
        mask_path:

    Returns:
    """
    image2path = get_id2_file_paths(image_path)
    mask2path = get_id2_file_paths(mask_path)

    return [
        (image_file_path, mask2path[file_id])
        for file_id, image_file_path in image2path.items()
    ]


def load_checkpoint(
    file_path: Union[Path, str], rename_in_layers: Optional[dict] = None
) -> Dict[str, Any]:
    """Loads PyTorch checkpoint, optionally renaming layer names.

    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}
    Returns: dictionary with the filtered checkpoint.
    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    if rename_in_layers is not None:
        model_state_dict = checkpoint["state_dict"]

        result = {}
        for key, value in model_state_dict.items():
            for key_r, value_r in rename_in_layers.items():
                key = re.sub(key_r, value_r, key)

            result[key] = value

        checkpoint["state_dict"] = result

    return checkpoint


def mask_overlay(
    image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Overlay image and mask.

    Args:
        image:
        mask:
        color:

    Returns:

    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 1, 0.0)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img
