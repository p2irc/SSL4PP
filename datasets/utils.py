import random
import warnings
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.filters import gaussian
from sklearn.model_selection import train_test_split
from torch import Tensor

from datasets.base import Dataset


def get_random_sample(
    path_target_pair: List[Tuple[str, int]],
    sample_size: Union[float, int],
    sampling_method: str,
    seed: int,
) -> List[Tuple[str, int]]:
    """Get a random sample from the dataset.

    Args:
        path_target_pair: List[Tuple[str, int]]
            A list of tuples containing the path to the image and the target value.
        sample_size: Union[float, int]
            The size of the sample to be drawn. If float, it is interpreted as the
            proportion of the dataset to include in the sample. If int, it is interpreted
            as the absolute number of samples to draw.
        sampling_method: str
            The sampling method to use. Can be either "stratified" or "uniform".
        seed: int
            The seed to use for the random number generator.

    Returns:
        List[Tuple[str, int]]
            A list of tuples containing the path to the image and the target value.

    """

    assert sampling_method in ["uniform", "stratified", None]

    subset = path_target_pair
    max_sample_size = len(subset)

    if sample_size is not None:
        if isinstance(sample_size, float):
            sample_size = int(max_sample_size * sample_size)

        if sample_size < max_sample_size:
            try:
                paths, targets = zip(*subset)  # unzip
                paths, _, targets, _ = train_test_split(
                    paths,
                    targets,
                    train_size=sample_size,
                    random_state=seed,
                    stratify=None if sampling_method == "uniform" else targets,
                )
                subset = list(zip(paths, targets))

            except:
                warnings.warn("Samples do not have labels.")
                subset, _ = train_test_split(
                    path_target_pair,
                    train_size=sample_size,
                    random_state=seed,
                    stratify=None,
                )

            print(
                f"Selected {len(subset)} samples from dataset with {max_sample_size} images."
            )
        else:
            warnings.warn(
                "Expected the sample_size to be less than the size of the dataset. \
                The full dataset will be used. Use sample_size=null to skip random sampling."
            )

    return subset


def plot_sample(
    dataset: Dataset,
    nrows_ncols: Optional[Union[int, Tuple[int, int]]] = (1, 3),
    with_transforms: Optional[bool] = False,
) -> None:
    """Plot a sample from the dataset.

    Args:
        dataset: Dataset
            The dataset to plot a sample from.
        nrows_ncols: Optional[Union[int, Tuple[int, int]]]
            The grid size to plot on. If int, the grid will be square. If tuple, the
            first element is the number of rows and the second element is the number of
            columns.
        with_transforms: Optional[bool]
            Whether to apply the transforms to the sample.

    Returns:
        None

    """
    assert isinstance(
        nrows_ncols, (tuple, int)
    ), f"Expected a Tuple or and integer, got {nrows_ncols}"
    if isinstance(nrows_ncols, int):
        nrows_ncols = (nrows_ncols, nrows_ncols)
    else:
        assert (
            len(nrows_ncols) == 2
        ), f"Expected a Tuple of length 2, got a Tuple of length {len(nrows_ncols)}"
    num_samples = nrows_ncols[0] * nrows_ncols[1]
    assert num_samples < len(
        dataset
    ), "Number of available images exceeds samples to be plotted."
    assert isinstance(dataset, Dataset)

    fig = plt.figure(figsize=(11.69, 8.27))  # landscape A4 paper size
    grid = ImageGrid(fig, 111, nrows_ncols=nrows_ncols, axes_pad=0.01, share_all=True)
    grid[0].get_yaxis().set_ticks([])  # remove yticks
    grid[0].get_xaxis().set_ticks([])  # remove xticks

    if not with_transforms:
        # temporarily disable the transform pipeline
        _temp_transforms = dataset.transform
        dataset.transform = None

    imgs = []
    for _ in range(num_samples):
        img, target = dataset[random.randint(0, len(dataset) - 1)]

        if isinstance(img, Tensor):
            img = img.T.numpy()

        if with_transforms:
            img = np.ascontiguousarray(img, dtype=np.uint8)

        # show bounding boxes on image
        if isinstance(target, dict) and "boxes" in target.keys():
            bboxes = target["boxes"]
            for (xmin, ymin, xmax, ymax) in bboxes:
                cv2.rectangle(
                    img,
                    (int(xmin.item()), int(ymin.item())),
                    (int(xmax.item()), int(ymax.item())),
                    (255, 0, 0),
                    3,
                )
        imgs.append(img)

    for ax, im in zip(grid, imgs):
        ax.imshow(im)
    plt.show()

    if not with_transforms:
        dataset.transform = _temp_transforms


def generate_density_map(
    keypoints: List, map_shape: np.ndarray, radius: Optional[int] = 5
) -> np.ndarray:
    """Generate a density map from a list of keypoints.

    Args:
        keypoints: List
            A list of keypoints.
        map_shape: np.ndarray
            The shape of the density map.
        radius: Optional[int]
            The radius of the gaussian kernel.

    Returns:
        np.ndarray
            The density map.

    """
    # from: https://towardsdatascience.com/objects-counting-by-estimating-a-density-map-with-convolutional-neural-networks-c01086f3b3ec

    label = np.zeros(map_shape, dtype=np.float32)

    for x, y in keypoints:
        label[int(y)][int(x)] = 100
    label = label / np.max(label)

    # apply a convolution with a Gaussian kernel
    sigma = (radius - 0.5) / 4
    label = gaussian(label, sigma=sigma)

    return label
