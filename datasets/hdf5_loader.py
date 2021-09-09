from typing import List, Tuple
from h5dataloader.common.structure import *
from h5dataloader.pytorch import HDF5Dataset
import h5dataloader.common.augmentation as Augmentation
import h5dataloader.common.crop as Crop
import h5dataloader.common.resize as Resize
import h5dataloader.common.normalization as Norm

AUG_RATE = 0.5

DATASET_RGB = 'rgb'
DATASET_LABEL = 'label'

class TrainHDF5Dataset(HDF5Dataset):
    def __init__(
        self,
        h5_paths: List[str],
        config: str,
        quiet: bool = False,
        block_size: int = 0,
        use_mods: Tuple[int, int] = None,
        visibility_filter_radius: int = 0,
        visibility_filter_threshold: float = 3.0
    ) -> None:
        super().__init__(h5_paths, config, quiet, block_size, use_mods, visibility_filter_radius, visibility_filter_threshold)
        output_size = self.minibatch[DATASET_RGB][CONFIG_TAG_SHAPE][:2]

        self.resize_img = Resize.Resize(output_size, INTER_LINEAR)
        self.resize_label = Resize.Resize(output_size, INTER_NEAREST)
        self.flip_h = Augmentation.Flip_2d(hflip_rate=0.5, vflip_rate=0.0)

        self.adjust_brightness = Augmentation.Adjust_brightness(factor_range=ValueRange(0.75, 1.25))
        self.adjust_contrast = Augmentation.Adjust_contrast(factor_range=ValueRange(0.75, 1.25))
        self.adjust_saturation = Augmentation.Adjust_saturation(factor_range=ValueRange(0.75, 1.25))
        self.adjust_hue = Augmentation.Adjust_hue(factor_range=ValueRange(0.75, 1.25))

        self.normalize_img = Norm.Normalization(TYPE_BGR8, ValueRange(0, 255))

        self.minibatch[DATASET_RGB][CONFIG_TAG_CREATEFUNC] = self.create_rgb
        self.minibatch[DATASET_LABEL][CONFIG_TAG_CREATEFUNC] = self.create_label

    def create_rgb(self, key: str, link_idx: int, minibatch_config: dict) -> np.ndarray:
        h5_key: str = self.create_h5_key(TYPE_BGR8, key, link_idx, minibatch_config)
        src: Data = Data(data=self.h5links[h5_key][()], type=self.h5links[h5_key].attrs[H5_ATTR_TYPE])
        resized = self.resize_img(h5_key, src)
        flipped = self.flip_h(h5_key, resized)
        adjusted_b = self.adjust_brightness(h5_key, flipped)
        adjusted_c = self.adjust_contrast(h5_key, adjusted_b)
        adjusted_s = self.adjust_saturation(h5_key, adjusted_c)
        adjusted_h = self.adjust_hue(h5_key, adjusted_s)
        normalized = self.normalize_img(h5_key, adjusted_h)
        return normalized.data

    def create_label(self, key: str, link_idx: int, minibatch_config: dict) -> np.ndarray:
        h5_key: str = self.create_h5_key(TYPE_SEMANTIC2D, key, link_idx, minibatch_config)
        src: Data = Data(data=self.h5links[h5_key][()], type=self.h5links[h5_key].attrs[H5_ATTR_TYPE])
        resized = self.resize_label(h5_key, src)
        flipped = self.flip_h(h5_key, resized)
        return flipped.data

class TestHDF5Dataset(HDF5Dataset):
    def __init__(
        self,
        h5_paths: List[str],
        config: str,
        quiet: bool = False,
        block_size: int = 0,
        use_mods: Tuple[int, int] = None,
        visibility_filter_radius: int = 0,
        visibility_filter_threshold: float = 3.0
    ) -> None:
        super().__init__(h5_paths, config, quiet, block_size, use_mods, visibility_filter_radius, visibility_filter_threshold)
        output_size = self.minibatch[DATASET_RGB][CONFIG_TAG_SHAPE][:2]

        self.resize_img = Resize.Resize(output_size, INTER_LINEAR)
        self.resize_label = Resize.Resize(output_size, INTER_NEAREST)

        self.normalize_img = Norm.Normalization(TYPE_BGR8, ValueRange(0, 255))

        self.minibatch[DATASET_RGB][CONFIG_TAG_CREATEFUNC] = self.create_rgb
        self.minibatch[DATASET_LABEL][CONFIG_TAG_CREATEFUNC] = self.create_label

    def create_rgb(self, key: str, link_idx: int, minibatch_config: dict) -> np.ndarray:
        h5_key: str = self.create_h5_key(TYPE_BGR8, key, link_idx, minibatch_config)
        src: Data = Data(data=self.h5links[h5_key][()], type=self.h5links[h5_key].attrs[H5_ATTR_TYPE])
        resized = self.resize_img(h5_key, src)
        normalized = self.normalize_img(h5_key, resized)
        return normalized.data

    def create_label(self, key: str, link_idx: int, minibatch_config: dict) -> np.ndarray:
        h5_key: str = self.create_h5_key(TYPE_SEMANTIC2D, key, link_idx, minibatch_config)
        src: Data = Data(data=self.h5links[h5_key][()], type=self.h5links[h5_key].attrs[H5_ATTR_TYPE])
        resized = self.resize_label(h5_key, src)
        return resized.data
