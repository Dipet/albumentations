from __future__ import division

import math
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from .transforms_interface import KeypointsArray, KeypointsInternalType, KeypointType
from .utils import DataProcessor, Params, ensure_internal_format

__all__ = [
    "angle_to_2pi_range",
    "check_keypoints",
    "convert_keypoints_from_internal",
    "convert_keypoints_to_internal",
    "filter_keypoints",
    "KeypointsProcessor",
    "KeypointParams",
    "use_keypoints_ndarray",
]

keypoint_formats = {"xy", "yx", "xya", "xys", "xyas", "xysa"}


def angle_to_2pi_range(angle: Union[np.ndarray, float]):
    two_pi = 2 * math.pi
    return angle % two_pi


def split_keypoints_targets(keypoints: Sequence[KeypointType], coord_length: int) -> Tuple[np.ndarray, List[Any]]:
    kps_array, targets = [], []
    for kp in keypoints:
        kps_array.append(kp[:coord_length])
        targets.append(kp[coord_length:])
    return np.array(kps_array, dtype=float), targets


def use_keypoints_ndarray(return_array: bool = True) -> Callable:
    """Decorate a function and return a decorator.
    Since most transformation functions does not alter the amount of bounding boxes, only update the internal
    keypoints' coordinates, thus this function provides a way to interact directly with
    the KeypointsInternalType's internal array member.

    Args:
        return_array (bool): whether the return of the decorated function is a KeypointsArray.

    Returns:
        Callable: A decorator function.
    """

    def dec(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(
            keypoints: Union[KeypointsInternalType, np.ndarray], *args, **kwargs
        ) -> Union[KeypointsInternalType, np.ndarray]:
            if isinstance(keypoints, KeypointsInternalType):
                ret = func(keypoints.array, *args, **kwargs)
                if not return_array:
                    return ret
                if not isinstance(ret, np.ndarray):
                    raise TypeError(f"The return from {func.__name__} must be a numpy ndarray.")
                keypoints.array = ret
            elif isinstance(keypoints, np.ndarray):
                keypoints = func(keypoints.astype(float), *args, **kwargs)
            return keypoints

        return wrapper

    return dec


class KeypointParams(Params):
    """
    Parameters of keypoints

    Args:
        format (str): format of keypoints. Should be 'xy', 'yx', 'xya', 'xys', 'xyas', 'xysa'.

            x - X coordinate,

            y - Y coordinate

            s - Keypoint scale

            a - Keypoint orientation in radians or degrees (depending on KeypointParams.angle_in_degrees)
        label_fields (list): list of fields that are joined with keypoints, e.g labels.
            Should be same type as keypoints.
        remove_invisible (bool): to remove invisible points after transform or not
        angle_in_degrees (bool): angle in degrees or radians in 'xya', 'xyas', 'xysa' keypoints
        check_each_transform (bool): if `True`, then keypoints will be checked after each dual transform.
            Default: `True`
    """

    def __init__(
        self,
        format: str,  # skipcq: PYL-W0622
        label_fields: Optional[Sequence[str]] = None,
        remove_invisible: bool = True,
        angle_in_degrees: bool = True,
        check_each_transform: bool = True,
    ):
        super(KeypointParams, self).__init__(format, label_fields)
        self.remove_invisible = remove_invisible
        self.angle_in_degrees = angle_in_degrees
        self.check_each_transform = check_each_transform

    def _to_dict(self) -> Dict[str, Any]:
        data = super(KeypointParams, self)._to_dict()
        data.update(
            {
                "remove_invisible": self.remove_invisible,
                "angle_in_degrees": self.angle_in_degrees,
                "check_each_transform": self.check_each_transform,
            }
        )
        return data

    @classmethod
    def is_serializable(cls) -> bool:
        return True

    @classmethod
    def get_class_fullname(cls) -> str:
        return "KeypointParams"


class KeypointsProcessor(DataProcessor):
    def __init__(self, params: KeypointParams, additional_targets: Optional[Dict[str, str]] = None):
        assert isinstance(params, KeypointParams)
        super().__init__(params, additional_targets)

    @property
    def default_data_name(self) -> str:
        return "keypoints"

    def ensure_data_valid(self, data: Dict[str, Any]) -> None:
        if self.params.label_fields:
            if not all(i in data.keys() for i in self.params.label_fields):
                raise ValueError(
                    "Your 'label_fields' are not valid - them must have same names as params in "
                    "'keypoint_params' dict"
                )

    def ensure_transforms_valid(self, transforms: Sequence[object]) -> None:
        # IAA-based augmentations supports only transformation of xy keypoints.
        # If your keypoints formats is other than 'xy' we emit warning to let user
        # be aware that angle and size will not be modified.

        try:
            from albumentations.imgaug.transforms import DualIAATransform
        except ImportError:
            # imgaug is not installed so we skip imgaug checks.
            return

        if self.params.format is not None and self.params.format != "xy":
            for transform in transforms:
                if isinstance(transform, DualIAATransform):
                    warnings.warn(
                        "{} transformation supports only 'xy' keypoints "
                        "augmentation. You have '{}' keypoints format. Scale "
                        "and angle WILL NOT BE transformed.".format(transform.__class__.__name__, self.params.format)
                    )
                    break

    def filter(self, data, rows: int, cols: int, target_name: str):
        self.params: KeypointParams
        data = filter_keypoints(data, rows, cols, remove_invisible=self.params.remove_invisible)
        return data

    def check(self, data, rows: int, cols: int) -> None:
        check_keypoints(data, rows, cols)

    def convert_from_internal(self, data, rows: int, cols: int):
        return convert_keypoints_from_internal(
            data,
            self.params.format,
            rows,
            cols,
            check_validity=self.params.remove_invisible,
            angle_in_degrees=self.params.angle_in_degrees,
        )

    def convert_to_internal(self, data, rows: int, cols: int):
        return convert_keypoints_to_internal(
            data,
            self.params.format,
            rows,
            cols,
            check_validity=self.params.remove_invisible,
            angle_in_degrees=self.params.angle_in_degrees,
        )


@use_keypoints_ndarray(return_array=False)
def check_keypoints(keypoints: KeypointsArray, rows: int, cols: int) -> None:
    """Check if keypoints boundaries are less than image shapes"""

    if not len(keypoints):
        return

    row_idx, *_ = np.where(
        ~np.logical_and(0 <= keypoints[..., 0], keypoints[..., 0] < cols)
        | ~np.logical_and(0 <= keypoints[..., 1], keypoints[..., 1] < rows)
    )
    if row_idx:
        raise ValueError(
            f"Expected keypoints `x` in the range [0.0, {cols}] and `y` in the range [0.0, {rows}]. "
            f"Got {keypoints[row_idx]}."
        )
    row_idx, *_ = np.where(~np.logical_and(0 <= keypoints[..., 2], keypoints[..., 2] < 2 * math.pi))
    if len(row_idx):
        raise ValueError(f"Keypoint angle must be in range [0, 2 * PI). Got: {keypoints[row_idx, 2]}.")


@ensure_internal_format
def filter_keypoints(
    keypoints: KeypointsInternalType, rows: int, cols: int, remove_invisible: bool
) -> KeypointsInternalType:
    """Remove keypoints that are not visible.
    Args:
        keypoints (KeypointsInternalType): A batch of keypoints in `x, y, a, s` format.
        rows (int): Image height.
        cols (int): Image width.
        remove_invisible (bool): whether to remove invisible keypoints or not.

    Returns:
        KeypointsInternalType: A batch of keypoints in `x, y, a, s` format.

    """
    if not remove_invisible:
        return keypoints
    if not len(keypoints):
        return keypoints

    x = keypoints.array[..., 0]
    y = keypoints.array[..., 1]
    idx, *_ = np.where(np.logical_and(0 <= x, x < cols) & np.logical_and(0 <= y, y < rows))

    return keypoints[idx] if len(idx) != len(keypoints) else keypoints


@ensure_internal_format
def convert_keypoints_to_internal(
    keypoints: Sequence[KeypointType],
    source_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> KeypointsInternalType:
    """Convert a batch of keypoints from source format to the format used by albumentations.

    Args:
        keypoints (Sequence[KeypointType]): a batch of keypoints in source format.
        source_format (str):
        rows (int):
        cols (int):
        check_validity (bool):
        angle_in_degrees (bool):

    Returns:
        KeypointsArray: A batch of keypoints in `albumentations` format, which is [x, y, a, s].

    Raises:
        ValueError: Unknown keypoint format is given.

    """
    if source_format not in keypoint_formats:
        raise ValueError(f"Unknown source_format {source_format}. " f"Supported formats are {keypoint_formats}.")
    if not len(keypoints):
        return KeypointsInternalType()

    if source_format == "xy":
        kps_array, targets = split_keypoints_targets(keypoints, coord_length=2)
        kps_array = np.concatenate((kps_array[..., :2], np.zeros_like(kps_array[..., :2])), axis=1)
    elif source_format == "yx":
        kps_array, targets = split_keypoints_targets(keypoints, coord_length=2)
        kps_array = np.concatenate((kps_array[..., :2][..., ::-1], np.zeros_like(kps_array[..., :2])), axis=1)
    elif source_format == "xya":
        kps_array, targets = split_keypoints_targets(keypoints, coord_length=3)
        kps_array = np.concatenate((kps_array[..., :3], np.zeros_like(kps_array[..., 0][..., np.newaxis])), axis=1)
    elif source_format == "xys":
        kps_array, targets = split_keypoints_targets(keypoints, coord_length=3)
        kps_array = np.insert(kps_array[..., :3], 2, np.zeros_like(kps_array[..., 0]), axis=1)
    elif source_format == "xyas":
        kps_array, targets = split_keypoints_targets(keypoints, coord_length=4)
        kps_array = kps_array[..., :4]
    elif source_format == "xysa":
        kps_array, targets = split_keypoints_targets(keypoints, coord_length=4)
        kps_array = kps_array[..., [0, 1, 3, 2]]
    else:
        raise ValueError(f"Unsupported source format. Got {source_format}.")

    if angle_in_degrees:
        kps_array[..., 2] = np.radians(kps_array[..., 2])
    kps_array[..., 2] = angle_to_2pi_range(kps_array[..., 2])

    kps_type = KeypointsInternalType(array=kps_array, targets=targets)
    if check_validity:
        check_keypoints(kps_type, rows=rows, cols=cols)
    return kps_type


def convert_keypoints_from_internal(
    keypoints: KeypointsInternalType,
    target_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> Sequence[KeypointType]:
    """Convert a batch of keypoints from `albumentations` format to target format.

    Args:
        keypoints (KeypointsInternalType): A batch of keypoints in `albumentations` format, which is [x, y, a, s].
        target_format (str):
        rows (int):
        cols (int):
        check_validity (bool):
        angle_in_degrees (bool):

    Returns:
        Sequence[KeypointType]: A batch of keypoints in target format.

    Raises:
        ValueError: Unknown target format is given.

    """
    if target_format not in keypoint_formats:
        raise ValueError(f"Unknown target_format {target_format}. " f"Supported formats are: {keypoint_formats}.")

    if not len(keypoints):
        return []

    keypoints.array[..., 2] = angle_to_2pi_range(keypoints.array[..., 2])
    if check_validity:
        check_keypoints(keypoints, rows, cols)

    if angle_in_degrees:
        keypoints.array[..., 2] = np.degrees(keypoints.array[..., 2])

    ret = []
    if target_format == "xy":
        for kp_array, target in zip(keypoints.array[..., :2], keypoints.targets):
            ret.append(cast(KeypointType, tuple(kp_array) + tuple(target)))
    elif target_format == "yx":
        for kp_array, target in zip(keypoints.array[..., [1, 0]], keypoints.targets):
            ret.append(cast(KeypointType, tuple(kp_array) + tuple(target)))
    elif target_format == "xya":
        for kp_array, target in zip(keypoints.array[..., [0, 1, 2]], keypoints.targets):
            ret.append(cast(KeypointType, tuple(kp_array) + tuple(target)))
    elif target_format == "xys":
        for kp_array, target in zip(keypoints.array[..., [0, 1, 3]], keypoints.targets):
            ret.append(cast(KeypointType, tuple(kp_array) + tuple(target)))
    elif target_format == "xyas":
        for kp_array, target in zip(keypoints.array, keypoints.targets):
            ret.append(cast(KeypointType, tuple(kp_array) + tuple(target)))
    elif target_format == "xysa":
        for kp_array, target in zip(keypoints.array[..., [0, 1, 3, 2]], keypoints.targets):
            ret.append(cast(KeypointType, tuple(kp_array) + tuple(target)))
    else:
        raise ValueError(f"Invalid target format. Got: {target_format}.")

    return ret
