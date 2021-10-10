from typing import Dict, Hashable, Mapping, Union

import numpy as np
import torch
from ..builder import PIPELINES
from monai.config import KeysCollection, IndexSelection
from monai.transforms.compose import MapTransform
from monai.transforms.utility.array import CastToType

from monai.utils import ensure_tuple_rep

@PIPELINES.register_module()
class CastToTyped_mmseg(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.CastToType`.
    """
    def __init__(
            self,
            keys: KeysCollection,
            dtypeinds: IndexSelection,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype: convert image to this data type, default is `np.float32`.
                it also can be a sequence of np.dtype or torch.dtype,
                each element corresponds to a key in ``keys``.

        """
        MapTransform.__init__(self, keys)
        dtypelist = [np.int8, np.float32]
        dtypes = []
        for idx, Dtype in enumerate(dtypeinds):
            # print(idx, dtypelist[0])
            dtypes.append(dtypelist[idx])
        # dtype: Union[Sequence[Union[np.dtype, torch.dtype]], np.dtype, torch.dtype] = np.float32,
        self.dtype = ensure_tuple_rep(dtypes, len(self.keys))
        self.converter = CastToType()

    def __call__(
            self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[key] = self.converter(d[key], dtype=self.dtype[idx])

        return d

