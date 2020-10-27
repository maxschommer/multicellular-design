
from enum import Enum

import numpy as np

from utils import make_sphere_node
from config import MICROMETER


LOW_RES_SPHERE = make_sphere_node(resolution=10, diameter=1)
HIGH_RES_SPHERE = make_sphere_node(
    resolution=20, diameter=1, rgba=np.asarray([0, 0, 1, .5]))


class BaseEnum(Enum):
    """Base Enum class gives enum values attributes of an id and data.
    """
    def __new__(cls, *args, **kwargs):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, id_, data):
        self.id = id_
        self.data = data


class GenGeomColor(BaseEnum):
    RED = 0, np.asarray([1, 0, 0])
    GREEN = 1, np.asarray([0, 1, 0])
    BLUE = 2, np.asarray([0, 0, 1])


class GenGeomAlpha(BaseEnum):
    p10 = 0, 0.1
    p20 = 1, 0.2
    p30 = 2, 0.3
    p40 = 3, 0.4
    p50 = 4, 0.5
    p60 = 5, 0.6
    p70 = 6, 0.7
    p80 = 7, 0.8
    p90 = 8, 0.9
    p100 = 9, 1.0


class GenGeomResolution(BaseEnum):
    low = 0, 10
    high = 1, 20


class FreeDict(dict):
    # called when trying to read a missing key
    def __missing__(self, key):
        self[key] = FreeDict()
        return self[key]

    # called during attribute access
    # note that this invokes __missing__ above
    def __getattr__(self, key):
        return self[key]

    # called during attribute assignment
    def __setattr__(self, key, value):
        self[key] = value


class SphereGenerator():
    def __init__(self):
        self._spheres = FreeDict()
        for resolution_enum in GenGeomResolution:
            for color_enum in GenGeomColor:
                for alpha_enum in GenGeomAlpha:
                    resolution = resolution_enum.data
                    color = color_enum.data
                    alpha = alpha_enum.data
                    self._spheres[resolution_enum.value][color_enum.value][alpha_enum.value] = make_sphere_node(
                        resolution=resolution,
                        diameter=1,
                        rgba=np.asarray([color[0], color[1], color[2], alpha]))

    def get_sphere(self, resolution: GenGeomResolution,
                   color: GenGeomColor,
                   alpha: GenGeomAlpha):
        return self._spheres[resolution.value][color.value][alpha.value]
