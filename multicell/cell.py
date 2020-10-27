import copy
import numbers
import numpy as np
from panda3d.core import NodePath, TransparencyAttrib
from config import MICROMETER
from geom_gen import HIGH_RES_SPHERE, LOW_RES_SPHERE


class Cell():
    cell_instance = 0

    def __init__(self, render: NodePath,
                 diameter: float = 10 * MICROMETER,
                 position: np.ndarray = np.asarray([0, 0, 0], dtype=float),
                 velocity: np.ndarray = np.asarray([0, 0, 0], dtype=float),
                 orientation: np.ndarray = np.asarray(
                     [1, 0, 0, 0], dtype=float),
                 angular_velocity: np.ndarray = np.asarray(
                     [0, 0, 0], dtype=float),
                 mass: float = 1.0,
                 resolution="low",
                 rgba=np.asarray([1, 1, 1, .5])
                 ):
        # Define model geometry
        self.resolution = resolution
        if resolution == "low":
            self.node = copy.deepcopy(LOW_RES_SPHERE)
        if resolution == "high":
            self.node = copy.deepcopy(HIGH_RES_SPHERE)
        self.node_path = render.attachNewNode(self.node)
        self.node_path.setTransparency(TransparencyAttrib.MAlpha)
        self.node_path.setColor(*rgba)
        # self.node_path.setTransparency(True)
        self.node_path.setDepthWrite(False)

        # Set position and orientation
        self.diameter = diameter
        self.position = position
        self.velocity = velocity
        self.orientation = orientation
        self.angular_velocity = angular_velocity
        self.mass = mass

    def divide(self,
               render,
               direction_preference: np.ndarray = np.asarray([1, 0, 0])):
        return Cell(render,
                    diameter=self.diameter,
                    position=self.position + direction_preference * self.diameter,
                    velocity=self.velocity,
                    orientation=self.orientation,
                    resolution=self.resolution,
                    rgba=[*np.abs(np.random.rand(3, 1)), 0.5])

    @property
    def diameter(self):
        return self._diameter

    @diameter.setter
    def diameter(self, value: float):
        assert isinstance(value, numbers.Number), "Diameter must be a float."

        self._diameter = value
        self.node_path.setScale(value, value, value)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value: np.ndarray):
        assert isinstance(value, np.ndarray), "Position must be a numpy array"
        assert value.shape == (3,), "Position vector must be of length 3."
        self.node_path.setPos(value[0], value[1], value[2])
        self._position = value
