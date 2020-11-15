import copy
import numbers
from enum import Enum
import numpy as np
from panda3d.core import NodePath, TransparencyAttrib
from multicell.config import MICROMETER
from multicell.geom_gen import HIGH_RES_SPHERE, LOW_RES_SPHERE


# Signal types:
# Direct Contact
# Short distance signal
# Electrical signals (EM)
#   High speed, very little gradient

class CellSignalType(Enum):
    # Speed is in micrometers per second
    # Frequency is in signals per second
    CONTACT = 0
    SHORT_DISTANCE = 1
    ELECTRICAL = 2
    ELECTRO_MAGNETIC = 2


class PhysicalObject():

    def __init__(self,
                 position: np.ndarray = np.asarray([0, 0, 0], dtype=float),
                 velocity: np.ndarray = np.asarray([0, 0, 0], dtype=float),
                 orientation: np.ndarray = np.asarray(
                     [1, 0, 0, 0], dtype=float),
                 angular_velocity: np.ndarray = np.asarray(
                     [0, 0, 0], dtype=float),
                 mass: float = 1.0):
        self.position = position
        self.velocity = velocity
        self.orientation = orientation
        self.angular_velocity = angular_velocity
        self.mass = mass

    @ property
    def position(self):
        return self._position

    @ position.setter
    def position(self, value: np.ndarray):
        assert isinstance(value, np.ndarray), "Position must be a numpy array"
        assert value.shape == (3,), "Position vector must be of length 3."
        self._position = value

        if hasattr(self, "node_path"):
            self.node_path.setPos(value[0], value[1], value[2])


class SphericalObject(PhysicalObject):
    def __init__(self, diameter: float = 10 * MICROMETER,
                 **kwargs):
        super().__init__(**kwargs)
        self.diameter = diameter

    @property
    def diameter(self):
        return self._diameter

    @diameter.setter
    def diameter(self, value: float):
        assert isinstance(value, numbers.Number), "Diameter must be a float."

        self._diameter = value
        if hasattr(self, "node_path"):
            self.node_path.setScale(value, value, value)


class CellSignal(SphericalObject):
    signal_instance = 0

    def __init__(self, render: NodePath,
                 signal_type: CellSignalType,
                 last_signaled: float = 0,
                 resolution="low",
                 **kwargs
                 ):

        self.set_signal_properties(signal_type)
        self.resolution = resolution

        # TODO: Make this different than a sphere?
        if resolution == "low":
            self.node = copy.deepcopy(LOW_RES_SPHERE)
        if resolution == "high":
            self.node = copy.deepcopy(HIGH_RES_SPHERE)

        self.node_path = render.attachNewNode(self.node)
        self.node_path.setTransparency(TransparencyAttrib.MAlpha)
        self.node_path.setColor(*self.rgba)
        self.node_path.setDepthWrite(False)

        self.clock = 0
        self.last_signaled = last_signaled
        self.is_active = False

        self.signal_instance += 1
        super().__init__(**kwargs)

    def set_signal_properties(self, signal_type: CellSignalType):
        if signal_type == CellSignalType.CONTACT:
            self.speed = 1
            self.max_frequency = 0.5
            self.rgba = np.asarray([1, 0, 0, .2])
            self.max_diameter = 30 * MICROMETER
        if signal_type == CellSignalType.SHORT_DISTANCE:
            self.speed = 2
            self.max_frequency = 0.2
            self.rgba = np.asarray([0, 1, 0, 0.2])
            self.max_diameter = 40 * MICROMETER
        if signal_type == CellSignalType.ELECTRICAL:
            self.speed = 10
            self.max_frequency = 0.05
            self.rgba = np.asarray([0, 0, 1, 0.2])
            self.max_diameter = 100 * MICROMETER
        if signal_type == CellSignalType.ELECTRO_MAGNETIC:
            self.speed = 20
            self.max_frequency = 0.2
            self.rgba = np.asarray([1, 0, 1, 0.2])
            self.max_diameter = 400 * MICROMETER

    def should_emit(self):
        if (self.clock - self.last_signaled) > 1.0 / self.max_frequency:
            if self.is_active == False:
                return True
        return False

    @property
    def is_active(self):
        return self._is_active

    @is_active.setter
    def is_active(self, value: bool):
        assert isinstance(value, bool), "is_active must be a bool."

        self._is_active = value
        if hasattr(self, "node_path"):
            # print(f"Set {value}!")
            new_color = self.rgba.copy()
            if not self._is_active:
                new_color[3] = 0
                self.diameter = 10 * MICROMETER
            print(new_color)
            self.node_path.setColor(*new_color)


class Cell(SphericalObject):
    cell_instance = 0

    def __init__(self, render: NodePath,
                 diameter: float = 10 * MICROMETER,
                 resolution="low",
                 rgba=np.asarray([1, 1, 1, .5]),
                 **kwargs
                 ):
        # Define model geometry
        self.rgba = rgba
        self.resolution = resolution
        if resolution == "low":
            self.node = copy.deepcopy(LOW_RES_SPHERE)
        if resolution == "high":
            self.node = copy.deepcopy(HIGH_RES_SPHERE)
        self.node_path = render.attachNewNode(self.node)
        self.node_path.setTransparency(TransparencyAttrib.MAlpha)
        self.node_path.setColor(*rgba)
        self.node_path.setDepthWrite(False)

        self.signals = []

        # # Set position and orientation
        self.diameter = diameter
        super().__init__(**kwargs)

    def divide(self,
               render: NodePath,
               direction_preference: np.ndarray = np.asarray([1, 0, 0])):
        return Cell(render,
                    diameter=self.diameter,
                    position=self.position + direction_preference * self.diameter,
                    velocity=self.velocity,
                    orientation=self.orientation,
                    resolution=self.resolution,
                    rgba=[*np.abs(np.random.rand(3, 1)), 0.5])

    def update_signals(self, dt):
        sending_signals = []
        for signal in self.signals:
            signal.clock += dt
            if signal.should_emit():
                signal.last_signaled = signal.clock
                sending_signals.append(signal)
        return sending_signals

    def add_signal(self, render, signal_type: CellSignalType):
        self.signals.append(CellSignal(
            render=render,
            signal_type=signal_type,
            diameter=self.diameter,
            resolution=self.resolution,
            position=self.position,
            velocity=self.velocity,
            orientation=self.orientation
        ))

    def receive_signal(self, signal_type: CellSignalType):
        # Do stuff?
        pass
