import copy
import numbers
from typing import List, Optional
from enum import Enum
import numpy as np
from panda3d.core import NodePath, TransparencyAttrib
from multicell.config import MICROMETER
from multicell.geom_gen import HIGH_RES_SPHERE, LOW_RES_SPHERE


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

        self.signal_type = signal_type
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
            self.speed = 10
            self.max_frequency = 0.5
            self.rgba = np.asarray([1, 0, 0, .2])
            self.max_diameter = 30 * MICROMETER
        if signal_type == CellSignalType.SHORT_DISTANCE:
            self.speed = 20
            self.max_frequency = 0.2
            self.rgba = np.asarray([0, 1, 0, 0.2])
            self.max_diameter = 40 * MICROMETER
        if signal_type == CellSignalType.ELECTRICAL:
            self.speed = 100
            self.max_frequency = 0.05
            self.rgba = np.asarray([0, 0, 1, 0.2])
            self.max_diameter = 100 * MICROMETER
        if signal_type == CellSignalType.ELECTRO_MAGNETIC:
            self.speed = 200
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

    def reset(self):
        """Resets the signal to the inactive signal diameter.
        """
        self.diameter = 1 * MICROMETER

    def update(self, dt: float, position_kd_tree):
        initial_idxs = position_kd_tree.query_ball_point(
            self.position, self.diameter)
        received_cells = []
        if self.diameter < self.max_diameter:
            self.diameter = self.diameter + dt * self.speed
            new_idxs = position_kd_tree.query_ball_point(
                self.position, self.diameter)
            received_cells = np.setdiff1d(new_idxs, initial_idxs)

        else:
            self.is_active = False
            self.last_signaled = self.clock

        return received_cells


class Cell(SphericalObject):
    cell_instance = 0

    def __init__(self, render: NodePath,
                 diameter: float = 10 * MICROMETER,
                 resolution: str = "low",
                 rgba: np.ndarray = np.asarray([1, 1, 1, .5]),
                 code: str = "",
                 resources: Optional[dict] = None
                 ** kwargs
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
        self.signals_received = set()

        # Set cell state
        self.resources =

        # # Set position and orientation
        self.diameter = diameter
        super().__init__(**kwargs)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        for key in value:
            self._state[key] = value[key]

    def divide(self,
               render: NodePath,
               direction_preference: np.ndarray = np.asarray([1, 0, 0])) -> "Cell":
        """Divides the current cell by returning a new Cell object based on
        the current cell. The direction that division occurs is given by
        direction_preference vector.

        Args:
            render (NodePath): The render node to add the new Cell to.
            direction_preference (np.ndarray, optional): A 1x3 vector indicating
                which direction the cell should prefer to split in. Defaults to
                np.asarray([1, 0, 0]).

        Returns:
            Cell: The new Cell which is the result of division.
        """
        return Cell(render,
                    diameter=self.diameter,
                    position=self.position + direction_preference * self.diameter,
                    velocity=self.velocity,
                    orientation=self.orientation,
                    resolution=self.resolution,
                    rgba=[*np.abs(np.random.rand(3, 1)), 0.5],
                    code=self.code)

    def update_signals(self, dt: float) -> List[CellSignal]:
        """Increments all active signal's clocks, and then emits the signals
        which are ready to be emitted.

        Args:
            dt (float): The change in time since the last update call

        Returns:
            List[CellSignal]: A list of cell signals to be emitted.
        """
        sending_signals = []
        for signal in self.signals:
            signal.clock += dt
            if signal.should_emit():
                signal.last_signaled = signal.clock
                sending_signals.append(signal)
        return sending_signals

    def add_signal(self, render: NodePath, signal_type: CellSignalType):
        """Adds a new signal to the Cell which will emit periodically.

        Args:
            render (NodePath): The render node to add the signal to.
            signal_type (CellSignalType): The signal type to add.
        """
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
        self.signals_received.add(signal_type)

    def execute_user_code(self):
        compiled_code = compile(self.code, '<string>', 'exec')

        exec_globals = {
            "__builtins__": None
        }
        exec_locals = {
            "received_signals": self.signals_received,
            "state": self.state,
            "actions": {},
            "print": print
        }

        exec(compiled_code, exec_globals, exec_locals)

        actions = exec_locals["actions"]

    def perform_actions(self, actions):
        pass
