"""The Cell module. Contains Cells and Cell Signals
"""
import copy
from dataclasses import dataclass
import numbers
from typing import List, Optional, Dict, Any, Set
from enum import IntEnum
import numpy as np
from scipy.spatial import cKDTree
import quaternion

from panda3d.core import NodePath, TransparencyAttrib
from multicell.config import MICROMETER
from multicell.geom_gen import HIGH_RES_SPHERE, LOW_RES_SPHERE


class CellSignalType(IntEnum):
    # Speed is in micrometers per second
    # Frequency is in signals per second
    CONTACT = 0
    SHORT_DISTANCE = 1
    ELECTRICAL = 2
    ELECTRO_MAGNETIC = 2


@dataclass
class CellActions():
    divide: bool = False
    send_contact: bool = False
    send_short_distance: bool = False
    send_electrical: bool = False
    send_electro_magnetic: bool = False


class CellResources():
    """Represents the resources availiable to a Cell. Values are between 0 and
    1.
    """

    def __init__(self,
                 oxygen: float = 1.0,
                 carbon: float = 1.0,
                 nitrogen: float = 1.0,
                 energy: float = 1.0):
        """Initializes CellResources

        Args:
            oxygen (float, optional): The oxygen level of the cell. Defaults to
                1.0.
            carbon (float, optional): The carbon level of the cell. Defaults to
                1.0.
            nitrogen (float, optional): The nitrogen level of the cell. Defaults to
                1.0.
            energy (float, optional): The energy level of the cell. Defaults to
                1.0.
        """

        self.oxygen = oxygen
        self.carbon = carbon
        self.nitrogen = nitrogen
        self.energy = energy

    def divide(self):
        """Divide the resources by half
        """
        for key in self.__dict__:
            self.__dict__[key] *= 0.5

    def all_greater_than(self, value):
        for key in self.__dict__:
            if self.__dict__[key] < value:
                return False
        return True


class PhysicalObject():
    """A base class which contains basic physical properties of an object.
    """

    def __init__(self,
                 position: np.ndarray = np.asarray([0, 0, 0], dtype=float),
                 velocity: np.ndarray = np.asarray([0, 0, 0], dtype=float),
                 orientation: np.ndarray = np.quaternion(1, 0, 0, 0),
                 angular_velocity: np.ndarray = np.asarray(
                     [0, 0, 0], dtype=float),
                 mass: float = 1.0):
        """A physical object which has physical properties such as position,
        velocity, orientation and angular velocity.

        Args:
            position (np.ndarray, optional): A 1x3 vector representing the
                objects position. Defaults to np.asarray([0, 0, 0],
                    dtype=float).
            velocity (np.ndarray, optional): A 1x3 vector representing the
                objects velocity. Defaults to np.asarray([0, 0, 0],
                dtype=float).
            orientation (np.ndarray, optional): A quaternion representing
                the orientation. Defaults to np.quaternion(1, 0, 0, 0).
            angular_velocity (np.ndarray, optional): A 1x3 vector representing
                the axis of rotation and magnitude of the rotational velocity.
                Defaults to np.asarray( [0, 0, 0], dtype=float).
            mass (float, optional): The mass of the object. Defaults to 1.0.
        """
        self.position = position
        self.velocity = velocity
        self.orientation = orientation
        self.angular_velocity = angular_velocity
        self.mass = mass

    def set_nodepath_position(self, value: np.ndarray):
        """Override this method to change the nodepath position when position
        is set.

        Args:
            value (np.ndarray): A 1x3 vector representing the position.

        Raises:
            NotImplementedError: override this method
        """
        raise NotImplementedError

    @ property
    def position(self) -> np.ndarray:
        """Get the current position of the object

        Returns:
            np.ndarray: a 1x3 vector representing the position
        """
        return self._position

    @ position.setter
    def position(self, value: np.ndarray):
        """Set the position of the object.

        Args:
            value (np.ndarray): A 1x3 vector representing the position
        """
        assert isinstance(value, np.ndarray), "Position must be a numpy array"
        assert value.shape == (3,), "Position vector must be of length 3."
        self._position = value
        self.set_nodepath_position(value)


class SphericalObject(PhysicalObject):
    """A base class for a spherical physical object with an associated
    node-path.
    """

    def __init__(self, render: NodePath = None,
                 diameter: float = 10 * MICROMETER,
                 resolution="low",
                 **kwargs):
        """Initializes the spherical object with a model and physical properties
        ready to be used by the physics engine.

        Args:
            render (NodePath): The parent nodepath to attach to.
            diameter (float, optional): The diameter of the sphere. Defaults to
                10*MICROMETER.
            resolution (str, optional): The resolution of the sphere to use.
                Options are "low" and "high". Defaults to "low".
        """

        self.render = render
        self.resolution = resolution

        if resolution == "low":
            self.node = copy.deepcopy(LOW_RES_SPHERE)
        if resolution == "high":
            self.node = copy.deepcopy(HIGH_RES_SPHERE)

        self.node_path = self.render.attachNewNode(self.node)
        self.node_path.setTransparency(TransparencyAttrib.MAlpha)
        self.node_path.setDepthWrite(False)

        self.diameter = diameter

        super().__init__(**kwargs)

    @property
    def diameter(self) -> float:
        """Gets the diameter of the sphere.

        Returns:
            float: The diameter of the sphere.
        """
        return self._diameter

    @diameter.setter
    def diameter(self, value: float):
        """Sets the diameter of the sphere and resizes the model as well.

        Args:
            value (float): The new diameter of the sphere.
        """
        assert isinstance(value, numbers.Number), "Diameter must be a float."
        self._diameter = value
        self.node_path.setScale(value, value, value)

    def set_nodepath_position(self, value: np.ndarray):
        """Set the nodepath position. Overrides the abstract method in
        PhysicalObject, which is called by the position setter.

        Args:
            value (np.ndarray): A 1x3 array representing the new position
        """
        self.node_path.setPos(value[0], value[1], value[2])


class CellSignal(SphericalObject):
    """A Cell Signal is a sphere which expands can detect which other cells
    have most recently received the signal. Cell signals are produced by
    cells and have an associated recharge-time.
    """
    signal_instance = 0

    def __init__(self,
                 signal_type: CellSignalType,
                 last_signaled: float = 0,
                 inactive_diameter: float = 1.0,
                 **kwargs):
        """Initializes the CellSignal

        Args:
            signal_type (CellSignalType): The type of signal. This determines
                properties such as recharge rate and speed.
            last_signaled (float, optional): The time when the singal was last
                emited. Defaults to 0.
            inactive_diameter (float, optional): The diameter the signal begins
                with and resets to when inactive. It is advised that this does
                not exceed the diameter of the cell which is the origin of the
                signal.
        """
        super().__init__(**kwargs)

        self.inactive_diameter = inactive_diameter
        self.signal_type = signal_type
        self.set_signal_properties(signal_type)

        self.node_path.setColor(*self.rgba)

        self.clock = 0.0
        self.last_signaled = last_signaled
        self.is_active = False

        self.signal_instance += 1

    def set_signal_properties(self, signal_type: CellSignalType):
        """Sets the properties of various cell signal types. This sets speed,
        maximum frequency, display color and maximum diameter of the signal.

        Args:
            signal_type (CellSignalType): The type of signal to set properties
                for.
        """
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

    # TODO (maxschommer): Change this to "can_emit" and make emitting signals
    # fully controlled by cell code.
    def can_emit(self) -> bool:
        """Returns true if the "cooldown time" has been exceeded for the signal.

        Returns:
            bool: if the cell should emit this signal or not.
        """
        if (self.clock - self.last_signaled) > 1.0 / self.max_frequency:
            if not self.is_active:
                return True
        return False

    @property
    def is_active(self) -> bool:
        """The current activity state of the signal. A signal which is in the
        process of being sent is considered active.

        Returns:
            bool: True if the signal is actively being sent, False otherwise.
        """
        return self._is_active

    @is_active.setter
    def is_active(self, value: bool):
        """Set the activity of the signal. This should be set if the signal
        has recently finished sending, or has recently begun sending.

        Args:
            value (bool): The new activity state of the signal.
        """
        assert isinstance(value, bool), "is_active must be a bool."

        self._is_active = value

        new_color = self.rgba.copy()
        if not self._is_active:
            new_color[3] = 0
            self.diameter = 10 * MICROMETER
        self.node_path.setColor(*new_color)

    def reset(self):
        """Resets the signal to the inactive signal diameter.
        """
        self.diameter = self.inactive_diameter

    def update(self, d_t: float, position_kd_tree: cKDTree) -> np.ndarray:
        """Updates the CellSignal given a cKDTree with all cells positions and
        a time step. This finds all cells within it's current diameter. It then
        increases the diameter using d_t and evaluates the cells inside that
        diameter. The indices of the difference between the two is returned.

        Args:
            d_t (float): The time step to evaluate.
            position_kd_tree (cKDTree): The KDTree containing the positions of
                the cells.

        Returns:
            np.asarray: An array of Cell indices into the KDTree of cells which
                have received the signal at this time step.
        """

        # First get the set of cells within the signal radius
        initial_idxs = position_kd_tree.query_ball_point(
            self.position, self.diameter / 2)
        received_cells = np.asarray([])
        # If the diameter exceeds the maximum diameter, the signal dies
        if self.diameter < self.max_diameter:
            self.diameter = self.diameter + d_t * self.speed
            new_idxs = position_kd_tree.query_ball_point(
                self.position, self.diameter / 2)
            received_cells = np.setdiff1d(new_idxs, initial_idxs)
        else:
            self.is_active = False
            self.last_signaled = self.clock

        return received_cells


class Cell(SphericalObject):
    """A Cell which can divide, execute user code, and perform actions.
    """
    cell_instance = 0

    def __init__(self,
                 rgba: np.ndarray = np.asarray([1, 1, 1, .5]),
                 code: str = "",
                 resources: Optional[CellResources] = CellResources(),
                 ** kwargs):
        """Initializes the Cell object

        Args:
            rgba (np.ndarray, optional): The color of the cell a 1x4 array
                representing red, green, blue, alpha values from 0 to 1.
                Defaults to np.asarray([1, 1, 1, .5]).
            code (str, optional): The code to execute in the Cell. This is
                a string of Python code which has access to a limited set
                of variables. See `execute_code()` for more information.
                Defaults to "".
            resources (Optional[CellResources], optional): The current resources
                of the cell. Defaults to CellResources().
        """
        super().__init__(**kwargs)
        # Define model geometry
        self.rgba = rgba
        self.node_path.setColor(*rgba)

        self.code = code

        self.actions = CellActions()
        self.clock = 0.0
        self.sending_signals: List[CellSignal] = []
        self.signals: List[CellSignal] = []
        self.signals_received: Set[CellSignalType] = set()

        # Set cell state
        self.resources = resources

    @property
    def state(self) -> Dict[str, Any]:
        """Gives the current state of the cell, which consists of resources
        and the diameter.

        Returns:
            Dict[str, Any]: The state of the cell. Consists of a dictionary of
                cell properties, and values which are the state of the
                properties.
        """
        state = {
            "resources": self.resources,
            "diameter": self.diameter,
            "clock": self.clock
        }
        return state

    def can_divide(self):
        if self.resources.all_greater_than(0.75):
            return True
        return False

    def should_divide(self):
        if self.actions.divide:
            return True
        return False

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
        self.resources.divide()

        return Cell(render=render,
                    diameter=self.diameter,
                    position=self.position + direction_preference * self.diameter,
                    velocity=self.velocity,
                    orientation=self.orientation,
                    resolution=self.resolution,
                    rgba=[*np.abs(np.random.rand(3, 1)), 0.5],
                    resources=copy.deepcopy(self.resources),
                    code=self.code)

    def update(self, d_t: float):
        """Updates the state of the cell.

        Args:
            d_t (float): The change in time since the last update call
        """
        self.clock += d_t

        sending_signals = []
        for signal in self.signals:
            signal.clock += d_t
            if signal.can_emit():
                if (self.actions.send_short_distance and
                        signal.signal_type == CellSignalType.SHORT_DISTANCE):
                    signal.last_signaled = signal.clock
                    sending_signals.append(signal)
        self.sending_signals = sending_signals

    def get_signals(self) -> List[CellSignal]:
        """Emits the signals which are ready to be emitted. An update step
        should be performed first.

        Returns:
            List[CellSignal]: A list of cell signals to be emitted.
        """
        return self.sending_signals

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
        """Add a signal which will be received by the cell. This can then be
        acted on by the cell code.

        Args:
            signal_type (CellSignalType): The CellSignalType received.
        """
        self.signals_received.add(signal_type)

    def execute_code(self):
        """Executes cell code. A restricted set of Python is availiable for
        execution (no builtins). The cell has access to received signals,
        the state, and can edit properties of a dictionary "actions" to instruct
        the cell to execute certain actions.
        """
        compiled_code = compile(self.code, '<string>', 'exec')

        exec_globals = {
            "__builtins__": None
        }

        exec_locals = {
            "received_signals": self.signals_received,
            "state": self.state,
            "actions": CellActions(),
            "print": print
        }

        # pylint: disable=exec-used
        exec(compiled_code, exec_globals, exec_locals)

        actions = exec_locals["actions"]
        self.actions = actions
        self.perform_actions(actions)
        self.signals_received = set()

    def perform_actions(self, actions: Dict[str, Any]):
        """Execute the actions if possible and alert invalid actions

        Args:
            actions (Dict[str, Any]): The actions the cell should perform
        """
        # if hasattr(self, "_last_actions"):
        #     if actions != self._last_actions:
        #         print(actions)
        # else:
        if actions.divide:
            print(actions)
        # self._last_actions = actions
