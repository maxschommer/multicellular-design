"""Entry point for running multicellular. To run, from the root directory
of this repository, run:

python -m multicell.run
"""
import sys

from direct.showbase.ShowBase import ShowBase


from panda3d.core import (
    DirectionalLight, OrthographicLens, GraphicsWindow, PythonTask
)
import numpy as np

from multicell.config import MICROMETER
from multicell.cell import Cell, CellSignalType
from multicell.physics import CellPhysics
from multicell.utils import make_rand_vector


class MultiCell(ShowBase):
    """The Root class of Multicellular. This instantiates all necessary
    components for multicellular to run, and can be run by:

        >>> app = MultiCell()
        >>> app.run()
    """

    def __init__(self):
        ShowBase.__init__(self)

        self.disableMouse()
        self.enableParticles()

        self.lens = OrthographicLens()
        # Or whatever is appropriate for your scene
        self.lens.setFilmSize(60, 45)
        self.cam.node().setLens(self.lens)

        self.camera.setPos(0, 0, 80 * MICROMETER)
        self.camera.setHpr(0, -90, 0)

        dlight = DirectionalLight('main_dlight')
        dlight.setColor((1, 1, 1, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(0, -60, 0)
        self.render.setLight(dlnp)

        self.last_mouse = [0, 0]
        self.mouse_state = {
            "mouse3": "up"
        }

        self.max_timestep = 1 / 20  # The maximum timestep size

        self.last_split = 0
        self.taskMgr.add(self.update, "update")
        self.taskMgr.add(self.drag_camera, "drag_camera")

        # Listen set window event listener
        self.accept("window-event", self.handle_window_events)
        self.configure_cam_control()
        self.initialize_cells()
        self.cell_physics = CellPhysics(
            self.cells, opposing_force=200, attraction_force=50, ignore_ratio=1.1)

    def initialize_cells(self):
        """Initialize cells -- Primarily for debugging purposes
        """
        code_1 = ("if state['clock']%10 > 9:\n"
                  "   actions.send_short_distance = True")

        cell_1 = Cell(render=self.render, resolution="high",
                      rgba=np.asarray([0, 0, 1, 0.5]), code=code_1)
        cell_1.add_signal(
            render=self.render, signal_type=CellSignalType.SHORT_DISTANCE)

        code_2 = ("for signal in received_signals:\n"
                  "   if signal == 1:\n"
                  "       actions.divide = True")
        cell_2 = Cell(render=self.render, resolution="high",
                      rgba=np.asarray([1, 0, 0, 0.5]), position=np.asarray([0, 5, 0]), code=code_2)
        self.cells = [cell_1, cell_2]

    def configure_cam_control(self) -> None:
        """Configures camera control event handlers for mouse and arrow keys.
        """

        right_dir = np.asarray([-1, 0, 0])
        left_dir = np.asarray([1, 0, 0])
        up_dir = np.asarray([0, -1, 0])
        down_dir = np.asarray([0, 1, 0])
        in_dir = np.asarray([0, 0, -1])
        out_dir = np.asarray([0, 0, 1])

        self.accept('arrow_right', self.control_camera, [right_dir])
        self.accept('arrow_left', self.control_camera, [left_dir])
        self.accept('arrow_up', self.control_camera, [up_dir])
        self.accept('arrow_down', self.control_camera, [down_dir])

        self.accept('arrow_right-repeat', self.control_camera, [right_dir])
        self.accept('arrow_left-repeat', self.control_camera, [left_dir])
        self.accept('arrow_up-repeat', self.control_camera, [up_dir])
        self.accept('arrow_down-repeat', self.control_camera, [down_dir])

        self.accept('wheel_up', self.control_camera, [in_dir])
        self.accept('wheel_down', self.control_camera, [out_dir])
        self.accept("mouse3", self.mouse_button, ["mouse3"])
        self.accept("mouse3-up", self.mouse_button, ["mouse3-up"])

    def mouse_button(self, key: str):
        """Sets the current mouse state based on the mouse key event.

        Args:
            key (str): a key event string.
        """
        if key == "mouse3":
            self.mouse_state["mouse3"] = "down"
        if key == "mouse3-up":
            self.mouse_state["mouse3"] = "up"

    def control_camera(self, move_vec: np.ndarray,
                       move_speed: float = 1,
                       zoom_speed: float = 0.1) -> None:
        """Controls the camera to move in a given direction at a given speed.
        The direction is described by a 3D vector, where the Z component is
        interpreted as zoom (z movement).

        Args:
            move_vec (np.ndarray): A 1x3 vector indicating the direction to
                move the camera. This does not need to be a normalized vector,
                but the values of the vector will be element-wise multiplied
                by move_speed to calculate the new camera position.
            move_speed (float, optional): The factor to multiply the move
                vector by (x and y values). Defaults to 1.
            zoom_speed (float, optional): The factor to multiply the zoom (film
                resize) by. This is the z component of move_vec. Defaults to
                0.1.
        """

        cur_pos = np.asarray(self.camera.getPos())
        cur_film_size = self.lens.getFilmSize()
        new_pos = cur_pos + move_vec * move_speed
        new_film_size = cur_film_size + \
            cur_film_size * move_vec[2] * zoom_speed
        self.camera.setPos(new_pos[0], new_pos[1], cur_pos[2])
        self.lens.setFilmSize(*new_film_size)

    def drag_camera(self, task: PythonTask) -> int:
        """Task to handle mouse camera drag actions. Holding down right click
        and dragging the screen will move the camera such that the mouse
        position is fixed in world space.

        Args:
            task (PythonTask): the current task being run

        Returns:
            int: return 1 to continue executing this task, and 0 to stop
                task execution.
        """
        if self.mouseWatcherNode.hasMouse():
            film_x, film_y = self.lens.getFilmSize()
            mouse_x = self.mouseWatcherNode.getMouseX()
            mouse_y = self.mouseWatcherNode.getMouseY()

            if self.mouse_state["mouse3"] == "down":
                mouse_dist = np.asarray(
                    [(self.last_mouse[0] - mouse_x) * film_x,
                     (self.last_mouse[1] - mouse_y) * film_y,
                     0])
                self.control_camera(mouse_dist / 2)
            self.last_mouse = np.asarray([mouse_x, mouse_y])
        return task.cont

    def handle_window_events(self, window: GraphicsWindow):
        """Handles window events, including resize and close.

        Args:
            window (GraphicsWindow): the current open window
        """
        if window.isClosed():
            sys.exit()
        x_size, y_size = window.size
        new_aspect = y_size / x_size
        film_x, _ = self.lens.getFilmSize()

        self.lens.setFilmSize(film_x, film_x * new_aspect)

    def update(self, task: PythonTask) -> int:
        """Updates the state of the world. Runs physics updates and cell
        update functions for code execution after each physics step.

        Args:
            task (PythonTask): the current task being run

        Returns:
            int: return 1 to continue executing this task, and 0 to stop
                task execution.
        """

        # Run cell divide
        res_cells = []
        for cell in self.cells:
            if cell.can_divide() and cell.should_divide():
                rand_vec = make_rand_vector(2) * .025
                new_cell = cell.divide(self.render, np.asarray(
                    [rand_vec[0], rand_vec[1], 0]))
                self.render.attachNewNode(new_cell.node)
                self.cell_physics.add_cell(new_cell)
                res_cells.append(new_cell)
            res_cells.append(cell)
        self.cells = res_cells

        d_t = globalClock.getDt()
        # Set maximum time step (for physics stability)
        if d_t > self.max_timestep:
            d_t = self.max_timestep

        self.cell_physics.update(
            d_t, num_neighbors=20, viscosity=10)

        # Execute the cells internal code
        for cell in self.cells:
            cell.execute_code()

        return task.cont


if __name__ == "__main__":

    # Main Program
    app = MultiCell()
    app.run()
