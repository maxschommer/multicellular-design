import sys
import time

from direct.showbase.ShowBase import ShowBase
from direct.task import Task


from panda3d.core import (
    DirectionalLight, OrthographicLens
)
import numpy as np

from multicell.utils import make_rand_vector
from multicell.config import MICROMETER, DEG
from multicell.cell import Cell, CellSignalType
from multicell.physics import CellPhysics


class MultiCell(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)

        base.disableMouse()
        base.enableParticles()

        self.lens = OrthographicLens()
        # Or whatever is appropriate for your scene
        self.lens.setFilmSize(60, 45)
        base.cam.node().setLens(self.lens)

        self.camera.setPos(0, 0, 80 * MICROMETER)
        self.camera.setHpr(0, -90, 0)

        dlight = DirectionalLight('main_dlight')
        dlight.setColor((1, 1, 1, 1))
        dlnp = render.attachNewNode(dlight)
        dlnp.setHpr(0, -60, 0)
        render.setLight(dlnp)

        self.last_mouse = [0, 0]
        self.mouse_state = {
            "mouse3": "up"
        }

        self.last_split = 0
        self.taskMgr.add(self.update, "update")
        self.taskMgr.add(self.drag_camera, "drag_camera")

        # Listen set window event listener
        self.accept("window-event", self.handle_window_events)
        self.configure_cam_control()

        cell = Cell(render=render, resolution="high")
        cell.add_signal(
            render=render, signal_type=CellSignalType.SHORT_DISTANCE)
        self.cells = [cell]
        self.cell_physics = CellPhysics(
            self.cells, opposing_force=200, attraction_force=50, ignore_ratio=1.1)

    def configure_cam_control(self):
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

    def mouse_button(self, key):
        if key == "mouse3":
            self.mouse_state["mouse3"] = "down"
        if key == "mouse3-up":
            self.mouse_state["mouse3"] = "up"

    def control_camera(self, move_vec, move_speed: float = 1, zoom_speed: float = 0.1):
        cur_pos = np.asarray(self.camera.getPos())
        cur_film_size = self.lens.getFilmSize()
        new_pos = cur_pos + move_vec * move_speed
        new_film_size = cur_film_size + \
            cur_film_size * move_vec[2] * zoom_speed
        self.camera.setPos(new_pos[0], new_pos[1], cur_pos[2])
        self.lens.setFilmSize(*new_film_size)

    def drag_camera(self, task):
        if base.mouseWatcherNode.hasMouse():
            film_x, film_y = self.lens.getFilmSize()
            x = base.mouseWatcherNode.getMouseX()
            y = base.mouseWatcherNode.getMouseY()

            if self.mouse_state["mouse3"] == "down":
                mouse_dist = np.asarray(
                    [(self.last_mouse[0] - x) * film_x, (self.last_mouse[1] - y) * film_y, 0])
                self.control_camera(mouse_dist / 2)
            self.last_mouse = np.asarray([x, y])
        return Task.cont

    def handle_window_events(self, window):
        if window.isClosed():
            sys.exit()
        x_size, y_size = window.size
        new_aspect = y_size / x_size
        film_x, _ = self.lens.getFilmSize()

        self.lens.setFilmSize(film_x, film_x * new_aspect)

    def update(self, task):
        if (task.time - self.last_split > 5) and len(self.cells) < 33:
            self.last_split = task.time
            res_cells = []
            for cell in self.cells:
                rand_vec = make_rand_vector(2) * .025
                new_cell = cell.divide(self.render, np.asarray(
                    [rand_vec[0], rand_vec[1], 0]))
                self.render.attachNewNode(new_cell.node)
                self.cell_physics.add_cell(new_cell)
                res_cells.append(cell)
                res_cells.append(new_cell)
            self.cells = res_cells

        t0 = time.time()
        dt = globalClock.getDt()
        self.cell_physics.update(
            dt, num_neighbors=20, viscosity=10)

        # Execute the cells internal code
        for i, cell in enumerate(self.cells):
            cell.execute_code()

        sim_t = time.time() - t0
        # print(len(self.cells), sim_t)
        return Task.cont


if __name__ == "__main__":
    # test_points = np.random.rand(2500, 3)

    # print(test_oct.node, test_oct.leafnode, test_oct.innernode)
    # get_forces_oct_tree(test_points, 0.1)

    # Run Tests
    # get_forces(test_points)
    # run_oct = functools.partial(get_forces_oct_tree, test_points, 0.1)
    # run_gpu = functools.partial(get_forces_gpu, test_points, 0.1)
    # run_cpu = functools.partial(get_forces, test_points, 0.1)
    # res = timeit.timeit(run_cpu, number=100)
    # print("CPU: ", res)
    # res = timeit.timeit(run_gpu, number=100)
    # print("GPU: ", res)
    # res = timeit.timeit(run_oct, number=100)
    # print("Oct Tree: ", res)
    # Main Program
    app = MultiCell()
    app.run()
