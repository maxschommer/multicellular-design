
import time

from direct.showbase.ShowBase import ShowBase
from direct.task import Task

from panda3d.core import (
    DirectionalLight
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

        self.camera.setPos(0, 0, 80 * MICROMETER)
        self.camera.setHpr(0, -90, 0)

        dlight = DirectionalLight('main_dlight')
        dlight.setColor((1, 1, 1, 1))
        dlnp = render.attachNewNode(dlight)
        dlnp.setHpr(0, -60, 0)
        render.setLight(dlnp)

        self.last_split = 0
        self.taskMgr.add(self.update, "update")
        self.taskMgr.add(self.control_camera, "control_camera")

        cell = Cell(render=render, resolution="high")
        cell.add_signal(
            render=render, signal_type=CellSignalType.SHORT_DISTANCE)
        self.cells = [cell]
        self.cell_physics = CellPhysics(
            self.cells, opposing_force=200, attraction_force=50, ignore_ratio=1.1)
        # myMaterial = Material()
        # myMaterial.setShininess(80)  # Make this material shiny
        # myMaterial.setAmbient((0, 0, 1, 1))  # Make this material blue
        # myMaterial.setDiffuse((1, 1, 0, 0))
        # cell_node = self.render.attachNewNode(self.cells.node)
        # cell_node.setMaterial(myMaterial)
        # cell_node.setDepthWrite(False)  # Disable
        # cell_node.setAntialias(AntialiasAttrib.MPolygon)

    def control_camera(self, task):
        if len(self.cells) < 2:
            return Task.cont
        horz_fov, vert_fov = self.camLens.fov * DEG
        min_bb, max_bb = self.cell_physics.bounding_box
        x_over = np.max([np.abs(min_bb[0]), np.abs(max_bb[0])])
        y_over = np.max([np.abs(min_bb[1]), np.abs(max_bb[1])])

        cam_height = self.camera.getPos()[2]
        cam_extent_x = cam_height * np.sin(horz_fov / 2)
        cam_extend_y = cam_height * np.sin(vert_fov / 2)

        if (cam_extent_x < x_over) or (cam_extend_y < y_over):
            cam_move_amount = np.max(
                [x_over - cam_extent_x, y_over - cam_extend_y])
            self.camera.setPos(0, 0, cam_height + .2 * cam_move_amount)
        return Task.cont

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
            dt, method="oct", num_neighbors=20, viscosity=10)
        # for i, cell in enumerate(self.cells):
        #     res_f = np.asarray([0, 0, 0], dtype=float)
        #     for j, o_cell in enumerate(self.cells):
        #         if i != j:
        #            np.sum(diff / norm(diff)**2)
        #             res_f += (cell.position - o_cell.position) / \
        #                 np.linalg.norm(cell.position - o_cell.position)**2
        #             # print(res_f)
        #     # print(res_f)
        #     # cell_forces.append(res_f)
        #     cell_acc = res_f / cell.mass
        #     cell_vel = cell.velocity + cell_acc * dt
        #     cell_pos = cell.position + cell_vel * dt
        #     cell.velocity = cell_vel
        #     cell.position = cell_pos

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
