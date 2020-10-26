import numbers
import copy
from random import gauss
import time
import cupy as cp
from typing import List, Optional
import scipy

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor

import panda3d
from panda3d.core import (
    GeomVertexFormat, GeomVertexData, Geom, GeomTriangles, GeomVertexWriter,
    GeomNode, DirectionalLight, VBase4, Material, Fog, AntialiasAttrib
)
import scipy.spatial.distance
import timeit
import functools
from numba import jit
import numpy as np
import quaternion
import meshzoo

# Yes, it should be "e-6", but it's a game and I don't want to
# fuss around with floating point errors that result from that.
MICROMETER = 1
DEG = np.pi / 180


def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return np.asarray([x / mag for x in vec])


def make_node_from_mesh(points: np.ndarray, faces: np.ndarray, normals: np.ndarray):
    vertex_normal_format = GeomVertexFormat.get_v3n3()

    v_data = GeomVertexData('sphere', vertex_normal_format, Geom.UHStatic)
    num_rows = np.max([points.shape[0], faces.shape[0]])
    v_data.setNumRows(int(num_rows))

    vertex_data = GeomVertexWriter(v_data, 'vertex')
    normal_data = GeomVertexWriter(v_data, 'normal')
    for point, normal in zip(points, normals):
        vertex_data.addData3(point[0], point[1], point[2])
        normal_data.addData3(normal[0], normal[1], normal[2])

    geom = Geom(v_data)
    for face in faces:
        tri = GeomTriangles(Geom.UHStatic)
        p_1 = points[face[0], :]
        p_2 = points[face[1], :]
        p_3 = points[face[2], :]
        norm = normals[face[0], :]
        if np.dot(np.cross(p_2 - p_1, p_3 - p_2), norm) < 0:
            tri.add_vertices(face[2], face[1], face[0])
        else:
            tri.add_vertices(face[0], face[1], face[2])
        geom.addPrimitive(tri)

    node = GeomNode('gnode')
    node.addGeom(geom)
    return node


def make_sphere_node(resolution: int, diameter: float):
    points, faces = meshzoo.uv_sphere(
        num_points_per_circle=resolution,
        num_circles=resolution, radius=diameter / 2)
    normals = points / np.expand_dims(np.linalg.norm(points, axis=1), 1)
    return make_node_from_mesh(np.asarray(points),
                               np.asarray(faces),
                               np.asarray(normals))


LOW_RES_SPHERE = make_sphere_node(resolution=10, diameter=1)


class Cell():
    cell_instance = 0

    def __init__(self, render: panda3d.core.NodePath,
                 diameter: float = 10 * MICROMETER,
                 position: np.ndarray = np.asarray([0, 0, 0], dtype=float),
                 velocity: np.ndarray = np.asarray([0, 0, 0], dtype=float),
                 orientation: np.ndarray = np.asarray(
                     [1, 0, 0, 0], dtype=float),
                 angular_velocity: np.ndarray = np.asarray(
                     [0, 0, 0], dtype=float),
                 mass: float = 1.0
                 ):
        # Define model geometry
        self.node = copy.deepcopy(LOW_RES_SPHERE)
        self.node_path = render.attachNewNode(self.node)

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
                    orientation=self.orientation)

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
# [np.newaxis, :, :]


def get_forces_oct_tree(positions, force_const, num_neighbors=10, radii=None, ignore_ratio=1.5):
    kd_tree = scipy.spatial.cKDTree(positions)

    num_neighbors = num_neighbors if num_neighbors <= positions.shape[0] else positions.shape[0]
    n_dist, n_idx = kd_tree.query(positions, k=num_neighbors)
    # print(radii[n_idx] + radii)
    dist_thresh = (radii[n_idx] + np.expand_dims(radii, 1)) < n_dist
    ignore_thresh = (radii[n_idx] + np.expand_dims(radii, 1)
                     ) * ignore_ratio < n_dist
    force_dirs = np.ones_like(dist_thresh, dtype=int)
    force_dirs[dist_thresh] = -1
    force_dirs[ignore_thresh] = 0
    diff = np.expand_dims(positions, 1) - \
        positions[n_idx]

    all_forces = diff / \
        np.expand_dims(n_dist, -1)**2 * np.expand_dims(force_dirs, -1)

    all_forces = np.where(np.isnan(all_forces), 0, all_forces)

    forces = np.sum(all_forces, axis=1) * force_const

    return forces


def get_forces_gpu(positions, force_const):
    positions = cp.asarray(positions)
    diff = cp.expand_dims(positions, 0) - \
        cp.expand_dims(positions, 1)

    dist = cp.sqrt(np.sum(diff**2, axis=-1))
    all_forces = diff / np.expand_dims(dist, 2)**2
    all_forces = cp.where(np.isnan(all_forces), 0, all_forces)
    forces = cp.sum(all_forces, axis=0) * force_const
    return cp.asnumpy(forces)


@jit(nopython=True)
def get_forces(positions: np.ndarray, force_const: float = 1):
    diff = np.expand_dims(positions, 0) - \
        np.expand_dims(positions, 1)

    dist = np.sqrt(np.sum(diff**2, axis=-1))
    all_forces = diff / np.expand_dims(dist, 2)**2

    all_forces = np.where(np.isnan(all_forces), 0, all_forces)

    forces = np.sum(all_forces, axis=0) * force_const
    return forces


class CellPhysics():
    def __init__(self, cells: Optional[List[Cell]] = None,
                 force_const: float = 1.0,
                 ignore_ratio=1.5):
        self._cells = []
        self._positions = None
        self._velocities = None
        self._radii = None
        self._masses = None
        self.force_const = force_const
        self.ignore_ratio = ignore_ratio

        if cells is not None:
            self.cells = cells

    @property
    def cells(self):
        return self._cells

    @cells.setter
    def cells(self, value: List[Cell]):
        assert isinstance(
            value[0], Cell), "cells must be a valid iterable of Cell objects."

        for cell in value:
            self.add_cell(cell)

    def _add_velocity(self, velocity: np.ndarray):
        assert isinstance(
            velocity, np.ndarray), "velocity must be a valid numpy ndarray."
        assert velocity.shape[0] == 3, "velocity must be a vector of length 3."

        if self._velocities is None:
            self._velocities = np.asarray([velocity], dtype=float)
        else:
            self._velocities = np.concatenate(
                [self._velocities, [velocity]], axis=0)

    def _add_position(self, position: np.ndarray):
        assert isinstance(
            position, np.ndarray), "position must be a valid numpy ndarray."
        assert position.shape[0] == 3, "position must be a vector of length 3."
        if self._positions is None:
            self._positions = np.asarray([position])
        else:
            self._positions = np.concatenate(
                [self._positions, [position]], axis=0)

    def _add_mass(self, mass: float):
        assert isinstance(mass, numbers.Number), "mass must be a valid number"
        if self._masses is None:
            self._masses = np.asarray([mass])
        else:
            self._masses = np.concatenate(
                [self._masses, [mass]])

    def _add_radii(self, radius: float):
        assert isinstance(
            radius, numbers.Number), "radius must be a valid number"
        if self._radii is None:
            self._radii = np.asarray([radius])
        else:
            self._radii = np.concatenate(
                [self._radii, [radius]])

    def add_cell(self, cell: Cell):
        assert isinstance(cell, Cell), "cell must be a valid Cell object."
        self._cells.append(cell)

        self._add_position(cell.position)
        self._add_velocity(cell.velocity)
        self._add_mass(cell.mass)
        self._add_radii(cell.diameter / 2)

    def update(self, dt: float, method: str = "cpu", num_neighbors=10, viscosity=0.5):
        if len(self.cells) == 0:
            return

        if method == "gpu":
            forces = get_forces_gpu(self._positions, self.force_const)
        if method == "cpu":
            forces = get_forces(self._positions, self.force_const)
        if method == "oct":
            forces = get_forces_oct_tree(
                self._positions, self.force_const,
                num_neighbors=num_neighbors,
                radii=self._radii,
                ignore_ratio=self.ignore_ratio)

        for i, cell in enumerate(self.cells):
            cell_acc = forces[i, :] / cell.mass
            cell.velocity = (
                cell.velocity * (1 - viscosity * dt)) + cell_acc * dt
            cell.position = cell.position + cell.velocity * dt

            self._positions[i, :] = cell.position
            self._velocities[i, :] = cell.velocity


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

        self.cells = [Cell(render), Cell(
            render, position=np.asarray([12, 0, 0]))]
        self.cell_physics = CellPhysics(
            self.cells, force_const=200, ignore_ratio=1.1)
        # myMaterial = Material()
        # myMaterial.setShininess(80)  # Make this material shiny
        # myMaterial.setAmbient((0, 0, 1, 1))  # Make this material blue
        # myMaterial.setDiffuse((1, 1, 0, 0))
        # cell_node = self.render.attachNewNode(self.cells.node)
        # cell_node.setMaterial(myMaterial)
        # cell_node.setDepthWrite(False)  # Disable
        # cell_node.setAntialias(AntialiasAttrib.MPolygon)

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
        print(len(self.cells), sim_t)
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
