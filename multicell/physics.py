from typing import List, Optional
import numbers
import numpy as np
import cupy as cp
import scipy
from scipy.spatial import cKDTree

from numba import jit
from cell import Cell


def get_forces_oct_tree(positions, force_const, num_neighbors=10, radii=None, ignore_ratio=1.5):
    kd_tree = cKDTree(positions)

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

    @property
    def bounding_box(self):
        return np.min(self._positions, axis=1), np.max(self._positions, axis=1)

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