from random import gauss
import numpy as np

from panda3d.core import (
    GeomVertexData, GeomVertexWriter, GeomVertexFormat, Geom,
    GeomTriangles, GeomNode
)
import meshzoo


def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return np.asarray([x / mag for x in vec])


def make_node_from_mesh(points: np.ndarray,
                        faces: np.ndarray,
                        normals: np.ndarray,
                        rgba: np.ndarray = np.asarray([1, 1, 1, 1])):
    vertex_normal_format = GeomVertexFormat.get_v3n3c4()

    v_data = GeomVertexData('sphere', vertex_normal_format, Geom.UHStatic)
    num_rows = np.max([points.shape[0], faces.shape[0]])
    v_data.setNumRows(int(num_rows))

    vertex_data = GeomVertexWriter(v_data, 'vertex')
    normal_data = GeomVertexWriter(v_data, 'normal')
    color_data = GeomVertexWriter(v_data, 'color')

    for point, normal in zip(points, normals):
        vertex_data.addData3(point[0], point[1], point[2])
        normal_data.addData3(normal[0], normal[1], normal[2])
        color_data.addData4(rgba[0], rgba[1], rgba[2], rgba[3])
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


def make_sphere_node(resolution: int,
                     diameter: float,
                     rgba: np.ndarray = np.asarray([1, 1, 1, 1])):
    points, faces = meshzoo.uv_sphere(
        num_points_per_circle=resolution,
        num_circles=resolution, radius=diameter / 2)
    normals = points / np.expand_dims(np.linalg.norm(points, axis=1), 1)
    return make_node_from_mesh(np.asarray(points),
                               np.asarray(faces),
                               np.asarray(normals),
                               np.asarray(rgba))
