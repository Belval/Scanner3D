"""
Custom naive implementation of BPA *not in working condition*

Attribution et source:
 - https://cs184team.github.io/cs184-final/writeup.html
 - https://vgc.poly.edu/~csilva/papers/tvcg99.pdf (page 5)
 - https://github.com/intel-isl/Open3D/blob/master/cpp/open3d/geometry/SurfaceReconstructionBallPivoting.cpp
"""

import numpy as np

# For performance and simplicity, we do not rool our own KDTree here
from sklearn.neighbors import KDTree


def ball_center(p1, p2, p3, n1, n2, n3, radius):
    c = np.linalg.norm((p2 - p1))
    b = np.linalg.norm((p1 - p3))
    a = np.linalg.norm((p3 - p2))

    alpha = a * (b + c - a)
    beta = b * (a + c - b)
    gamma = c * (a + b - c)
    abg = alpha + beta + gamma

    if abg < 1e-16:
        return None

    alpha = alpha / abg
    beta = beta / abg
    gamma = gamma / abg

    circ_center = alpha * p1 + beta * p2 + gamma * p3
    circ_radius2 = a * b * c

    a = np.sqrt(a)
    b = np.sqrt(b)
    c = np.sqrt(c)
    circ_radius2 = circ_radius2 / (
        (a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)
    )

    height = radius * radius - circ_radius2
    if height >= 0.0:
        tr_norm = np.cross(p2 - p1, p3 - p1)
        tr_norm /= np.linalg.norm(tr_norm)
        pt_norm = n1 + n2 + n3
        pt_norm /= np.linalg.norm(pt_norm)
        if np.dot(tr_norm, pt_norm) < 0:
            tr_norm *= -1

        height = np.sqrt(height)
        center = circ_center + height * tr_norm
        return center
    return None


def compute_face_normal(v0, v1, v2):
    normal = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal /= norm
    return normal


def is_compatible(p0, n0, p1, n1, p2, n2):
    normal = compute_face_normal(p0, p1, p2)

    if np.dot(normal, n0) < -1e-16:
        normal *= -1

    ret = (
        np.dot(normal, n0) > -1e-16
        and np.dot(normal, n1) > -1e-16
        and np.dot(normal, n2) > -1e-16
    )

    return ret


def get_linking_edge(v0_edges, v1_edges):
    for edge0 in v0_edges:
        for edge1 in v1_edges:
            if edge0[1] == edge1[1] and edge0[2] == edge1[2]:
                return edge0
    return None


def try_triangle_seed(
    p0, p1, p2, v0, v1, v2, n0, n1, n2, points, edges, nb_indices, radius, center
):
    if is_compatible(p0, n0, p1, n1, p2, n2):
        return False

    e0 = get_linking_edge(edges[v0[0]], edges[v2[0]])
    e1 = get_linking_edge(edges[v1[0]], edges[v2[0]])
    if e0 is not None and e0[0] == 0:
        return False
    if e1 is not None and e1[0] == 0:
        return False

    if not ball_center(p0, p1, p2, n0, n1, n2, radius):
        return False

    for i in range(nb_indices):
        v = vertices[i]
        if v[0] == v0[0] or v[0] == v1[0] or v[0] == v2[0]:
            continue
        if np.linalg.norm(center - points[v[0]]) < radius - 1e-16:
            return False

    return True


def try_seed(vertex, vertices, edges, radius, kdtree):
    indices = kdtree.query_radius(vertex, r=radius)

    if len(indices) < 3:
        return False

    for i in range(len(indices)):
        nb0_idx, nb0_type = vertices[indices[i]]
        if nb0_type != 0:
            continue
        if nb0_idx == vertex[0]:
            continue

        candidate_vidx2 = -1

        for j in range(len(indices)):
            nb1_idx, nb1_type = vertices[indices[j]]
            if nb1_type != 0:
                continue
            if nb1_idx_ == vertex[0]:
                continue
            if TryTriangleSeed(
                vertex,
                vertices[indices[i]],
                vertices[indices[j]],
                indices,
                radius,
                center,
            ):
                candidate_vidx2 = nb1_idx
                break

        if candidate_vidx2 >= 0:
            nb1_idx, nb1_type = vertices[candidate_vidx2]

            e0 = get_linking_edge(edges[v[0]], edges[nb1_idx])
            if e0 is not None and e0[0] != 1:
                continue
            e1 = get_linking_edge(edges[nb0_idx], edges[nb1_idx])
            if e1 is not None and e1[0] != 1:
                continue
            e2 = get_linking_edge(edges[v[0]], edges[nb0_idx])
            if e2 is not None and e2[0] != 1:
                continue

            triangle = create_triangle(v, nb0, nb1)

            e0 = get_linking_edge(v, nb1)
            e1 = get_linking_edge(nb0, nb1)
            e2 = get_linking_edge(v, nb0)

            if e0[0] == 1:
                edges.append(e0)
            if e1[0] == 1:
                edges.append(e1)
            if e2[0] == 1:
                edges.append(e2)

            if e[0] == 1 or e1[0] == 1 or e2[0] == 1:
                return True
    return False


def create_triangle(v, v1, v2):
    # TODO
    return None


def find_seed_triangle(points, vertices, radius):
    for i in range(vertices.shape[0]):
        if vertices[i, 1] == 0:
            if try_seed(vertices[i], vertices, radius):
                expand_triangulation(radius)


def expand_triangulation(points, vertices, radius):
    # TODO
    return None


def bpa(points, normals, radius):
    # Store for vertices features
    # 1: Point idx
    # 2: Vertex type (0 = orphan, 1 = used)
    vertices = np.zeros((len(points), 2))
    for i in range(len(points)):
        vertices[i, :] = [i, 0]

    if radius <= 0:
        raise Exception("Radius must be greater than 0")

    kdtree = KDTree(points, leaf_size=10)

    find_seed_triangle(points, vertices, radius)
    expand_triangulation(radius)

    return None
