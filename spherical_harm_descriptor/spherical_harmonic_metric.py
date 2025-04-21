import ast
import dataclasses
from typing import List, Iterable, Any, Tuple, Dict, Tuple

import icosphere
import numpy as np
from ipywidgets import widgets
from numpy import real
from scipy.special import sph_harm

from CGAL.CGAL_AABB_tree import AABB_tree_Polyhedron_3_Facet_handle
from CGAL.CGAL_Kernel import Ray_3, Point_3
from CGAL.CGAL_Polygon_mesh_processing import Polylines
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3, Polyhedron_3_Modifier_3, Integer_triple
from CGAL.CGAL_Surface_mesh_skeletonization import surface_mesh_skeletonization


def v_f_to_mesh(v: np.ndarray, f: np.ndarray) -> Polyhedron_3:
    p = Polyhedron_3()
    modifier = Polyhedron_3_Modifier_3()
    point_list = [Point_3(*vertex.tolist()) for vertex in v]
    triple_list = [Integer_triple(*facet.tolist()) for facet in f]
    modifier.set_modifier_data(point_list, triple_list)
    p.delegate(modifier.get_modifier())
    return p


def _mesh_to_v_f(mesh: Polyhedron_3) -> Tuple[np.ndarray, np.ndarray]:
    vertices = np.ndarray((mesh.size_of_vertices(), 3))
    for i, vertex in enumerate(mesh.vertices()):
        vertex.set_id(i)
        vertices[i, :] = point_2_list(vertex.point())

    facets = np.ndarray((mesh.size_of_facets(), 3)).astype("uint")
    for i, facet in enumerate(mesh.facets()):
        circulator = facet.facet_begin()
        j = 0
        begin = facet.facet_begin()
        while circulator.hasNext():
            halfedge = circulator.next()
            v = halfedge.vertex()
            facets[i, j] = (v.id())
            j += 1
            if circulator == begin or j == 3:
                break
    return vertices, facets


def polar2cart(az: np.ndarray, elev: np.ndarray, radius: np.ndarray) -> np.ndarray:
    return np.stack([radius * np.sin(elev) * np.cos(az),
                     radius * np.sin(elev) * np.sin(az),
                     radius * np.cos(elev)], axis=-1)


def cart2polar(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    x, y, z = np.array(x), np.array(y), np.array(z)
    XsqPlusYsq = np.power(x, 2) + np.power(y, 2)
    r = np.sqrt(XsqPlusYsq + np.power(z, 2))    # r
    elev = np.arctan2(np.sqrt(XsqPlusYsq), z)   # theta
    az = np.arctan2(y, x)                       # phi
    return [r, elev, az]


def point_2_list(point: Point_3) -> List[float]:
    return [point.x(), point.y(), point.z()]


class SpineMetric:
    pass

@dataclasses.dataclass
class SGSample:
    az: float
    elev: float
    sg_coefficients: List[float]


class SphericalGarmonicsSpineMetric(SpineMetric):
    DEFAULT_L_SIZE: int = 10
    _m_l_map: Dict[int, Tuple[int, int]]
    _mc_samples: List[SGSample] = None

    def __init__(self, spine_mesh: Polyhedron_3 = None, l_range: Iterable = None, sqrt_sample_size: int = 100):
        if l_range is None:
            l_range = range(self.DEFAULT_L_SIZE)
        self._m_l_map = {}
        self._basis = []
        i = 0
        for _l in l_range:
            for m in range(-_l, _l + 1):
                self._m_l_map[i] = (m, _l)
                self._basis.append(self._get_basis(m, _l))
                i += 1

        if SphericalGarmonicsSpineMetric._mc_samples is None:
            SphericalGarmonicsSpineMetric._mc_samples = self._generate_sample(sqrt_n=sqrt_sample_size)  # monte carlo sample on sphere
        super().__init__(spine_mesh)

    @property
    def m_l_map(self):
        return dict(self._m_l_map)

    def surface_norm(self, point: List[float], dx: float = 0.01) -> np.ndarray:
        f_p = self.surface_value(point)
        d_theta, d_phi = point[0] + dx, point[1] + dx
        point_sin, point_cos = np.sin(point), np.cos(point)

        ab = np.array([self.surface_value([d_theta, point[1]]), d_theta, point[1]]) - np.array([f_p, *point])
        ac = np.array([self.surface_value([point[0], d_phi]), point[0], d_phi]) - np.array([f_p, *point])
        norm = np.cross(ab, ac)

        norm = np.matmul(np.array([
            [point_sin[0]*point_cos[1], point_cos[0]*point_cos[1], -point_sin[1]],
            [point_sin[0]*point_sin[1], point_cos[0]*point_sin[1], point_cos[1]],
            [point_cos[0], -point_sin[0], 0]
        ]), norm)
        return norm / np.linalg.norm(norm)

    def _get_mesh(self, steps_num: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = np.linspace(0, np.pi, steps_num[0])
        phi = np.linspace(0, 2 * np.pi, steps_num[1])
        T, P = np.meshgrid(theta, phi)
        radiuses = np.array([[real(self.surface_value([t, p])) for t, p in zip(t_values, p_values)] for t_values, p_values in zip(T, P)])
        xyz = polar2cart(T, P, radiuses)
        return xyz[..., 0], xyz[..., 1], xyz[..., 2]

    def _get_mesh_2(self):
        pass

    def _get_basis(self, order: int, degree: int) -> callable:
        if order > 0:
            return lambda az, elev: sph_harm(order, degree, az, elev).real * np.sqrt(2)
        elif order < 0:
            return lambda az, elev: sph_harm(-order, degree, az, elev).imag * np.sqrt(2)
        else:
            return lambda az, elev: sph_harm(order, degree, az, elev).real

    def _generate_sample(self, sqrt_n: int) -> List[SGSample]:
        oneoverN = 1 / sqrt_n
        result = []
        r = np.random.random(sqrt_n*sqrt_n*2)
        for a in range(sqrt_n):
            for b in range(sqrt_n):
                x, y = a + r[b * 2 + a * 2 * sqrt_n], b + r[b * 2 + a * 2 * sqrt_n + 1]
                x, y = x * oneoverN, y * oneoverN
                elev = 2 * np.arccos(np.sqrt(1-x))
                az = 2 * np.pi * y
                result.append(SGSample(az, elev, [basis_f(az, elev) for basis_f in self._basis]))
        return result

    def _calculate(self, spine_mesh: Polyhedron_3) -> np.ndarray:
        subdivided_mesh = subdivide_mesh(spine_mesh, relative_max_facet_area=0.0003)
        v = subdivided_mesh.vertices()
        points = [point_2_list(vertex.point()) for vertex in v]

        def perpendicular(line, point):
            P1 = np.array(point_2_list(line[0]))
            P2 = np.array(point_2_list(line[1]))
            t = np.dot(point - P1, P2 - P1) / (np.linalg.norm(P2 - P1) ** 2)
            return P1 + t * (P2 - P1)

        skeleton_polylines = Polylines()
        correspondence_polylines = Polylines()
        surface_mesh_skeletonization(subdivided_mesh, skeleton_polylines, correspondence_polylines)
        center = np.array((np.min(points, axis=0) + np.max(points, axis=0)) / 2)
        inner_center = center
        min_dist = 1 << 20
        for line in skeleton_polylines:
            for i in range(len(line) - 1):
                p = perpendicular((line[i], line[i + 1]), center)
                d = np.linalg.norm(p - center)
                if d < min_dist:
                    inner_center = p
                    min_dist = d

        center = inner_center
        radius_elev_az = np.array([cart2polar(*(p - center)) for p in points])
        max_radius = max(radius_elev_az[:, 0]) * 2
        mean_radius = np.mean(radius_elev_az[:, 0])

        def intersecton_point(facet, ray):
            a_matrix = []
            circulator = facet.facet_begin()
            begin = facet.facet_begin()
            while circulator.hasNext():
                halfedge = circulator.next()
                pnt = halfedge.vertex().point()
                a_matrix.append([pnt.x(), pnt.y(), pnt.z()])
                # check for end of loop
                if circulator == begin:
                    break
            b_matrix = [-1] * len(a_matrix)
            plane_coefs = np.linalg.lstsq(a_matrix, b_matrix)[0]
            start, end = ray.point(0), ray.point(1)
            line_equations = [[end.y() - start.y(), start.x() - end.x(), 0],
                              [0, end.z() - start.z(), start.y() - end.y()],
                              [*plane_coefs]]
            b2_matrix = [start.x() * (end.y() - start.y()) + start.y() * (start.x() - end.x()),
                         start.y() * (end.z() - start.z()) + start.z() * (start.y() - end.y()),
                         -1]
            return np.linalg.lstsq(line_equations, b2_matrix)[0]

        def estimate_mesh(az, elev):
            intersections = []
            origin = np.array([0, 0, 0])
            tree = AABB_tree_Polyhedron_3_Facet_handle(subdivided_mesh.facets())
            ray = Ray_3(Point_3(*(origin + center)), Point_3(*(polar2cart(az, elev, max_radius) + center)))
            tree.all_intersections(ray, intersections)
            if len(intersections) != 1:
                print(f"WARNING intersections len = {len(intersections)}")
            # else:
            #     print("OK")
            if len(intersections) < 1:
                return mean_radius
            point = intersecton_point(intersections[0].second, ray)
            r, elev_est, az_est = cart2polar(*(point - center))
            return r

        res = np.zeros(len(self._basis))
        mesh_sample = [estimate_mesh(sample.az, sample.elev) for sample in SphericalGarmonicsSpineMetric._mc_samples]
        factor = 4 * np.pi / len(mesh_sample)
        for n in range(len(self._basis)):
            res[n] = sum(estim * sample.sg_coefficients[n] for estim, sample in
                         zip(mesh_sample, SphericalGarmonicsSpineMetric._mc_samples)) * factor

        return res

    def show(self, **kwargs) -> widgets.Widget:
        mesh, v, f = self.get_basis_composition()
        viewer = mp.Viewer({"width": 200, "height": 200})
        viewer.add_mesh(v, f)
        viewer._renderer.layout = widgets.Layout(border="solid 1px")
        return viewer._renderer

    def get_basis_composition(self) -> Polyhedron_3:

        def composition_callback(az, elev):
            return sum(a_i * self._basis[i](az, elev) for i, a_i in enumerate(self.value))

        v, c = icosphere.icosphere(10)
        rad, elev, az = cart2polar(v[:, 0], v[:, 1], v[:, 2])

        rad = np.array([composition_callback(a_i, e_i) for a_i, e_i in zip(az, elev)])
        v = polar2cart(az, elev, rad)

        mesh = v_f_to_mesh(v, c)

        return mesh, v, c

    def parse_value(self, value_str: str):
        value_str = value_str.replace('\n', '')
        value_str = value_str.replace('  ', ' ')
        try:
            value = ast.literal_eval(value_str)
        except Exception:
            value = np.fromstring(value_str[1:-1], dtype="float", sep=" ")
        self.value = value

    def value_as_lists(self) -> List[Any]:
        return [*self.value]