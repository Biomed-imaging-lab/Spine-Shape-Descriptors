import math
from math import factorial
from typing import List, Iterable, Any, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import meshplot as mp
from ipywidgets import widgets
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
#from utils import make_circle



def point_in_circle(circle, point):
    return circle is not None and math.hypot(*np.subtract(point, circle[:-1])) - circle[-1] <= 1e-14


def get_enclosing_circle(points):
    if len(points) < 1:
        return None
    c = (*points[0], 0.0)
    for i, point_1 in enumerate(points[:]):
        if not point_in_circle(c, point_1):
            c = (*point_1, 0.0)
            for j, point_2 in enumerate(points[:i+1]):
                if not point_in_circle(c, point_2):
                    c = make_circle(points[:j+1], point_1, point_2)
    return c


def polar2cart(az: np.ndarray, elev: np.ndarray, radius: np.ndarray) -> np.ndarray:
    return np.stack([radius * np.sin(elev) * np.cos(az),
                     radius * np.sin(elev) * np.sin(az),
                     radius * np.cos(elev)], axis=-1)


def point_2_list(point: Point_3) -> List[float]:
    return [point.x(), point.y(), point.z()]


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


def cart2polar(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    x, y, z = np.array(x), np.array(y), np.array(z)
    XsqPlusYsq = np.power(x, 2) + np.power(y, 2)
    r = np.sqrt(XsqPlusYsq + np.power(z, 2))    # r
    elev = np.arctan2(np.sqrt(XsqPlusYsq), z)   # theta
    az = np.arctan2(y, x)                       # phi
    return [r, elev, az]


class SpineMetric:
    def __init__(self, spine_mesh: Polyhedron_3 = None):
        pass

class LightFieldZernikeMomentsSpineMetric(SpineMetric):
    def __init__(self, spine_mesh: Polyhedron_3 = None, radius: int = 1, view_points: int = 5, order: int = 15):
        #v, _ = icosphere.icosphere(view_points // 5)
        #_, elev, az = cart2polar(v[:, 0], v[:, 1], v[:, 2])
        #self._view_points = np.array([[az[i], elev[i], radius*2]
        #                              for i in range(0, len(elev))])
        sphere_points_data = {
            3: np.array([ 
                                      [0, 0, radius*2], 
                                      [0, np.pi/2, radius*2], 
                                      [np.pi/2, np.pi/2, radius*2],
                                      ]),
            5: np.array([ 
                                      [0, 0, radius*2], 
                                      [0, np.pi/2, radius*2], 
                                      [np.pi/2, np.pi/2, radius*2],
                                      [np.pi/3, np.pi/3, radius*2],
                                      [np.pi/3, 2*np.pi/3, radius*2],
                                      ]),
            7: np.array([ 
                                      [0, 0, radius*2], 
                                      [0, np.pi/2, radius*2], 
                                      [np.pi/2, np.pi/2, radius*2],
                                      [np.pi/3, np.pi/3, radius*2],
                                      [np.pi/3, 2*np.pi/3, radius*2],
                                      [2*np.pi/3, np.pi/3, radius*2],
                                      [2*np.pi/3, 2*np.pi/3, radius*2],
                                      ]),
            10: np.array([
                                      [0, 0, radius*2], 
                                      [0, np.pi/2, radius*2], 
                                      [0, -np.pi/2, radius*2], 
                                      [0, np.pi, radius*2],
                                      [-np.pi/2, np.pi/2, radius*2],
                                      [np.pi/2, np.pi/2, radius*2],
                                      [np.pi/3, np.pi/3, radius*2],
                                      [np.pi/3, 2*np.pi/3, radius*2],
                                      [2*np.pi/3, np.pi/3, radius*2],
                                      [2*np.pi/3, 2*np.pi/3, radius*2],
                                      ])
        }
        if view_points not in sphere_points_data.keys():
            raise ValueError(f"Invalid argument value. view_points could be equal to one of the value - {sphere_points_data.keys()}")
        self._view_points = sphere_points_data[view_points]
        self._zernike_radius = radius
        self._zernike_order = order
        super().__init__(spine_mesh)

    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        self._value = []
        for projection in self.get_projections(spine_mesh):
            self._value.append(self._calculate_moment(projection, degree=self._zernike_order).tolist())
        return self._value
    
    def _get_rotation(self, normal):
        # prepare rotation angle for coordinate transformation
        z_0 = [0, 0, 1]
        x, y = normal[0], normal[1]
        normal[1] = 0
        norm = np.linalg.norm(normal)
        sin_ay, cos_ay = np.linalg.norm(np.cross(normal, z_0)) / norm, np.dot(normal, z_0) / norm
        if normal[0] < 0:
            sin_ay = -sin_ay

        normal[1] = y
        normal[0] = 0
        norm = np.linalg.norm(normal)
        sin_ax, cos_ax = np.linalg.norm(np.cross(normal, z_0)) / norm, np.dot(normal, z_0) / norm
        if normal[1] > 0:
            sin_ax = -sin_ax
        normal[0] = x

        # create transformation matrix
        rotation_matrix = np.array([[1, 0, 0], [0, cos_ax, -sin_ax], [0, sin_ax, cos_ax]])
        rotation_matrix = np.matmul(np.array([[cos_ay, 0, sin_ay], [0, 1, 0], [-sin_ay, 0, cos_ay]]), rotation_matrix)
        return rotation_matrix

    def _get_image(self, poly_contour: Polygon):
        contour = np.array(poly_contour.exterior.coords)

        cx, cy, radius = get_enclosing_circle(contour)
        scale = 100 / radius

        contour = np.matmul([[scale, 0], [0, scale]], contour.T).T
        contour[:, 0] = contour[:, 0] - cx * scale + 100
        contour[:, 1] = contour[:, 1] - cy * scale + 100

        contour = contour.astype(int)
        mask = np.zeros((200, 200))
        cv2.fillPoly(mask, pts=[contour], color=(255,255,255))
        return mask
    
    def get_projections(self, spine_mesh: Polyhedron_3) -> Iterable[np.ndarray]:
        result = []
        mp.plot(*_mesh_to_v_f(spine_mesh))
        fig, ax = plt.subplots(ncols=2, nrows=(len(self._view_points) + 1) // 2, figsize=(12, 6 * (len(self._view_points) + 1) // 2))
        for i in range(len(self._view_points)):
            normal = polar2cart(self._view_points[i, 0, ...], self._view_points[i, 1, ...], self._view_points[i, 2, ...])
            contour = self._get_contour(normal, spine_mesh)
            result.append(self._get_image(contour))
            ax[i // 2, i % 2].imshow(result[-1])
            ax[i // 2, i % 2].set_title(f'view_point#{i}')
        plt.savefig("projectionsss.pdf", dpi=600)
        plt.show()
        return result
    
    def _get_contour(self, normal, mesh: Polyhedron_3):
        rotation_matrix = self._get_rotation(normal)
        res_poly = MultiPolygon()
        facet_points = map(lambda facet: np.array([point_2_list(h.vertex().point()) for h in [facet.halfedge(), facet.halfedge().next(), facet.halfedge().next().next()]]), mesh.facets())
        facet_points = map(lambda points: np.matmul(points, rotation_matrix)[...,[0,1]], facet_points)
        
        for facet_2d in facet_points:
            cur_poly = Polygon(facet_2d).buffer(1).buffer(-1)
            if not cur_poly.is_valid:
                cur_poly = cur_poly.convex_hull
                if not cur_poly.is_valid:
                    continue
            res_poly = res_poly.union(cur_poly)
            if type(res_poly) is Polygon:
                res_poly = MultiPolygon([res_poly])
            res_poly = make_valid(res_poly)
        
        res_poly = max(res_poly.geoms, key=lambda a: a.area)
        return res_poly


    @staticmethod
    def distance(mesh_descr1: "LightFieldZernikeMomentsSpineMetric", mesh_descr2: "LightFieldZernikeMomentsSpineMetric") -> float:
        return LightFieldZernikeMomentsSpineMetric.repr_distance(np.array(mesh_descr1._value), np.array(mesh_descr2._value))

    @staticmethod
    def repr_distance(data1: np.ndarray, data2: np.ndarray, view_points_squared: int = 25):
        if data1.ndim != 2:
            data1 = data1.reshape(view_points_squared, int(data1.shape[0]/view_points_squared))
            data2 = data2.reshape(view_points_squared, int(data2.shape[0]/view_points_squared))
        cost_matrix = [[distance.cityblock(m1, m2) if not(np.isnan(m2).any() or np.isnan(m1).any()) else 0 for m2 in data1] for m1 in data2]
        m2_ind, m1_ind = linear_sum_assignment(cost_matrix)
        return sum(distance.cityblock(data2[m2_i], data1[m1_i]) for m2_i, m1_i in zip(m2_ind, m1_ind))

    @classmethod
    def get_distribution(cls, metrics: List["SpineMetric"]) -> np.ndarray:
        spikes_deskr = np.asarray([metric.value for metric in metrics])
        return np.mean(spikes_deskr, 0)

    @classmethod
    def _show_distribution(cls, metrics: List["SpineMetric"]) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot(cls.get_distribution(metrics))

    def parse_value(self, value_str):
        value_str = value_str.replace('\n', '')
        value_str = value_str.replace('  ', ' ')
        k = value_str.split("] [ ")
        k[0] = k[0][3:]
        k[-1] = k[-1][:-2]
        value = [np.fromstring(val, sep=" ", dtype="complex") for val in k]
        self.value = value

    def value_as_lists(self) -> List[Any]:
        return [*self.value]

    def clasterization_preprocess(self, zernike_postprocess='real', **kwargs) -> Any:
        self.value = [[m.real if zernike_postprocess == 'real' else abs(m) for m in moments] for moments in self._value]

    def show(self, image_size: int = 30) -> widgets.Widget:
        out = widgets.Output()
        with out:
            fig, ax = plt.subplots(ncols=2, nrows=(len(self.value) + 1) // 2, figsize=(12, 6 * (len(self.value) + 1) // 2))
            for i, projection in enumerate(self.value):
                ax[i // 2, i % 2].imshow(self._recover_projection(projection, image_size))
            plt.savefig("zernike_moments_images.pdf", dpi=600)
            plt.show()
        return out

    @staticmethod
    def _recover_projection(zernike_moments: List[float], image_size: int):
        radius = image_size // 2
        Y, X = np.meshgrid(range(image_size), range(image_size))
        Y, X = ((Y - radius)/radius).ravel(), ((X - radius)/radius).ravel()

        circle_mask = (np.sqrt(X ** 2 + Y ** 2) <= 1)
        result = np.zeros(len(circle_mask), dtype=complex)
        Y, X = Y[circle_mask], X[circle_mask]
        computed = np.zeros(len(Y), dtype=complex)
        i = 0
        n = 0
        while i < len(zernike_moments):
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    vxy = LightFieldZernikeMomentsSpineMetric.zernike_poly(Y, X, n, l)
                    computed += zernike_moments[i] * vxy
                    i += 1
            n += 1
        #computed = computed - min(computed) #/ (max(computed) - min(computed))
        result[circle_mask] = computed
        return result.reshape((image_size,)*2).real

    @staticmethod
    def zernike_poly(Y, X, n, l):
        l = abs(l)
        if (n - l) % 2 == 1:
            return np.zeros(len(Y), dtype=complex)
        Rho, _, Phi = cart2polar(X, Y, np.zeros(len(X)))
        multiplier = (1.*np.cos(Phi) + 1.j*np.sin(Phi)) ** l
        radial = np.sum([(-1.) ** m * factorial(n - m) /
                         (factorial(m) * factorial((n + l - 2 * m) // 2) * factorial((n - l - 2 * m) // 2)) *
                         np.power(Rho, n - 2 * m)
                         for m in range(int((n - l) // 2 + 1))],
                        axis=0)
        return radial * multiplier

    def _calculate_moment(self, image: np.ndarray, degree: int = 8):
        radius = image.shape[0] // 2
        moments = []
        Y, X = np.meshgrid(range(image.shape[0]), range(image.shape[1]))
        Y, X = ((Y - radius) / radius).ravel(), ((X - radius) / radius).ravel()

        circle_mask = (np.sqrt(X ** 2 + Y ** 2) <= 1)
        Y, X = Y[circle_mask], X[circle_mask]

        frac_center = np.array(image.ravel()[circle_mask], np.double)
        frac_center /= frac_center.sum()

        for n in range(degree + 1):
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    vxy = self.zernike_poly(Y, X, n, l)
                    moments.append(sum(frac_center * np.conjugate(vxy)) * (n + 1)/np.pi)

        return np.array(moments)

