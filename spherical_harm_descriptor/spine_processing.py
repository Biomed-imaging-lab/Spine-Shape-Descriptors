import functools
import os
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from spherical_harmonic_metric import SpineMetric, _mesh_to_v_f, v_f_to_mesh, SphericalGarmonicsSpineMetric

import sys
import tblib.pickling_support
tblib.pickling_support.install()


class ExceptionWrapper(object):

    def __init__(self, ee):
        self.ee = ee
        __, __, self.tb = sys.exc_info()

    def re_raise(self):
        raise self.ee.with_traceback(self.tb)



def load_spine_meshes(dirpath: str, spine_file_pattern: str) -> dict:
    spine_meshes = {}
    path = Path(dirpath)
    spine_names = list(path.glob(spine_file_pattern))
    for spine_name in spine_names:
        spine_meshes[str(spine_name).replace('\\', '/')] = Polyhedron_3(str(spine_name).replace('\\', '/'))
    return spine_meshes


def calculate_sph_harm_metric(spine_meshes: dict, params, processes=-1) -> dict:
    processes = min(processes, os.cpu_count()) if processes > 0 else os.cpu_count()
    spines = [spine_name for spine_name in spine_meshes.keys()]
    spines_vf = [_mesh_to_v_f(spine_meshes[spine_name]) for spine_name in spines]
    metrics = {}
    calculate_parallel = functools.partial(calculate_sph_harm_spine, params=params)

    print(f"run pool with {processes} processes num")
    with Pool(processes) as pool:
        results = list(tqdm.tqdm(pool.imap(calculate_parallel, spines_vf, chunksize=100), total=len(spines_vf)))

        for spine_name, result in zip(spines, results):
            if isinstance(result, ExceptionWrapper):
                print(result)
                print(spine_name)
                print()
                continue
            metrics[spine_name] = result
    return metrics


def calculate_sph_harm_spine(mesh_v_f: Tuple[np.ndarray, np.ndarray], params):
    if params is None:
        params = {}
    spine_mesh = v_f_to_mesh(*mesh_v_f)
    try:
        metric = SphericalGarmonicsSpineMetric(spine_mesh, **params)
    except Exception as e:
        return ExceptionWrapper(e)
    return metric.value


def save_metrics(spine_metrics: dict, output_path: str):
    lines = ["SpineFileName,SphericalGarmonics"]
    for k, v in spine_metrics.items():
        lines.append(f"{k},\"{str(v)}\"")
    with open(output_path, "w") as f:
        f.writelines(lines)