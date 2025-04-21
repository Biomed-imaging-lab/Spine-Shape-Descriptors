import functools
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

import numpy as np
import tqdm
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from light_field import _mesh_to_v_f, v_f_to_mesh, LightFieldZernikeMomentsSpineMetric
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


def calculate_light_field_metric(spine_meshes: dict, params, processes=-1) -> dict:
    processes = min(processes, os.cpu_count()) if processes > 0 else os.cpu_count()
    spines = [spine_name for spine_name in spine_meshes.keys()]
    spines_vf = [_mesh_to_v_f(spine_meshes[spine_name]) for spine_name in spines]
    metrics = {}
    calculate_parallel = functools.partial(calculate_light_field_spine, params=params)

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


def calculate_light_field_spine(mesh_v_f: Tuple[np.ndarray, np.ndarray], params):
    if params is None:
        params = {}
    spine_mesh = v_f_to_mesh(*mesh_v_f)
    try:
        metric = LightFieldZernikeMomentsSpineMetric(spine_mesh, **params)
    except Exception as e:
        return ExceptionWrapper(e)
    return metric.value


def save_metrics(spine_metrics: dict, output_path: str):
    lines = ["Spine File,LightFieldZernikeMoments"]
    for k, v in spine_metrics.items():
        lines.append(f"{k},\"{str(v)}\"")
    with open(output_path, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    dataset_path = "9009"
    spine_file_pattern = "**/output/**/spine_*.off"
    spine_dataset = load_spine_meshes(dataset_path, spine_file_pattern="**/output/**/spine_*.off")
    metrics_dataset = calculate_light_field_metric(spine_dataset, {"view_points": 5, "order": 10})
    save_metrics(metrics_dataset, f"{dataset_path}/light_field.csv")
