import os
import sys
import numpy as np
from pymoo.core.problem import Problem

def load_tsp_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    coords = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    has_coord_section = any("NODE_COORD_SECTION" in line for line in lines)
    in_coord = not has_coord_section

    for line in lines:
        line = line.strip()
        if not line or line == "EOF":
            if in_coord and has_coord_section:
                break
            continue
        if "NODE_COORD_SECTION" in line:
            in_coord = True
            continue
        if not in_coord:
            continue

        parts = line.split()
        try:
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))
            elif len(parts) == 2:
                coords.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue

    return np.array(coords)

def generate_random_cities(n_cities=25, seed=42, x_range=(10, 90), y_range=(10, 90)):
    np.random.seed(seed)
    xs = np.random.uniform(x_range[0], x_range[1], size=n_cities)
    ys = np.random.uniform(y_range[0], y_range[1], size=n_cities)
    return np.column_stack([xs, ys])

class BiObjectiveTSP(Problem):
    def __init__(self, coordinates):
        self.coordinates = np.array(coordinates, dtype=float)
        self.n_cities = len(coordinates)
        
        diff = self.coordinates[:, np.newaxis, :] - self.coordinates[np.newaxis, :, :]
        self.dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        super().__init__(
            n_var=self.n_cities,
            n_obj=2,
            n_ieq_constr=0,
            xl=0,
            xu=self.n_cities - 1,
            vtype=int
        )

    def _evaluate(self, x, out, *args, **kwargs):
        n_pop = x.shape[0]
        f1 = np.zeros(n_pop)
        f2 = np.zeros(n_pop)

        for i in range(n_pop):
            tour = x[i]

            tour_shifted = np.roll(tour, -1)
            dist_edges = self.dist_matrix[tour, tour_shifted]
            f1[i] = np.sum(dist_edges)

            step_dists = self.dist_matrix[tour[:-1], tour[1:]]
            arrival_times = np.cumsum(step_dists)
            f2[i] = np.sum(arrival_times)

        out["F"] = np.column_stack([f1, f2])
