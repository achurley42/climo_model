#!/usr/bin/env python3
# mesh_triangulation.py
import numpy as np
import torch
from scipy.spatial import Delaunay
from torch_geometric.data import Data


class TriangularPrismMesh:
    """
    Constructs an unstructured horizontal Delaunay triangulation extended vertically
    into triangular prisms across discrete pressure levels.
    """
    def __init__(self, bbox, num_nodes=250, plevs=[850, 700, 500, 250], random_seed=42):
        self.bbox = bbox
        self.plevs = sorted(plevs, reverse=True)
        self.num_plevs = len(self.plevs)
        self.num_nodes = num_nodes
        
        np.random.seed(random_seed)
        lats = np.random.uniform(bbox["lat_min"], bbox["lat_max"], num_nodes)
        lons = np.random.uniform(bbox["lon_min"], bbox["lon_max"], num_nodes)
        
        corners_lat = [bbox["lat_min"], bbox["lat_min"], bbox["lat_max"], bbox["lat_max"]]
        corners_lon = [bbox["lon_min"], bbox["lon_max"], bbox["lon_min"], bbox["lon_max"]]
        self.node_coords = np.vstack([
            np.column_stack([lats, lons]),
            np.column_stack([corners_lat, corners_lon])
        ]).astype(np.float32)
        self.total_nodes = len(self.node_coords)
        
        self.tri = Delaunay(self.node_coords)
        self.edge_index = self._build_bidirectional_edges()
        
    def _build_bidirectional_edges(self):
        edges = set()
        for simplex in self.tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    u, v = simplex[i], simplex[j]
                    edges.add((u, v))
                    edges.add((v, u))
        edge_array = np.array(list(edges), dtype=np.int64).T
        return torch.tensor(edge_array, dtype=torch.long)

    def barycentric_regrid(self, source_lats, source_lons, gridded_2d_field):
        """Interpolates regular lat/lon fields onto Delaunay mesh vertices."""
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (source_lats, source_lons),
            gridded_2d_field,
            method="linear",
            bounds_error=False,
            fill_value=None
        )
        return interp(self.node_coords).astype(np.float32)

    def build_pyg_graph(self, node_features, target_tendencies=None):
        x = torch.tensor(node_features, dtype=torch.float32)
        y = torch.tensor(target_tendencies, dtype=torch.float32) if target_tendencies is not None else None
        pos = torch.tensor(self.node_coords, dtype=torch.float32)
        return Data(x=x, edge_index=self.edge_index, y=y, pos=pos)
