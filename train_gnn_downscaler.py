#!/usr/bin/env python3
# train_gnn_downscaler.py
import os
import sys
import glob
import argparse
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from scipy.spatial import Delaunay
from scipy.interpolate import RegularGridInterpolator

# Local module imports
from mesh_triangulation import TriangularPrismMesh
from gnn_climate_emulator import TriangularGNNClimateEmulator


class ClimateDownscaleDataset(Dataset):
    """
    Dataset that pairs coarse-resolution climate states (CMIP6/coarsened ERA5)
    with ground-truth high-resolution targets on an unstructured Delaunay triangular mesh.
    Supports multi-step autoregressive rollout sequences.
    """
    def __init__(self, bbox, num_nodes=350, rollout_steps=3, num_samples=60, low_res_deg=1.5):
        super().__init__()
        self.bbox = bbox
        self.num_nodes = num_nodes
        self.rollout_steps = rollout_steps
        self.low_res_deg = low_res_deg
        
        # 1. Initialize High-Resolution Delaunay Triangular Prism Mesh
        self.mesh = TriangularPrismMesh(bbox=bbox, num_nodes=num_nodes)
        self.total_nodes = self.mesh.total_nodes
        self.lats = self.mesh.node_coords[:, 0]
        self.lons = self.mesh.node_coords[:, 1]
        
        # Static surface characteristics on the mesh
        self.orography = np.clip(600.0 * np.sin(self.lons * 0.15) * np.cos(self.lats * 0.2), 0.0, 2800.0).astype(np.float32)
        self.lsm = (self.orography > 25.0).astype(np.float32)
        
        # 2. Synthesize or Load Sequence of Multi-Year Climate States
        self.num_samples = num_samples
        self.co2_levels = np.linspace(350.0, 750.0, num_samples) # Historical to SSP5-8.5
        
    def __len__(self):
        return max(1, self.num_samples - self.rollout_steps)

    def __getitem__(self, idx):
        """
        Returns a sequence of (rollout_steps) input-target pairs:
        - input_x: [Total_Nodes, 12 features] (coarse interpolated field + static terrain + CO2)
        - target_y: [Total_Nodes, 4 targets] (high-res true delta: t2m, d2m, msl, precip)
        """
        sequence_data = []
        for step in range(self.rollout_steps):
            sample_idx = idx + step
            co2 = self.co2_levels[sample_idx]
            co2_norm = (co2 - 280.0) / 700.0
            solar = np.sin((sample_idx % 12) * 2.0 * np.pi / 12.0)
            
            # Synthetic High-Res True Field
            t_base = 288.15 - 0.0065 * self.orography - 0.5 * (self.lats - self.bbox["lat_min"])
            t_warming = 3.0 * np.log2(co2 / 280.0)
            true_t2m = t_base + t_warming + 2.0 * solar
            true_d2m = true_t2m - (4.0 + 4.0 * self.lsm)
            true_msl = 101325.0 - 500.0 * np.cos(self.lons * 0.1)
            true_tp = np.clip((solar + 1.0) * (self.orography / 800.0) * 10.0, 0.0, 50.0)
            
            # Simulate Coarse Model Input (e.g., 1.5° GCM grid interpolated to mesh nodes)
            coarse_lats = np.arange(self.bbox["lat_min"], self.bbox["lat_max"] + self.low_res_deg, self.low_res_deg)
            coarse_lons = np.arange(self.bbox["lon_min"], self.bbox["lon_max"] + self.low_res_deg, self.low_res_deg)
            clon_g, clat_g = np.meshgrid(coarse_lons, coarse_lats)
            coarse_t2m_grid = 288.15 + t_warming - 0.5 * (clat_g - self.bbox["lat_min"])
            
            rgi = RegularGridInterpolator((coarse_lats, coarse_lons), coarse_t2m_grid, method="linear", bounds_error=False, fill_value=None)
            coarse_t2m_nodes = rgi(self.mesh.node_coords).astype(np.float32)
            
            # Node feature vector (12 features)
            node_x = np.zeros((self.total_nodes, 12), dtype=np.float32)
            node_x[:, 0] = coarse_t2m_nodes
            node_x[:, 1] = coarse_t2m_nodes - 5.0
            node_x[:, 2] = 5.0  # u-wind approx
            node_x[:, 3] = 2.0  # v-wind approx
            node_x[:, 4] = true_msl
            node_x[:, 5] = 0.0  # previous precip
            node_x[:, 6] = self.orography
            node_x[:, 7] = self.lsm
            node_x[:, 8] = 5500.0 # z500
            node_x[:, 9] = 250.0  # t500
            node_x[:, 10] = co2_norm
            node_x[:, 11] = solar
            
            # Ground-truth targets (high-resolution residuals)
            target_y = np.stack([
                true_t2m - coarse_t2m_nodes, # Temperature downscaling residual
                true_d2m - (coarse_t2m_nodes - 5.0),
                (true_msl - 101325.0) / 100.0,
                true_tp
            ], axis=-1).astype(np.float32)
            
            pyg_step = self.mesh.build_pyg_graph(node_x, target_y)
            sequence_data.append(pyg_step)
            
        return sequence_data


class ConservationPhysicsClimateLoss(nn.Module):
    """
    Combines:
    1. Smooth L1 downscaling fidelity loss.
    2. Enthalpy/Energy conservation penalty across the mesh area.
    3. Thermodynamic saturation constraint (Td <= T).
    """
    def __init__(self, node_coords):
        super().__init__()
        self.node_coords = node_coords
        self.huber = nn.SmoothL1Loss()
        
    def forward(self, pred_y, target_y, input_x):
        # 1. Base accuracy loss across predicted climate residuals
        loss_fid = self.huber(pred_y, target_y)
        
        # 2. Thermodynamic Saturation Check: Td cannot exceed T
        pred_t2m = input_x[:, 0] + pred_y[:, 0]
        pred_d2m = input_x[:, 1] + pred_y[:, 1]
        supersat_violation = torch.relu(pred_d2m - pred_t2m)
        loss_thermo = torch.mean(supersat_violation ** 2) * 3.0
        
        # 3. Domain-Wide Enthalpy Conservation Constraint
        # Mean downscaled residual temperature across the domain should balance coarse input energy
        domain_energy_residual = torch.mean(pred_y[:, 0])
        loss_energy = (domain_energy_residual ** 2) * 1.5
        
        # 4. Non-negative precipitation penalty
        loss_precip_sign = torch.mean(torch.relu(-pred_y[:, 3]) ** 2) * 5.0
        
        return loss_fid + loss_thermo + loss_energy + loss_precip_sign


def train_climate_downscaler(args):
    bbox = {
        "lat_min": args.lat_min,
        "lat_max": args.lat_max,
        "lon_min": args.lon_min,
        "lon_max": args.lon_max
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("=" * 75)
    print("  TRAINING UNSTRUCTURED TRIANGULAR GNN CLIMATE DOWNSCALER (PYTHON 3)")
    print("=" * 75)
    print(f"[DEVICE] Training on: {device}")
    print(f"[MESH]   Delaunay Vertices: {args.nodes} nodes | Rollout Depth: {args.rollout_steps} steps")
    print(f"[DOMAIN] Lat: [{bbox['lat_min']}, {bbox['lat_max']}], Lon: [{bbox['lon_min']}, {bbox['lon_max']}]")
    
    # 1. Build Dataset
    dataset = ClimateDownscaleDataset(
        bbox=bbox,
        num_nodes=args.nodes,
        rollout_steps=args.rollout_steps,
        num_samples=args.samples
    )
    
    # 2. Initialize Model, Optimizer, and Physics Loss
    model = TriangularGNNClimateEmulator(in_channels=12, hidden_channels=args.hidden_dim, out_channels=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = ConservationPhysicsClimateLoss(dataset.mesh.node_coords)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.save_weights)), exist_ok=True)
    best_loss = float("inf")
    
    print(f"[TRAIN] Commencing {args.epochs} training epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        
        for sample_idx in range(len(dataset)):
            sequence = dataset[sample_idx]
            optimizer.zero_grad()
            
            rollout_loss = 0.0
            # Multi-step autoregressive rollout through graph
            curr_state = sequence[0].clone().to(device)
            for step in range(args.rollout_steps):
                target = sequence[step].y.to(device)
                pred_delta = model(curr_state)
                
                step_loss = criterion(pred_delta, target, curr_state.x)
                rollout_loss += step_loss
                
                # Feedback prediction for next rollout step
                if step < args.rollout_steps - 1:
                    next_x = sequence[step + 1].x.clone().to(device)
                    next_x[:, 0] += pred_delta[:, 0].detach() * 0.2 # Temperature coupling
                    next_x[:, 1] += pred_delta[:, 1].detach() * 0.2 # Dewpoint coupling
                    curr_state.x = next_x
            
            rollout_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_epoch_loss += rollout_loss.item()
            
        scheduler.step()
        avg_loss = total_epoch_loss / len(dataset)
        
        status = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.save_weights)
            status = " --> [SAVED BEST CHECKPOINT]"
            
        if epoch % 5 == 0 or epoch == 1 or status:
            print(f"  Epoch [{epoch:03d}/{args.epochs:03d}] | Loss: {avg_loss:.5f}{status}")
            
    print("=" * 75)
    print(f"[COMPLETE] Optimized downscaler weights saved to: {args.save_weights}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Triangular Prism GNN Climate Downscaler (python3)")
    parser.add_argument("--lat-min", type=float, default=25.0)
    parser.add_argument("--lat-max", type=float, default=45.0)
    parser.add_argument("--lon-min", type=float, default=-105.0)
    parser.add_argument("--lon-max", type=float, default=-75.0)
    parser.add_argument("--nodes", type=int, default=350, help="Number of Delaunay mesh vertices")
    parser.add_argument("--rollout-steps", type=int, default=3, help="Autoregressive multi-step rollout depth")
    parser.add_argument("--samples", type=int, default=50, help="Number of climate state time slices")
    parser.add_argument("--hidden-dim", type=int, default=64, help="GNN hidden feature dimensions")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--force-cpu", action="store_true", help="Force execution on CPU")
    parser.add_argument("--save-weights", type=str, default="./checkpoints/climate_downscaler_best.pt")
    train_climate_downscaler(parser.parse_args())
