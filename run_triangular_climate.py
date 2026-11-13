# run_triangular_climate.py
import os
import argparse
import numpy as np
import pandas as pd
import torch
from mesh_triangulation import TriangularPrismMesh
from gnn_climate_emulator import TriangularGNNClimateEmulator


def execute_triangular_climate_run(bbox, num_nodes, years, co2_ppm, outdir):
    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[COMPUTE] Executing on device: {device}")

    # 1. Build Delaunay Triangulation
    mesh = TriangularPrismMesh(bbox=bbox, num_nodes=num_nodes)
    print(f"[MESH] Generated Delaunay Graph: {mesh.total_nodes} nodes, {mesh.edge_index.shape[1]} directional edges.")

    # 2. Synthetic Baseline Node Initial State
    # Shape: [Total_Nodes, 12 features] (t2m, d2m, u10, v10, msl, tp, orog, lsm, z500, t500, co2_norm, solar)
    init_features = np.zeros((mesh.total_nodes, 12), dtype=np.float32)
    lats = mesh.node_coords[:, 0]
    lons = mesh.node_coords[:, 1]
    
    init_features[:, 0] = 288.15 - 0.5 * (lats - bbox["lat_min"]) # 2m Temp (K)
    init_features[:, 1] = init_features[:, 0] - 5.0              # 2m Dewpoint (K)
    init_features[:, 4] = 101325.0                               # MSLP (Pa)
    init_features[:, 6] = np.clip(500 * np.sin(lons * 0.2), 0, 2000) # Elevation
    init_features[:, 7] = (init_features[:, 6] > 10).astype(np.float32) # LSM
    init_features[:, 10] = (co2_ppm - 280.0) / 700.0             # Normalized CO2 anomaly

    model = TriangularGNNClimateEmulator(in_channels=12, hidden_channels=64, out_channels=4).to(device)
    model.eval()

    state = init_features.copy()
    results = []

    print(f"[SIMULATION] Rolling forward {years} annual cycles under {co2_ppm} ppm CO2 forcing...")
    with torch.no_grad():
        for step in range(1, years + 1):
            # Dynamic solar forcing
            state[:, 11] = np.sin(step * 2 * np.pi / 1.0)
            graph_data = mesh.build_pyg_graph(state).to(device)
            
            # Predict annual tendencies across triangular prism faces
            tendency = model(graph_data).cpu().numpy()
            
            # Radiative physics feedback: CO2 greenhouse warming scaled by elevation lapse rate
            co2_warming = 3.0 * np.log2(co2_ppm / 280.0) * (step / years)
            state[:, 0] += tendency[:, 0] * 0.1 + (co2_warming * 0.05)
            state[:, 1] = np.minimum(state[:, 1] + tendency[:, 1] * 0.08, state[:, 0]) # Td <= T
            state[:, 4] += tendency[:, 2] * 10.0
            precip = np.clip(tendency[:, 3] + (0.02 * co2_warming), 0, 50.0)

            mean_t_c = float(np.mean(state[:, 0]) - 273.15)
            mean_pr = float(np.mean(precip))
            results.append({"Year": step, "Global_Mean_T2m_C": round(mean_t_c, 3), "Mean_Precip_mm": round(mean_pr, 3)})
            print(f"  -> Year {step:02d} | Mean Temp: {mean_t_c:.2f}°C | Mean Precip: {mean_pr:.2f} mm")

    # 3. Export Node Summaries
    df_out = pd.DataFrame(results)
    csv_path = os.path.join(outdir, "triangular_climate_timeseries.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"[EXPORT] Saved Triangular Prism Timeseries: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triangular Prism Mesh Climate Emulator")
    parser.add_argument("--lat-min", type=float, default=20.0)
    parser.add_argument("--lat-max", type=float, default=50.0)
    parser.add_argument("--lon-min", type=float, default=-120.0)
    parser.add_argument("--lon-max", type=float, default=-70.0)
    parser.add_argument("--nodes", type=int, default=300, help="Number of unstructured Delaunay mesh vertices")
    parser.add_argument("--years", type=int, default=20, help="Simulation duration in years")
    parser.add_argument("--co2-ppm", type=float, default=585.0, help="Greenhouse gas concentration scenario")
    parser.add_argument("--outdir", type=str, default="./triangular_climate_output")
    args = parser.parse_args()

    bbox = {"lat_min": args.lat_min, "lat_max": args.lat_max, "lon_min": args.lon_min, "lon_max": args.lon_max}
    execute_triangular_climate_run(bbox, args.nodes, args.years, args.co2_ppm, args.outdir)
