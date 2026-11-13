#!/usr/bin/env python3
# visualize_climate_output.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

try:
    import xarray as xr
except ImportError:
    xr = None


def plot_csv_timeseries(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[PLOT] Reading CSV timeseries: {csv_path}")
    df = pd.read_csv(csv_path)

    time_col = None
    for col in ["Year", "Forecast_Hour", "step", "time"]:
        if col in df.columns:
            time_col = col
            break
    if time_col is None:
        time_col = df.columns[0]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    fig.suptitle(f"Climate Simulation Timeseries: {os.path.basename(csv_path)}", fontsize=14, fontweight="bold")

    temp_col = [c for c in df.columns if "temp" in c.lower() or "t2m" in c.lower()]
    if temp_col:
        col = temp_col[0]
        axes[0].plot(df[time_col], df[col], color="crimson", linewidth=2.2, marker="o", markersize=4, label=col)
        axes[0].set_ylabel("Temperature (°C)", fontsize=11, fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend(loc="upper left")

    precip_col = [c for c in df.columns if "precip" in c.lower() or "tp" in c.lower()]
    if precip_col:
        col = precip_col[0]
        axes[1].bar(df[time_col], df[col], color="royalblue", alpha=0.7, width=0.6, label=col)
        axes[1].set_ylabel("Precipitation (mm)", fontsize=11, fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend(loc="upper left")

    axes[1].set_xlabel(time_col, fontsize=11, fontweight="bold")
    out_png = os.path.join(output_dir, "climate_timeseries_plot.png")
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[SUCCESS] Saved CSV Timeseries Chart: {out_png}")


def plot_netcdf_spatial(nc_path, output_dir, step_idx=0):
    if xr is None:
        print("[ERROR] 'xarray' is required. Run: python3 -m pip install xarray netcdf4")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"[PLOT] Loading NetCDF file: {nc_path}")
    ds = xr.open_dataset(nc_path)

    lat_key = [k for k in ds.coords if k.lower() in ["lat", "latitude", "y"]]
    lon_key = [k for k in ds.coords if k.lower() in ["lon", "longitude", "x"]]
    if not lat_key or not lon_key:
        print(f"[ERROR] Latitude/longitude coordinates not identified.")
        return

    lats = ds[lat_key[0]].values
    lons = ds[lon_key[0]].values
    is_unstructured = (lats.ndim == 1 and lons.ndim == 1 and len(lats) == len(lons))

    plot_vars = [v for v in ds.data_vars if v not in ["orog", "lsm", "time_bnds"]][:4]
    if not plot_vars:
        plot_vars = list(ds.data_vars.keys())[:4]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()
    fig.suptitle(f"Spatial Map | File: {os.path.basename(nc_path)} | Step: {step_idx}", fontsize=14, fontweight="bold")

    for i, var_name in enumerate(plot_vars):
        if i >= 4: break
        ax = axes[i]
        da = ds[var_name]
        if "time" in da.dims:
            da = da.isel(time=step_idx)
        data_vals = np.squeeze(da.values)

        if is_unstructured:
            triang = mtri.Triangulation(lons, lats)
            cf = ax.tricontourf(triang, data_vals, levels=14, cmap="Spectral_r" if "t2m" in var_name else "Viridis")
            ax.triplot(triang, color="black", linewidth=0.3, alpha=0.3)
            ax.scatter(lons, lats, color="black", s=3, alpha=0.6)
            ax.set_title(f"{var_name} (Unstructured Delaunay Mesh)", fontsize=11, fontweight="bold")
        else:
            lon_g, lat_g = np.meshgrid(lons, lats) if data_vals.shape == (len(lats), len(lons)) else (lons, lats)
            cf = ax.contourf(lon_g, lat_g, data_vals, levels=14, cmap="RdYlBu_r" if "t2m" in var_name else "YlGnBu")
            ax.set_title(f"{var_name} (Regular Grid)", fontsize=11, fontweight="bold")

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(cf, ax=ax)

    for j in range(len(plot_vars), 4):
        fig.delaxes(axes[j])

    out_png = os.path.join(output_dir, f"spatial_map_step_{step_idx}.png")
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[SUCCESS] Saved Spatial Map: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Unified Climate Simulation Visualizer (python3)")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file")
    parser.add_argument("--nc", type=str, default=None, help="Path to NetCDF4 file")
    parser.add_argument("--step", type=int, default=0, help="Time index for spatial map")
    parser.add_argument("--outdir", type=str, default="./visualizations", help="Output directory")
    args = parser.parse_args()

    if args.csv and os.path.exists(args.csv):
        plot_csv_timeseries(args.csv, args.outdir)
    if args.nc and os.path.exists(args.nc):
        plot_netcdf_spatial(args.nc, args.outdir, step_idx=args.step)


if __name__ == "__main__":
    main()
