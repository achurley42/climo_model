\# Standalone AI Climate Model (Delaunay Triangular Prism GNN Edition)



\## 1. System Overview



This repository provides an operational and research-grade Python 3 pipeline engineered for regional climate downscaling and decadal climate emulation in standalone or air-gapped environments.



Unlike short-term weather models that solve initial-value problems on regular latitude/longitude grids, this system treats regional climate as a \*\*boundary-value problem\*\* using an \*\*Unstructured Triangular Prism Grid\*\* (horizontal Delaunay triangulation coupled across vertical isobaric layers) powered by a \*\*Graph Neural Network (GNN)\*\* using `torch\_geometric`.



\### Primary Capabilities:

1\. \*\*Statistical Downscaling \& Super-Resolution:\*\* Ingests coarse global climate model states (e.g., $1^\\circ\\text{--}2^\\circ$ CMIP6 scenarios or coarsened ERA5) and projects them onto high-density Delaunay surface meshes ($\\sim 5\\text{--}10\\,\\text{km}$), conditioned on high-resolution terrain elevation ($z$) and land-sea masks (LSM).

2\. \*\*Multi-Decadal Climate Emulation:\*\* Simulates multi-year and decadal thermodynamic anomalies driven by prescribed greenhouse gas radiative forcing (e.g., atmospheric $\\text{CO}\_2$ concentration in ppm, Shared Socioeconomic Pathways like SSP2-4.5 or SSP5-8.5).

3\. \*\*Sub-seasonal to Seasonal (S2S) Forecasting:\*\* Projects monthly-mean anomaly evolutions conditioned on astronomical solar declination cycles and persistent boundary anomalies.



\---



\## 2. Mathematical \& Spatial Formulation



```

&#x20;                    (Node j, Level k+1)

&#x20;                           /\\

&#x20;                          /  \\

&#x20;                         /\_\_\_\_\\ (Node k, Level k+1)

&#x20;            (Node i, Level k+1) \\

&#x20;                        |  /\\  |

&#x20;                        | /  \\ |  <--- Vertical Prism Walls (Advection Edges)

&#x20;                        |/\_\_\_\_\\|

&#x20;                  (Node i, Level k)  (Node k, Level k)

```



\### 2.1 Horizontal Delaunay Triangulation ($\\mathcal{M}\_H$)

Rather than projecting climate states onto rectilinear latitude/longitude grids (which suffer from severe polar distortion, unphysical boundary reflections, and numerical convergence issues), the horizontal domain is discretized into a planar Delaunay graph $\\mathcal{G} = (\\mathcal{V}, \\mathcal{E})$:

\* \*\*Vertices ($\\mathcal{V}$):\*\* $N$ quasi-uniformly distributed spatial coordinates $\\mathbf{x}\_i = (\\text{lat}\_i, \\text{lon}\_i)$, anchored along domain perimeter bounds.

\* \*\*Edges ($\\mathcal{E}$):\*\* Bidirectional graph connections ensuring the circumcircle of any triangle contains no interior vertices.

\* \*\*Dual Cells:\*\* Each vertex is assigned a dual Voronoi area $A\_i$ to ensure conservative heat, mass, and moisture fluxes.



\### 2.2 Vertical Extension to Triangular Prisms ($\\mathcal{M}\_{3D}$)

Each horizontal triangle $\\triangle\_{ijk}$ is extruded vertically across $K$ discrete isobaric surfaces ($p \\in \\{850, 700, 500, 250\\}\\,\\text{hPa}$):

\* \*\*Prism Volume:\*\* $V = A\_{\\triangle} \\times \\Delta z\_k = A\_{\\triangle} \\left( \\frac{R\_d \\bar{T}}{g} \\ln \\frac{p\_{k}}{p\_{k+1}} \\right)$

\* \*\*Flux Coupling:\*\* Horizontal flux exchange is evaluated across adjacent rectangular prism walls via message-passing graph convolutions (`GCNConv`), while vertical coupling is maintained hydrostatically across prism caps.



\### 2.3 Physics-Constrained Loss Formulation

The GNN is optimized using a compound loss function that enforces thermodynamic saturation limits and global domain enthalpy conservation:

$$\\mathcal{L}\_{\\text{total}} = \\mathcal{L}\_{\\text{Huber}}(\\hat{\\mathbf{y}}, \\mathbf{y}) + \\lambda\_1 \\left( \\frac{1}{|\\mathcal{V}|} \\sum\_{i \\in \\mathcal{V}} \\Delta \\hat{T}\_i \\right)^2 + \\lambda\_2 \\, \\text{ReLU}(\\hat{T}\_d - \\hat{T})^2 + \\lambda\_3 \\, \\text{ReLU}(-\\hat{P})^2$$



\---



\## 3. Directory Layout



```text

/af\_triangular\_climate\_ai

|-- checkpoints/                  # Trained neural weights (.pt)

|-- output\_simulations/           # Output directory for NetCDF, CSV, and mesh data

|-- mesh\_triangulation.py         # Delaunay mesh generator \& Barycentric regridder

|-- gnn\_climate\_emulator.py       # Message-Passing GNN (PyTorch Geometric)

|-- train\_gnn\_downscaler.py       # Training pipeline with multi-step autoregressive rollout

|-- run\_triangular\_climate.py     # CLI driver for decadal and seasonal climate simulations

|-- visualize\_climate\_output.py   # Multi-format CSV \& NetCDF visualizer (mtri tricontourf)

`-- README.md                     # This documentation file

```



\---



\## 4. Environment Setup \& Dependencies



All components are written in and executed using \*\*`python3`\*\*.



\### Conda Environment Setup (Recommended)

Because this architecture operates on graph neural networks and geospatial arrays, install dependencies via `conda-forge`:



```bash

\# 1. Create Base Geospatial \& Scientific Environment

conda create -n af\_climate\_ai -c conda-forge python=3.11 numpy pandas xarray netcdf4 zarr scipy matplotlib cdsapi gcsfs fsspec -y

conda activate af\_climate\_ai



\# 2. Install PyTorch (CPU configuration)

python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu



\# 2b. Install PyTorch (NVIDIA CUDA configuration - alternative)

\# python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121



\# 3. Install PyTorch Geometric

python3 -m pip install torch\_geometric

```



\### Optional Copernicus CDS API Setup

If fetching baseline monthly ERA5 observations for S2S validation, configure your `\~/.cdsapirc`:

```bash

cat << 'EOF' > \~/.cdsapirc

url: https://cds.climate.copernicus.eu/api

key: YOUR\_PERSONAL\_API\_KEY

EOF

chmod 600 \~/.cdsapirc

```



\---



\## 5. Execution Workflows



\### 5.1 Training the GNN Climate Downscaler (`train\_gnn\_downscaler.py`)

Trains the model to downscale coarse climate anomalies to dense Delaunay mesh vertices using a \*\*multi-step autoregressive rollout (3–5 steps)\*\* and enthalpy conservation physics:



```bash

python3 train\_gnn\_downscaler.py \\

&#x20; --lat-min 25.0 --lat-max 45.0 \\

&#x20; --lon-min -100.0 --lon-max -80.0 \\

&#x20; --nodes 350 \\

&#x20; --rollout-steps 3 \\

&#x20; --epochs 30 \\

&#x20; --lr 1e-3 \\

&#x20; --save-weights ./checkpoints/gulf\_climate\_downscaler.pt

```



\#### Key Training Parameters:

\* `--nodes`: Number of quasi-uniform Delaunay vertices generated across the bounding box.

\* `--rollout-steps`: Autoregressive rollout depth during training to prevent numerical drift over decadal runs.

\* `--force-cpu`: Forces execution on multi-core CPU if dedicated GPU hardware is unavailable.



\---



\### 5.2 Decadal Climate Simulation (`run\_triangular\_climate.py`)

Simulates multi-year or decadal regional climate trends conditioned on greenhouse gas concentrations ($\\text{CO}\_2$ ppm):



```bash

\# Run a 25-year projection under high-emission forcing (SSP5-8.5 / 585 ppm CO2)

python3 run\_triangular\_climate.py \\

&#x20; --lat-min 20.0 --lat-max 50.0 \\

&#x20; --lon-min -120.0 --lon-max -70.0 \\

&#x20; --nodes 350 \\

&#x20; --years 25 \\

&#x20; --co2-ppm 585.0 \\

&#x20; --outdir ./climate\_run\_ssp585

```



\---



\### 5.3 Seasonal (S2S) Climate Projection

Runs a monthly-stepped simulation initialized from recent boundary states:



```bash

python3 run\_triangular\_climate.py \\

&#x20; --lat-min 25.0 --lat-max 45.0 \\

&#x20; --lon-min -105.0 --lon-max -75.0 \\

&#x20; --nodes 300 \\

&#x20; --years 1 \\

&#x20; --co2-ppm 425.0 \\

&#x20; --outdir ./seasonal\_run

```



\---



\### 5.4 Visualization \& Diagnostic Plotting (`visualize\_climate\_output.py`)

The visualizer handles both time-series CSV metrics and spatial NetCDF maps. For unstructured triangular meshes, it automatically deploys `matplotlib.tri.Triangulation` (`tricontourf`):



```bash

\# 1. Plot Climate Metric Time Series (Temperature trends, Precipitation anomalies)

python3 visualize\_climate\_output.py \\

&#x20; --csv ./climate\_run\_ssp585/triangular\_climate\_timeseries.csv \\

&#x20; --outdir ./climate\_plots



\# 2. Plot Spatial Map on Unstructured Mesh Vertices

python3 visualize\_climate\_output.py \\

&#x20; --nc ./climate\_run\_ssp585/mesh\_nodes\_export.nc \\

&#x20; --step 10 \\

&#x20; --outdir ./climate\_plots

```



\---



\## 6. Output Deliverables \& Schema



| Deliverable | File Name | Format | Description |

| :--- | :--- | :--- | :--- |

| \*\*Climate Time Series\*\* | `triangular\_climate\_timeseries.csv` | CSV | Year-by-year or month-by-month regional mean temperature anomalies, precipitation rates, and energy residuals. |

| \*\*Unstructured Mesh NetCDF\*\* | `mesh\_nodes\_export.nc` | NetCDF4 (UGRID) | High-resolution node coordinates, Delaunay edge indices, and downscaled 3D atmospheric variables. |

| \*\*Diagnostic Time Series Chart\*\* | `climate\_timeseries\_plot.png` | PNG (200 DPI) | 2-panel chart displaying multi-decadal temperature trajectories (with linear trend fitting) and precipitation bars. |

| \*\*Unstructured Spatial Maps\*\* | `spatial\_map\_step\_XX.png` | PNG (200 DPI) | 4-panel spatial diagnostic contour maps rendered via Delaunay triangulation across the triangular prism mesh. |

