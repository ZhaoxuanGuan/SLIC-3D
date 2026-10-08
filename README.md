# SLIC-3D
Three-dimensional Stress Likelihood Inversion by Coulomb Failure Criticality.

## Files

```text
SLIC-3D/
├── Config.py
├── SLIC-3D.py
├── Plot_figures.py
├── README.md
├── input/
│   └── events.csv
└── output/
    ├── stress.csv
    ├── Aphi.png / Aphi.pdf
    ├── coverage.png / coverage.pdf
    ├── phi.png / phi.pdf
    ├── sigma1_plunge.png / sigma1_plunge.pdf
    ├── sigma1_azimuth.png / sigma1_azimuth.pdf
    ├── sigma2_azimuth.png / sigma2_azimuth.pdf
    ├── sigma2_plunge.png / sigma2_plunge.pdf
    ├── sigma3_azimuth.png / sigma3_azimuth.pdf
    └── sigma3_plunge.png / sigma3_plunge.pdf
```

`Config.py` defines paths and parameters. `SLIC-3D.py` contains the inversion
and its command-line entry point. `Plot_figures.py` reads `stress.csv` directly
without importing the inversion module.

## Installation and usage

Requires Python 3.10+ and compatible NumPy, pandas, Numba, Matplotlib, and h5py:

```powershell
python -m pip install numpy pandas numba matplotlib h5py
cd SLIC-3D
python -B SLIC-3D.py --mode plot
python -B Plot_figures.py
```

The default `MODE = "plot"` redraws the saved results. Figures are exported as
600 dpi PNG and PDF files at depths of 5, 10, 15, 20, 25, 30, 35, 40, and 45 km.
Edit `Config.py` or override individual settings on the command line:

```powershell
python -B SLIC-3D.py --dry-run
python -B SLIC-3D.py --mode plot --plot-depths-km 5,15,25 --figure-dpi 300
python -B SLIC-3D.py --mode invert --output output_new
python -B SLIC-3D.py --mode all --output output_new
python -B SLIC-3D.py --help
```

The configured input, output, and relief paths are `input/events.csv`, `output`,
and `../Data/relief.nc`. They are resolved relative to the code directory,
so default paths work from other working directories. The shared `Data` folder
is a sibling of `SLIC-3D`; move both folders together when relocating the project.
Change the relative paths in `Config.py`, or use `--relief path/to/relief.nc`.
Relative command-line path overrides are resolved from the working directory.

## Parameters

Edit the upper section of `Config.py`. Related numeric parameters use lists,
with their order and units stated in the adjacent English comments:

```python
LONGITUDE = [119.0, 124.0]
LATITUDE = [21.0, 26.0]
DEPTH_KM = [0.0, 60.0]
SPACING_KM = [10.0, 5.0]
NEIGHBOR_COUNTS = [45, 15, 3]
SUPPORT_KM = [25.0, 5.0]
ANGLE_STEPS_DEG = [5.0, 5.0, 1.0]
PHI_STEPS = [0.1, 0.1, 0.01]
PLOT_DEPTHS_KM = [5, 10, 15, 20, 25, 30, 35, 40, 45]
```

`get_settings()` expands these lists into the parameters used by the command-line
interface and inversion. Plot depths and file formats accept both configuration
lists and comma-separated command-line values.

## Input and retained results

`input/events.csv` is copied unchanged from the original catalogue. Required
columns are `Latitude`, `Longitude`, `Depth`, and the two nodal planes
`strike1,dip1,rake1,strike2,dip2,rake2`. Angles are in degrees and depths in km.
Additional columns such as `Date` and `Time` do not enter the inversion.


`--mode invert` and `--mode all` write `stress.csv` to the specified output
folder. Use `--output output_new` to save a new inversion in a separate folder.
Resuming requires the saved `inversion_signature` to match the code, input,
grid, and inversion parameters.

## Figures and colorbars

| Filename | Quantity | Colorbar range |
| --- | --- | --- |
| Aphi | Continuous stress-regime index Aphi, preferably read from the saved table | 0-3 |
| coverage | Three-dimensional distance to the farthest selected event | 0-25 km, extended for larger values |
| phi | Stress shape ratio Phi | 0-1 |
| sigma1_plunge | Downward plunge of the maximum principal stress axis | 0-90 degrees |
| sigma1_azimuth | Horizontal axial azimuth of the maximum principal stress axis | 0-180 degrees |
| sigma2_azimuth / sigma3_azimuth | Horizontal axial azimuth of the intermediate and minimum principal stress axes | 0-180 degrees |
| sigma2_plunge / sigma3_plunge | Downward plunge of the intermediate and minimum principal stress axes | 0-90 degrees |
