from pathlib import Path

HERE = Path(__file__).resolve().parent

# Run mode: plot redraws saved results; invert runs inversion; all does both.
MODE = "plot"
RESUME = False  # True resumes a matching checkpoint; False starts a new inversion.
DRY_RUN = False  # True checks the catalogue and grid without running inversion.

CATALOG = Path("input/events.csv")
OUTPUT = Path("output")
RELIEF = Path("../Data/relief.nc")

LONGITUDE = [119.0, 124.0]  # [west, east], degrees.
LATITUDE = [21.0, 26.0]  # [south, north], degrees.
DEPTH_KM = [0.0, 60.0]  # Inclusive catalogue bounds; inversion nodes exclude both boundaries.
SPACING_KM = [10.0, 5.0]  # [horizontal grid spacing, depth layer spacing].
NEIGHBOR_COUNTS = [45, 15, 3]  # [nearest events, minimum depth-matched events, minimum quadrants].
SUPPORT_KM = [25.0, 5.0]  # [maximum nearest-event radius, depth-support half-width].
EVENT_WEIGHTING = "gaussian_distance"  # exp[-(k_i/d)^2]; use equal for uniform weights.

FRICTION_MU = 0.6  # Fixed friction coefficient.
MECHANISM_UNCERTAINTY_DEG = None  # Common perturbation bound; None uses the default 10 degrees.
PERTURBATION_DRAWS = 100  # Perturbation draws per mechanism, separate from bootstrap repeats.
RANDOM_SEED = 20251201

SEARCH_MODE = "two_stage"  # Coarse search with refinement; one_stage uses a single grid.
ANGLE_STEPS_DEG = [5.0, 5.0, 1.0]  # [one-stage, coarse, fine] orientation increments.
PHI_STEPS = [0.1, 0.1, 0.01]  # [one-stage, coarse, fine] Phi increments.
REFINE_HALF_WIDTH = [5.0, 0.1]  # [orientation half-width in degrees, Phi half-width].
NUMBA_THREADS = 0  # 0 automatically uses the available threads.
BOOTSTRAP = [0, 1]  # [resampling repeats, worker processes]; 0 repeats disables bootstrap.

PLOT_DEPTHS_KM = [5, 10, 15, 20, 25, 30, 35, 40, 45]  # Exact saved depths, in display order.
FIGURE_DPI = 600
FIGURE_FORMATS = ["png", "pdf"]
SAVE_EVERY = [100, 10]  # [progress interval, stress.csv checkpoint interval], in new nodes.


def get_settings():
    return {
        "mode": MODE,
        "resume": RESUME,
        "dry_run": DRY_RUN,
        "catalog": (HERE / CATALOG).resolve(),
        "output": (HERE / OUTPUT).resolve(),
        "relief": (HERE / RELIEF).resolve(),
        "lon_min": LONGITUDE[0],
        "lon_max": LONGITUDE[1],
        "lat_min": LATITUDE[0],
        "lat_max": LATITUDE[1],
        "depth_min_km": DEPTH_KM[0],
        "depth_max_km": DEPTH_KM[1],
        "grid_spacing_km": SPACING_KM[0],
        "layer_spacing_km": SPACING_KM[1],
        "nearest_events": NEIGHBOR_COUNTS[0],
        "min_depth_events": NEIGHBOR_COUNTS[1],
        "min_azimuth_sectors": NEIGHBOR_COUNTS[2],
        "max_radius_km": SUPPORT_KM[0],
        "depth_support_half_width_km": SUPPORT_KM[1],
        "event_weighting": EVENT_WEIGHTING,
        "friction_mu": FRICTION_MU,
        "mechanism_uncertainty_deg": MECHANISM_UNCERTAINTY_DEG,
        "perturbation_draws": PERTURBATION_DRAWS,
        "random_seed": RANDOM_SEED,
        "search_mode": SEARCH_MODE,
        "one_stage_angle_step_deg": ANGLE_STEPS_DEG[0],
        "coarse_angle_step_deg": ANGLE_STEPS_DEG[1],
        "refine_angle_step_deg": ANGLE_STEPS_DEG[2],
        "one_stage_phi_step": PHI_STEPS[0],
        "coarse_phi_step": PHI_STEPS[1],
        "refine_phi_step": PHI_STEPS[2],
        "refine_angle_half_width_deg": REFINE_HALF_WIDTH[0],
        "refine_phi_half_width": REFINE_HALF_WIDTH[1],
        "numba_threads": NUMBA_THREADS,
        "bootstrap_repeats": BOOTSTRAP[0],
        "bootstrap_workers": BOOTSTRAP[1],
        "plot_depths_km": PLOT_DEPTHS_KM.copy(),
        "figure_dpi": FIGURE_DPI,
        "figure_formats": FIGURE_FORMATS.copy(),
        "progress_every": SAVE_EVERY[0],
        "checkpoint_every": SAVE_EVERY[1],
    }
