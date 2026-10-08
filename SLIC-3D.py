from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

sys.dont_write_bytecode = True
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "SLIC-3D-numba"))
import numpy as np
import pandas as pd
from Config import get_settings
from numba import config as numba_config
from numba import get_num_threads, njit, prange, set_num_threads

if __name__ == "__main__":
    sys.modules["SLIC-3D"] = sys.modules[__name__]
MECHANISM_DIVERSITY_LABELS = ("Poor", "Marginal", "Good")
ALGORITHM_VERSION = "3d-slic-2.0-ned-full-orientation-distance-weighted"
COORDINATE_SYSTEM = "North-East-Down; azimuth clockwise from north; plunge positive downward"
TINY = 1e-12
ZOBACK_INDEX = {"SS": 0, "TS": 1, "NS": 2, "NF": 3, "TF": 4, "U": 5}
ZOBACK_CODE_BY_INDEX = {value: key for key, value in ZOBACK_INDEX.items()}
ZOBACK_LABEL_BY_CODE = {
    "NF": "Normal faulting (NF)",
    "NS": "Normal–strike-slip (NS)",
    "SS": "Strike-slip (SS)",
    "TS": "Thrust–strike-slip (TS)",
    "TF": "Thrust faulting (TF)",
    "U": "Unclassified",
}
KM_PER_DEGREE_LAT = 111.32
REGIME_LABELS = ("normal-faulting", "strike-slip", "reverse-faulting")
MECHANISM_DIVERSITY_POOR_MAX_DEG = 30.0
MECHANISM_DIVERSITY_GOOD_MIN_DEG = 45.0
RESULT_KEY_COLUMNS = ("grid_x", "grid_y", "grid_z")
CELL_CONTEXT_FIELDS = frozenset(
    {
        "grid_x",
        "grid_y",
        "grid_z",
        "longitude",
        "latitude",
        "depth_km",
        "local_event_count",
        "nearest_event_radius_km",
        "depth_matched_event_count",
        "horizontal_azimuth_sector_count",
        "event_weighting",
        "event_weight_sum",
        "weight_effective_event_count",
        "local_catalogue_signature",
        "source_inversion_id",
        "reused_local_catalogue",
        "inversion_signature",
    }
)


@dataclass(frozen=True)
class SLICConfig:
    friction_mu: float = 0.6
    perturbation_draws: int = 100
    random_seed: int = 20251201
    mechanism_uncertainty_deg: float | None = None
    default_mechanism_uncertainty_deg: float = 10.0
    minimum_mechanism_uncertainty_deg: float = 0.0
    maximum_mechanism_uncertainty_deg: float = 50.0
    search_mode: str = "two_stage"
    one_stage_angle_step_deg: float = 5.0
    one_stage_phi_step: float = 0.1
    coarse_angle_step_deg: float = 5.0
    coarse_phi_step: float = 0.1
    refine_angle_half_width_deg: float = 5.0
    refine_phi_half_width: float = 0.1
    refine_angle_step_deg: float = 1.0
    refine_phi_step: float = 0.01
    grid_dtype: str = "float32"
    rounding_decimals: int = 6
    numba_threads: int = 0

    def validate(self) -> "SLICConfig":
        mode = str(self.search_mode).strip().lower()
        if mode not in {"one_stage", "two_stage"}:
            raise ValueError("search_mode must be 'one_stage' or 'two_stage'.")
        if not abs(float(self.friction_mu) - 0.6) <= 1e-12:
            raise ValueError("friction_mu is fixed at 0.6.")
        if int(self.perturbation_draws) < 1:
            raise ValueError("perturbation_draws must be at least 1.")
        if int(self.numba_threads) < 0:
            raise ValueError("numba_threads cannot be negative.")
        if str(self.grid_dtype) not in {"float32", "float64"}:
            raise ValueError("grid_dtype must be 'float32' or 'float64'.")
        if int(self.rounding_decimals) < 0:
            raise ValueError("rounding_decimals cannot be negative.")
        positive = {
            "one_stage_angle_step_deg": self.one_stage_angle_step_deg,
            "one_stage_phi_step": self.one_stage_phi_step,
            "coarse_angle_step_deg": self.coarse_angle_step_deg,
            "coarse_phi_step": self.coarse_phi_step,
            "refine_angle_step_deg": self.refine_angle_step_deg,
            "refine_phi_step": self.refine_phi_step,
        }
        bad = [name for name, value in positive.items() if float(value) <= 0.0]
        if bad:
            raise ValueError(f"Grid steps must be positive: {', '.join(bad)}.")
        if float(self.refine_angle_half_width_deg) < 0.0:
            raise ValueError("refine_angle_half_width_deg cannot be negative.")
        if float(self.refine_phi_half_width) < 0.0:
            raise ValueError("refine_phi_half_width cannot be negative.")
        lo = float(self.minimum_mechanism_uncertainty_deg)
        default = float(self.default_mechanism_uncertainty_deg)
        hi = float(self.maximum_mechanism_uncertainty_deg)
        if not 0.0 <= lo <= default <= hi <= 90.0:
            raise ValueError(
                "Mechanism-uncertainty bounds must satisfy 0 <= minimum <= default <= maximum <= 90 degrees."
            )
        if self.mechanism_uncertainty_deg is not None:
            value = float(self.mechanism_uncertainty_deg)
            if not lo <= value <= hi:
                raise ValueError(
                    "mechanism_uncertainty_deg must lie within the configured uncertainty bounds."
                )
        return self

    def resolved_friction_mu_grid(self) -> tuple[float, ...]:
        return (0.6,)

    def to_dict(self) -> dict:
        return asdict(self)


def wrap_360(angle):
    value = np.mod(angle, 360.0)
    return float(value) if np.isscalar(angle) else value


def wrap_180(angle):
    value = (np.asarray(angle) + 180.0) % 360.0 - 180.0
    return float(value) if np.isscalar(angle) else value


def strike_dip_to_normal(strike_deg: float, dip_deg: float) -> np.ndarray:
    strike, dip = np.deg2rad([strike_deg, dip_deg])
    normal = np.array(
        [-np.sin(dip) * np.sin(strike), np.sin(dip) * np.cos(strike), -np.cos(dip)],
        dtype=np.float64,
    )
    return normal / np.linalg.norm(normal)


def strike_vector(strike_deg: float) -> np.ndarray:
    strike = math.radians(float(strike_deg))
    return np.array([math.cos(strike), math.sin(strike), 0.0], dtype=np.float64)


def dip_vector(strike_deg: float, dip_deg: float) -> np.ndarray:
    direction = np.cross(strike_dip_to_normal(strike_deg, dip_deg), strike_vector(strike_deg))
    direction /= np.linalg.norm(direction)
    return -direction if direction[2] > 0.0 else direction


def euler_basis(a1_deg: float, a2_deg: float, a3_deg: float) -> np.ndarray:
    a1, a2, a3 = np.deg2rad([a1_deg, a2_deg, a3_deg])
    c1, s1 = (math.cos(a1), math.sin(a1))
    c2, s2 = (math.cos(a2), math.sin(a2))
    c3, s3 = (math.cos(a3), math.sin(a3))
    axis1 = np.array([c1 * c3 - s1 * c2 * s3, s1 * c3 + c1 * c2 * s3, s2 * s3])
    axis2 = np.array([-c1 * s3 - s1 * c2 * c3, -s1 * s3 + c1 * c2 * c3, s2 * c3])
    axis3 = np.array([s1 * s2, -c1 * s2, c2])
    return np.column_stack((axis1, axis2, axis3)).astype(np.float64)


def basis_phi_to_tensor(basis: np.ndarray, phi: float) -> np.ndarray:
    basis = np.asarray(basis, dtype=np.float64)
    return basis @ np.diag([1.0, 2.0 * float(phi) - 1.0, -1.0]) @ basis.T


def vector_to_azimuth_plunge(vector: np.ndarray) -> tuple[float, float]:
    vector = np.asarray(vector, dtype=np.float64).copy().ravel()
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= TINY:
        return (float("nan"), float("nan"))
    vector /= norm
    if vector[2] < 0.0:
        vector = -vector
    plunge = math.degrees(math.asin(float(np.clip(vector[2], -1.0, 1.0))))
    if plunge > 89.999999:
        return (0.0, float(plunge))
    azimuth = math.degrees(math.atan2(vector[1], vector[0])) % 360.0
    if abs(plunge) < 1e-08:
        plunge = 0.0
        azimuth %= 180.0
    return (float(azimuth), float(plunge))


def canonicalize_basis(basis: np.ndarray) -> np.ndarray:
    basis = np.asarray(basis, dtype=np.float64).copy()
    for column in range(3):
        vector = basis[:, column]
        if vector[2] < -TINY:
            basis[:, column] = -vector
        elif abs(vector[2]) <= TINY:
            azimuth = math.degrees(math.atan2(vector[1], vector[0])) % 360.0
            if azimuth >= 180.0:
                basis[:, column] = -vector
    if np.linalg.det(basis) < 0.0:
        basis[:, 2] *= -1.0
    return basis


def tensor_to_basis_phi(tensor: np.ndarray) -> tuple[np.ndarray, float]:
    tensor = np.asarray(tensor, dtype=np.float64)
    symmetric = 0.5 * (tensor + tensor.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    basis = canonicalize_basis(eigenvectors[:, order])
    denominator = float(eigenvalues[0] - eigenvalues[2])
    phi = (
        0.5 if abs(denominator) <= TINY else float((eigenvalues[1] - eigenvalues[2]) / denominator)
    )
    return (basis, float(np.clip(phi, 0.0, 1.0)))


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def align_basis_signs(basis: np.ndarray, reference: np.ndarray) -> np.ndarray:
    options = (
        np.diag([1.0, 1.0, 1.0]),
        np.diag([-1.0, -1.0, 1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([1.0, -1.0, -1.0]),
    )
    best = np.asarray(basis, dtype=np.float64).copy()
    best_angle = float("inf")
    for signs in options:
        candidate = basis @ signs
        angle = rotation_angle_deg(candidate @ reference.T)
        if angle < best_angle:
            best_angle = angle
            best = candidate
    return best


def basis_rotation_angle_deg(basis: np.ndarray, reference: np.ndarray) -> float:
    aligned = align_basis_signs(basis, reference)
    return rotation_angle_deg(aligned @ reference.T)


def axis_angle_deg(axis: np.ndarray, reference: np.ndarray) -> float:
    cosine = abs(float(np.dot(axis, reference)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


@njit(cache=True, fastmath=True, inline="always")
def failure_criticality_numba(sigma_n, tau, friction_mu):
    scale = math.sqrt(1.0 + friction_mu * friction_mu)
    sigma_c = -friction_mu / scale
    tau_c = 1.0 / scale
    distance = math.sqrt((sigma_n - sigma_c) * (sigma_n - sigma_c) + (tau - tau_c) * (tau - tau_c))
    distance_max = math.sqrt((1.0 - sigma_c) * (1.0 - sigma_c) + tau_c * tau_c)
    return min(1.0, max(0.0, 1.0 - distance / distance_max))


def _array(values: Iterable, *, dtype=np.float64) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(list(values), dtype=dtype))


@dataclass(frozen=True)
class FocalMechanismCatalog:
    s1: np.ndarray
    d1: np.ndarray
    r1: np.ndarray
    s2: np.ndarray
    d2: np.ndarray
    r2: np.ndarray
    uncertainty_deg: np.ndarray
    weights: np.ndarray

    @property
    def size(self) -> int:
        return int(self.s1.size)

    @classmethod
    def from_nodal_planes(
        cls,
        s1: Iterable,
        d1: Iterable,
        r1: Iterable,
        s2: Iterable,
        d2: Iterable,
        r2: Iterable,
        *,
        uncertainty_deg: Iterable | float | None = None,
        weights: Iterable | float | None = None,
    ) -> "FocalMechanismCatalog":
        arrays = [_array(values) for values in (s1, d1, r1, s2, d2, r2)]
        sizes = {array.size for array in arrays}
        if len(sizes) != 1 or not arrays[0].size:
            raise ValueError("All nodal-plane columns must have one equal, nonzero length.")
        n_events = int(arrays[0].size)

        def expand(value, default, dtype=np.float64):
            if value is None:
                return np.full(n_events, default, dtype=dtype)
            if np.isscalar(value):
                return np.full(n_events, value, dtype=dtype)
            result = np.asarray(list(value), dtype=dtype)
            if result.size != n_events:
                raise ValueError("Optional catalogue columns must match event count.")
            return np.ascontiguousarray(result)

        uncertainty = expand(uncertainty_deg, np.nan)
        event_weights = expand(weights, 1.0)
        catalog = cls(
            s1=np.ascontiguousarray(wrap_360(arrays[0])),
            d1=arrays[1],
            r1=np.ascontiguousarray(wrap_180(arrays[2])),
            s2=np.ascontiguousarray(wrap_360(arrays[3])),
            d2=arrays[4],
            r2=np.ascontiguousarray(wrap_180(arrays[5])),
            uncertainty_deg=uncertainty,
            weights=event_weights,
        )
        catalog.validate()
        return catalog

    def validate(self) -> None:
        if self.size < 1:
            raise ValueError("The focal-mechanism catalogue cannot be empty.")
        data = np.column_stack((self.s1, self.d1, self.r1, self.s2, self.d2, self.r2))
        if not np.all(np.isfinite(data)):
            raise ValueError("Nodal-plane angles must be finite.")
        if np.any((self.d1 < 0.0) | (self.d1 > 90.0)):
            raise ValueError("Plane-1 dips must lie within [0, 90] degrees.")
        if np.any((self.d2 < 0.0) | (self.d2 > 90.0)):
            raise ValueError("Plane-2 dips must lie within [0, 90] degrees.")
        if np.any(~np.isfinite(self.weights)) or np.any(self.weights <= 0.0):
            raise ValueError("Event weights must be finite and positive.")

    def subset(self, indices: np.ndarray) -> "FocalMechanismCatalog":
        indices = np.asarray(indices, dtype=np.int64)
        return FocalMechanismCatalog(
            **{
                name: np.ascontiguousarray(getattr(self, name)[indices])
                for name in ("s1", "d1", "r1", "s2", "d2", "r2", "uncertainty_deg", "weights")
            }
        )


@dataclass(frozen=True)
class PreparedCatalog:
    n1: np.ndarray
    n2: np.ndarray
    strike1: np.ndarray
    dip1: np.ndarray
    strike2: np.ndarray
    dip2: np.ndarray
    cos_rake1: np.ndarray
    sin_rake1: np.ndarray
    cos_rake2: np.ndarray
    sin_rake2: np.ndarray
    weights: np.ndarray
    weight_sum: float
    eq6: np.ndarray
    uncertainty_deg: np.ndarray
    uncertainty_missing_count: int = 0


def prepare_catalog(catalog: FocalMechanismCatalog, config: SLICConfig) -> PreparedCatalog:
    config.validate()
    catalog.validate()
    eq6 = np.round(
        np.column_stack((catalog.s1, catalog.d1, catalog.r1, catalog.s2, catalog.d2, catalog.r2)),
        int(config.rounding_decimals),
    ).astype(np.float64)
    uncertainty = np.asarray(catalog.uncertainty_deg, dtype=np.float64).copy()
    uncertainty_missing_count = int(np.count_nonzero(~np.isfinite(uncertainty)))
    uncertainty[~np.isfinite(uncertainty)] = config.default_mechanism_uncertainty_deg
    uncertainty = np.clip(
        uncertainty,
        config.minimum_mechanism_uncertainty_deg,
        config.maximum_mechanism_uncertainty_deg,
    )
    uncertainty = np.round(uncertainty, int(config.rounding_decimals))

    def plane_arrays(strike_deg, dip_deg, rake_deg):
        strike, dip, rake = np.deg2rad(np.column_stack((strike_deg, dip_deg, rake_deg))).T
        cs, ss = (np.cos(strike), np.sin(strike))
        cd, sd = (np.cos(dip), np.sin(dip))
        normal = np.stack((-sd * ss, sd * cs, -cd), axis=1)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        strike_vector = np.stack((cs, ss, np.zeros_like(cs)), axis=1)
        dip_vector = np.stack((ss * cd, -cs * cd, -sd), axis=1)
        return (normal, strike_vector, dip_vector, np.cos(rake), np.sin(rake))

    p1 = plane_arrays(eq6[:, 0], eq6[:, 1], eq6[:, 2])
    p2 = plane_arrays(eq6[:, 3], eq6[:, 4], eq6[:, 5])

    def f32(value):
        return np.ascontiguousarray(
            np.round(value, int(config.rounding_decimals)), dtype=np.float32
        )

    weights = f32(catalog.weights)
    return PreparedCatalog(
        n1=f32(p1[0]),
        n2=f32(p2[0]),
        strike1=f32(p1[1]),
        dip1=f32(p1[2]),
        strike2=f32(p2[1]),
        dip2=f32(p2[2]),
        cos_rake1=f32(p1[3]),
        sin_rake1=f32(p1[4]),
        cos_rake2=f32(p2[3]),
        sin_rake2=f32(p2[4]),
        weights=weights,
        weight_sum=float(weights.sum()),
        eq6=np.ascontiguousarray(eq6),
        uncertainty_deg=np.ascontiguousarray(uncertainty),
        uncertainty_missing_count=uncertainty_missing_count,
    )


def resolve_catalog_uncertainty(prepared: PreparedCatalog, config: SLICConfig) -> tuple[float, str]:
    if config.mechanism_uncertainty_deg is not None:
        return (float(config.mechanism_uncertainty_deg), "explicit configuration")
    if prepared.uncertainty_missing_count == prepared.weights.size:
        return (
            float(config.default_mechanism_uncertainty_deg),
            "default because catalogue uncertainties are missing",
        )
    values = np.asarray(prepared.uncertainty_deg, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size and np.allclose(finite, finite[0], rtol=0.0, atol=1e-09):
        if prepared.uncertainty_missing_count:
            return (float(finite[0]), "common value after missing event uncertainties use default")
        return (float(finite[0]), "common catalogue value")
    return (
        float(config.default_mechanism_uncertainty_deg),
        "default because event uncertainties are not common",
    )


def make_perturbation_ensemble(
    prepared: PreparedCatalog, config: SLICConfig, *, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, float, str]:
    draws = int(config.perturbation_draws)
    if draws < 1:
        raise ValueError("perturbation_draws must be at least 1.")
    uncertainty_deg, uncertainty_mode = resolve_catalog_uncertainty(prepared, config)
    rng = np.random.default_rng(int(config.random_seed if seed is None else seed))
    mechanisms = prepared.eq6
    n_events = mechanisms.shape[0]
    strike0, dip0, rake0 = np.deg2rad(mechanisms[:, :3]).T
    normal0 = np.stack(
        (-np.sin(dip0) * np.sin(strike0), np.sin(dip0) * np.cos(strike0), -np.cos(dip0)), axis=1
    )
    strike_vector0 = np.stack((np.cos(strike0), np.sin(strike0), np.zeros(n_events)), axis=1)
    dip_vector0 = np.stack(
        (np.sin(strike0) * np.cos(dip0), -np.cos(strike0) * np.cos(dip0), -np.sin(dip0)), axis=1
    )
    slip0 = np.cos(rake0)[:, None] * strike_vector0 + np.sin(rake0)[:, None] * dip_vector0
    sign1 = np.sign(np.sum(-normal0 * -prepared.n1, axis=1))
    sign2 = np.sign(np.sum(slip0 * -prepared.n2, axis=1))
    sign1[sign1 == 0.0] = 1.0
    sign2[sign2 == 0.0] = 1.0
    normals1 = np.empty((n_events, draws, 3), dtype=np.float32)
    normals2 = np.empty((n_events, draws, 3), dtype=np.float32)
    uncertainty = np.full(n_events, uncertainty_deg, dtype=np.float64)
    for draw in range(draws):
        noise = rng.uniform(-uncertainty[:, None], uncertainty[:, None], size=(n_events, 3))
        strike = np.deg2rad(np.mod(mechanisms[:, 0] + noise[:, 0], 360.0))
        dip = np.deg2rad(np.clip(mechanisms[:, 1] + noise[:, 1], 0.0, 90.0))
        rake = np.deg2rad(np.mod(mechanisms[:, 2] + noise[:, 2] + 180.0, 360.0) - 180.0)
        normal = np.stack(
            (-np.sin(dip) * np.sin(strike), np.sin(dip) * np.cos(strike), -np.cos(dip)), axis=1
        )
        strike_vector = np.stack((np.cos(strike), np.sin(strike), np.zeros(n_events)), axis=1)
        dip_vector = np.stack(
            (np.sin(strike) * np.cos(dip), -np.cos(strike) * np.cos(dip), -np.sin(dip)), axis=1
        )
        slip = np.cos(rake)[:, None] * strike_vector + np.sin(rake)[:, None] * dip_vector
        normals1[:, draw, :] = sign1[:, None] * -normal
        normals2[:, draw, :] = sign2[:, None] * slip
    return (
        np.ascontiguousarray(normals1),
        np.ascontiguousarray(normals2),
        float(uncertainty_deg),
        uncertainty_mode,
    )


@dataclass(frozen=True)
class SLICResult:
    sigma1_azimuth_deg: float
    sigma1_plunge_deg: float
    sigma2_azimuth_deg: float
    sigma2_plunge_deg: float
    sigma3_azimuth_deg: float
    sigma3_plunge_deg: float
    phi: float
    objective: float
    friction_mu: float
    search_mode: str
    a1_deg: float
    a2_deg: float
    a3_deg: float
    coarse_a1_deg: float | None
    coarse_a2_deg: float | None
    coarse_a3_deg: float | None
    coarse_phi: float | None
    mechanism_uncertainty_deg: float
    uncertainty_mode: str
    perturbation_draws: int
    perturbation_seed: int
    basis: np.ndarray

    def to_dict(self) -> dict:
        values = {
            "sigma1_azimuth_deg": float(self.sigma1_azimuth_deg),
            "sigma1_plunge_deg": float(self.sigma1_plunge_deg),
            "sigma2_azimuth_deg": float(self.sigma2_azimuth_deg),
            "sigma2_plunge_deg": float(self.sigma2_plunge_deg),
            "sigma3_azimuth_deg": float(self.sigma3_azimuth_deg),
            "sigma3_plunge_deg": float(self.sigma3_plunge_deg),
            "Phi": float(self.phi),
            "objective": float(self.objective),
            "friction_mu": float(self.friction_mu),
            "search_mode": self.search_mode,
            "a1_deg": float(self.a1_deg),
            "a2_deg": float(self.a2_deg),
            "a3_deg": float(self.a3_deg),
            "coarse_a1_deg": None if self.coarse_a1_deg is None else float(self.coarse_a1_deg),
            "coarse_a2_deg": None if self.coarse_a2_deg is None else float(self.coarse_a2_deg),
            "coarse_a3_deg": None if self.coarse_a3_deg is None else float(self.coarse_a3_deg),
            "coarse_Phi": None if self.coarse_phi is None else float(self.coarse_phi),
            "mechanism_uncertainty_deg": float(self.mechanism_uncertainty_deg),
            "uncertainty_mode": self.uncertainty_mode,
            "perturbation_draws": int(self.perturbation_draws),
            "perturbation_seed": int(self.perturbation_seed),
            "algorithm_version": ALGORITHM_VERSION,
            "coordinate_system": COORDINATE_SYSTEM,
            "objective_definition": "sum(pi_i*log(w1*exp(g1)+w2*exp(g2)))/sum(pi_i); pi_i=event weight; g_j=mean perturbed unit shear dot observed slip",
        }
        return values


def _grid_dtype(config: SLICConfig):
    return np.float32 if config.grid_dtype == "float32" else np.float64


def full_search_grids(
    angle_step_deg: float, phi_step: float, dtype
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.unique(
            np.mod(np.arange(0.0, 360.0, angle_step_deg, dtype=np.float64).astype(dtype), 360.0)
        ),
        _clamped_grid(45.0, 45.0, angle_step_deg, 0.0, 90.0, dtype),
        np.unique(
            np.mod(np.arange(0.0, 180.0, angle_step_deg, dtype=np.float64).astype(dtype), 180.0)
        ),
        _clamped_grid(0.5, 0.5, phi_step, 0.0, 1.0, dtype),
    )


def _clamped_grid(
    center: float, half_width: float, step: float, minimum: float, maximum: float, dtype
) -> np.ndarray:
    start = max(minimum, float(center) - float(half_width))
    stop = min(maximum, float(center) + float(half_width))
    values = np.arange(start, stop + 1e-12, step, dtype=np.float64)
    values = np.concatenate((values, [start, stop, np.clip(center, minimum, maximum)]))
    return np.unique(np.clip(values, minimum, maximum).astype(dtype))


def _local_grid(center: float, half_width: float, step: float, dtype) -> np.ndarray:
    start = float(center) - float(half_width)
    stop = float(center) + float(half_width)
    values = np.arange(start, stop + 1e-12, step, dtype=np.float64)
    return np.unique(np.concatenate((values, [center, stop])).astype(dtype))


def _periodic_grid(
    center: float, half_width: float, step: float, period: float, dtype
) -> np.ndarray:
    values = _local_grid(center, half_width, step, np.float64)
    return np.unique(np.mod(np.mod(values, period).astype(dtype), period))


def refinement_grids(
    coarse: tuple[float, float, float, float], config: SLICConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a1, a2, a3, phi = coarse
    dtype = _grid_dtype(config)
    return (
        _periodic_grid(
            a1, config.refine_angle_half_width_deg, config.refine_angle_step_deg, 360.0, dtype
        ),
        _local_grid(a2, config.refine_angle_half_width_deg, config.refine_angle_step_deg, dtype),
        _periodic_grid(
            a3, config.refine_angle_half_width_deg, config.refine_angle_step_deg, 180.0, dtype
        ),
        _clamped_grid(phi, config.refine_phi_half_width, config.refine_phi_step, 0.0, 1.0, dtype),
    )


@njit(cache=True, inline="always")
def _plane_values(normal, sigma2, v11, v12, v13, v21, v22, v23, v31, v32, v33):
    p1 = normal[0] * v11 + normal[1] * v12 + normal[2] * v13
    p2 = normal[0] * v21 + normal[1] * v22 + normal[2] * v23
    p3 = normal[0] * v31 + normal[1] * v32 + normal[2] * v33
    sigma_n = p1 * p1 + sigma2 * p2 * p2 - p3 * p3
    q1 = (1.0 - sigma_n) * p1
    q2 = (sigma2 - sigma_n) * p2
    q3 = (-1.0 - sigma_n) * p3
    tau = math.sqrt(max(q1 * q1 + q2 * q2 + q3 * q3, 0.0))
    if tau <= 1e-14:
        return (sigma_n, tau, 0.0, 0.0, 0.0)
    return (
        sigma_n,
        tau,
        (q1 * v11 + q2 * v21 + q3 * v31) / tau,
        (q1 * v12 + q2 * v22 + q3 * v32) / tau,
        (q1 * v13 + q2 * v23 + q3 * v33) / tau,
    )


@njit(cache=True)
def _mean_objective(
    v11,
    v12,
    v13,
    v21,
    v22,
    v23,
    v31,
    v32,
    v33,
    phi_values,
    original_slip1,
    original_slip2,
    perturbed_normal1,
    perturbed_normal2,
    event_weights,
    weight_sum,
    friction_mu,
):
    draws = perturbed_normal1.shape[1]
    inverse_draws = 1.0 / draws
    best_value = -1e300
    best_phi = 0.0
    for phi_index in range(phi_values.size):
        sigma2 = 2.0 * phi_values[phi_index] - 1.0
        total = 0.0
        for event in range(event_weights.size):
            wins1 = 0.0
            mean_ux1 = mean_uy1 = mean_uz1 = 0.0
            mean_ux2 = mean_uy2 = mean_uz2 = 0.0
            for draw in range(draws):
                sigma1, tau1, ux1, uy1, uz1 = _plane_values(
                    perturbed_normal1[event, draw],
                    sigma2,
                    v11,
                    v12,
                    v13,
                    v21,
                    v22,
                    v23,
                    v31,
                    v32,
                    v33,
                )
                sigma2_n, tau2, ux2, uy2, uz2 = _plane_values(
                    perturbed_normal2[event, draw],
                    sigma2,
                    v11,
                    v12,
                    v13,
                    v21,
                    v22,
                    v23,
                    v31,
                    v32,
                    v33,
                )
                criticality1 = failure_criticality_numba(sigma1, tau1, friction_mu)
                criticality2 = failure_criticality_numba(sigma2_n, tau2, friction_mu)
                mean_ux1 += ux1
                mean_uy1 += uy1
                mean_uz1 += uz1
                mean_ux2 += ux2
                mean_uy2 += uy2
                mean_uz2 += uz2
                if abs(criticality1 - criticality2) <= 1e-12:
                    wins1 += 0.5
                elif criticality1 > criticality2:
                    wins1 += 1.0
            score1 = inverse_draws * (
                mean_ux1 * original_slip1[event, 0]
                + mean_uy1 * original_slip1[event, 1]
                + mean_uz1 * original_slip1[event, 2]
            )
            score2 = inverse_draws * (
                mean_ux2 * original_slip2[event, 0]
                + mean_uy2 * original_slip2[event, 1]
                + mean_uz2 * original_slip2[event, 2]
            )
            weight1 = wins1 * inverse_draws
            if weight1 <= 1e-15:
                log_mixture = score2
            elif weight1 >= 1.0 - 1e-15:
                log_mixture = score1
            else:
                term1 = math.log(weight1) + score1
                term2 = math.log(1.0 - weight1) + score2
                maximum = max(term1, term2)
                log_mixture = maximum + math.log(
                    math.exp(term1 - maximum) + math.exp(term2 - maximum)
                )
            total += log_mixture * event_weights[event]
        value = total / weight_sum
        if value > best_value:
            best_value = value
            best_phi = phi_values[phi_index]
    return (best_value, best_phi)


@njit(parallel=True, cache=True)
def _search_grid(
    a1_values,
    a2_values,
    a3_values,
    phi_values,
    cos_a1,
    sin_a1,
    cos_a2,
    sin_a2,
    cos_a3,
    sin_a3,
    original_slip1,
    original_slip2,
    perturbed_normal1,
    perturbed_normal2,
    event_weights,
    weight_sum,
    friction_mu,
):
    jobs = a1_values.size * a3_values.size
    values = np.full(jobs, -1e30, dtype=np.float64)
    parameters = np.zeros((jobs, 4), dtype=np.float32)
    for job in prange(jobs):
        index1 = job // a3_values.size
        index3 = job % a3_values.size
        local_best = -1e300
        for index2 in range(a2_values.size):
            c1, s1 = (cos_a1[index1], sin_a1[index1])
            c2, s2 = (cos_a2[index2], sin_a2[index2])
            c3, s3 = (cos_a3[index3], sin_a3[index3])
            value, phi = _mean_objective(
                c1 * c3 - s1 * c2 * s3,
                s1 * c3 + c1 * c2 * s3,
                s2 * s3,
                -c1 * s3 - s1 * c2 * c3,
                -s1 * s3 + c1 * c2 * c3,
                s2 * c3,
                s1 * s2,
                -c1 * s2,
                c2,
                phi_values,
                original_slip1,
                original_slip2,
                perturbed_normal1,
                perturbed_normal2,
                event_weights,
                weight_sum,
                friction_mu,
            )
            if value > local_best:
                local_best = value
                parameters[job, 0] = a1_values[index1]
                parameters[job, 1] = a2_values[index2]
                parameters[job, 2] = a3_values[index3]
                parameters[job, 3] = phi
        values[job] = local_best
    return (values, parameters)


def _observed_slips(prepared: PreparedCatalog) -> tuple[np.ndarray, np.ndarray]:
    slip1 = (
        prepared.cos_rake1[:, None] * prepared.strike1 + prepared.sin_rake1[:, None] * prepared.dip1
    )
    slip2 = (
        prepared.cos_rake2[:, None] * prepared.strike2 + prepared.sin_rake2[:, None] * prepared.dip2
    )
    return (np.ascontiguousarray(slip1), np.ascontiguousarray(slip2))


def _invert_grid(
    grids: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    prepared: PreparedCatalog,
    perturbations: tuple[np.ndarray, np.ndarray],
    friction_mu: float,
) -> tuple[float, float, float, float, float]:
    a1, a2, a3, phi = grids
    slip1, slip2 = _observed_slips(prepared)

    def trig(values):
        radians = np.asarray(values, dtype=np.float64) * np.pi / 180.0
        return (np.cos(radians), np.sin(radians))

    cos_a1, sin_a1 = trig(a1)
    cos_a2, sin_a2 = trig(a2)
    cos_a3, sin_a3 = trig(a3)
    values, parameters = _search_grid(
        a1,
        a2,
        a3,
        phi,
        cos_a1,
        sin_a1,
        cos_a2,
        sin_a2,
        cos_a3,
        sin_a3,
        slip1,
        slip2,
        perturbations[0],
        perturbations[1],
        prepared.weights,
        prepared.weight_sum,
        float(friction_mu),
    )
    index = int(np.argmax(values))
    return (
        float(parameters[index, 0]),
        float(parameters[index, 1]),
        float(parameters[index, 2]),
        float(parameters[index, 3]),
        float(values[index]),
    )


def invert(catalog: FocalMechanismCatalog, config: SLICConfig | None = None) -> SLICResult:
    config = (config or SLICConfig()).validate()
    if config.numba_threads > 0:
        set_num_threads(int(config.numba_threads))
    prepared = prepare_catalog(catalog, config)
    perturbation_seed = int(config.random_seed)
    normals1, normals2, uncertainty_deg, uncertainty_mode = make_perturbation_ensemble(
        prepared, config, seed=perturbation_seed
    )
    perturbations = (normals1, normals2)
    dtype = _grid_dtype(config)
    mode = str(config.search_mode).strip().lower()
    best_result = None
    best_coarse_result = None
    best_friction_mu = None
    for friction_mu in config.resolved_friction_mu_grid():
        coarse_result = None
        if mode == "one_stage":
            result = _invert_grid(
                full_search_grids(
                    config.one_stage_angle_step_deg, config.one_stage_phi_step, dtype
                ),
                prepared,
                perturbations,
                friction_mu,
            )
        else:
            coarse_result = _invert_grid(
                full_search_grids(config.coarse_angle_step_deg, config.coarse_phi_step, dtype),
                prepared,
                perturbations,
                friction_mu,
            )
            refined_result = _invert_grid(
                refinement_grids(coarse_result[:4], config), prepared, perturbations, friction_mu
            )
            result = refined_result if refined_result[4] > coarse_result[4] else coarse_result
        if best_result is None or result[4] > best_result[4]:
            best_result = result
            best_coarse_result = coarse_result
            best_friction_mu = float(friction_mu)
    if best_result is None or best_friction_mu is None:
        raise RuntimeError("No friction candidate was evaluated.")
    result = best_result
    coarse_result = best_coarse_result
    a1, a2, a3, phi_grid, objective = result
    raw_basis = euler_basis(a1, a2, a3)
    basis, phi = tensor_to_basis_phi(basis_phi_to_tensor(raw_basis, phi_grid))
    axes = [vector_to_azimuth_plunge(basis[:, index]) for index in range(3)]
    return SLICResult(
        sigma1_azimuth_deg=axes[0][0],
        sigma1_plunge_deg=axes[0][1],
        sigma2_azimuth_deg=axes[1][0],
        sigma2_plunge_deg=axes[1][1],
        sigma3_azimuth_deg=axes[2][0],
        sigma3_plunge_deg=axes[2][1],
        phi=phi,
        objective=objective,
        friction_mu=best_friction_mu,
        search_mode=mode,
        a1_deg=a1,
        a2_deg=a2,
        a3_deg=a3,
        coarse_a1_deg=None if coarse_result is None else coarse_result[0],
        coarse_a2_deg=None if coarse_result is None else coarse_result[1],
        coarse_a3_deg=None if coarse_result is None else coarse_result[2],
        coarse_phi=None if coarse_result is None else coarse_result[3],
        mechanism_uncertainty_deg=uncertainty_deg,
        uncertainty_mode=uncertainty_mode,
        perturbation_draws=int(config.perturbation_draws),
        perturbation_seed=perturbation_seed,
        basis=basis,
    )


@dataclass(frozen=True)
class BootstrapResult:
    original: SLICResult
    runs: pd.DataFrame
    summary: pd.DataFrame


def _bootstrap_indices(n_events: int, repeats: int, random_seed: int) -> list[np.ndarray]:
    output = []
    for run_id in range(1, repeats + 1):
        seed = np.random.SeedSequence([int(random_seed), 1, int(run_id)])
        rng = np.random.default_rng(seed)
        output.append(rng.integers(0, n_events, size=n_events))
    return output


def _bootstrap_worker(payload):
    run_id, catalog, indices, config = payload
    result = invert(catalog.subset(indices), config)
    return (run_id, int(np.unique(indices).size), result)


def _result_row(
    run_id: int, unique_events: int, n_events: int, result: SLICResult, original: SLICResult
) -> dict:
    row = {"run_id": int(run_id)}
    row.update(result.to_dict())
    row.update(
        bootstrap_unique_events=int(unique_events),
        bootstrap_unique_fraction=float(unique_events / n_events),
        rotation_from_original_deg=basis_rotation_angle_deg(result.basis, original.basis),
        sigma1_axis_angle_deg=axis_angle_deg(result.basis[:, 0], original.basis[:, 0]),
        sigma2_axis_angle_deg=axis_angle_deg(result.basis[:, 1], original.basis[:, 1]),
        sigma3_axis_angle_deg=axis_angle_deg(result.basis[:, 2], original.basis[:, 2]),
    )
    return row


def _summarize(runs: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "sigma1_azimuth_deg",
        "sigma1_plunge_deg",
        "sigma2_azimuth_deg",
        "sigma2_plunge_deg",
        "sigma3_azimuth_deg",
        "sigma3_plunge_deg",
        "Phi",
        "objective",
        "rotation_from_original_deg",
        "sigma1_axis_angle_deg",
        "sigma2_axis_angle_deg",
        "sigma3_axis_angle_deg",
    ]
    rows = []
    for column in columns:
        values = pd.to_numeric(runs[column], errors="coerce").dropna().to_numpy()
        if not values.size:
            continue
        rows.append(
            {
                "quantity": column,
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1) if values.size > 1 else 0.0),
                "median": float(np.median(values)),
                "q025": float(np.quantile(values, 0.025)),
                "q975": float(np.quantile(values, 0.975)),
                "n": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def bootstrap(
    catalog: FocalMechanismCatalog,
    config: SLICConfig | None = None,
    *,
    repeats: int = 100,
    workers: int = 1,
) -> BootstrapResult:
    config = (config or SLICConfig()).validate()
    repeats = int(repeats)
    workers = int(workers)
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    cpu_count = max(1, int(os.cpu_count() or 1))
    workers = min(workers, repeats)
    worker_config = config
    if workers > 1 and config.numba_threads == 0:
        worker_config = replace(config, numba_threads=max(1, cpu_count // workers))
    original = invert(catalog, worker_config)
    index_sets = _bootstrap_indices(catalog.size, repeats, config.random_seed)
    outputs: dict[int, tuple[int, SLICResult]] = {}
    if workers == 1:
        for run_id, indices in enumerate(index_sets, start=1):
            _, unique_events, result = _bootstrap_worker((run_id, catalog, indices, worker_config))
            outputs[run_id] = (unique_events, result)
    else:
        payloads = [
            (run_id, catalog, indices, worker_config)
            for run_id, indices in enumerate(index_sets, start=1)
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_bootstrap_worker, payload): payload[0] for payload in payloads
            }
            for future in as_completed(futures):
                run_id, unique_events, result = future.result()
                outputs[int(run_id)] = (int(unique_events), result)
    rows = [
        _result_row(run_id, outputs[run_id][0], catalog.size, outputs[run_id][1], original)
        for run_id in range(1, repeats + 1)
    ]
    runs = pd.DataFrame(rows)
    return BootstrapResult(original=original, runs=runs, summary=_summarize(runs))


def classify_zoback_1992(plunges_deg: Iterable[float]) -> tuple[int, str, str]:
    values = np.abs(np.asarray(list(plunges_deg), dtype=float))
    if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values > 90.0):
        code = "U"
    else:
        p1, p2, p3 = values
        if p1 >= 52.0 and p3 <= 35.0:
            code = "NF"
        elif 40.0 <= p1 < 52.0 and p3 <= 20.0:
            code = "NS"
        elif (
            p1 < 40.0 and p2 >= 45.0 and (p3 <= 20.0) or (p1 <= 20.0 and p2 >= 45.0 and (p3 < 40.0))
        ):
            code = "SS"
        elif p1 <= 20.0 and 40.0 <= p3 < 52.0:
            code = "TS"
        elif p1 <= 35.0 and p3 >= 52.0:
            code = "TF"
        else:
            code = "U"
    return (ZOBACK_INDEX[code], code, ZOBACK_LABEL_BY_CODE[code])


@dataclass(frozen=True)
class Projection:
    lon0: float
    lat0: float
    km_per_degree_lon: float

    @classmethod
    def from_bounds(
        cls, lon_min: float, lon_max: float, lat_min: float, lat_max: float
    ) -> "Projection":
        lat0 = 0.5 * (lat_min + lat_max)
        return cls(
            lon0=0.5 * (lon_min + lon_max),
            lat0=lat0,
            km_per_degree_lon=KM_PER_DEGREE_LAT * math.cos(math.radians(lat0)),
        )

    def xy(
        self, longitude: Iterable[float] | float, latitude: Iterable[float] | float
    ) -> tuple[np.ndarray, np.ndarray]:
        lon = np.asarray(longitude, dtype=float)
        lat = np.asarray(latitude, dtype=float)
        return ((lon - self.lon0) * self.km_per_degree_lon, (lat - self.lat0) * KM_PER_DEGREE_LAT)


@dataclass(frozen=True)
class Grid:
    longitudes: np.ndarray
    latitudes: np.ndarray
    depths_km: np.ndarray

    @property
    def spatial_cell_count(self) -> int:
        return int(len(self.longitudes) * len(self.latitudes) * len(self.depths_km))


@dataclass(frozen=True)
class LocalSelection:
    indices: np.ndarray
    weights: np.ndarray
    distances_km: np.ndarray
    nearest_radius_km: float
    depth_matched_event_count: int
    horizontal_azimuth_sector_count: int


def read_catalog(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    frame = pd.read_csv(path, skipinitialspace=True, low_memory=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    required = [
        "Latitude",
        "Longitude",
        "Depth",
        "strike1",
        "dip1",
        "rake1",
        "strike2",
        "dip2",
        "rake2",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"The catalogue is missing required columns: {', '.join(missing)}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required).copy()
    frame = frame[np.isfinite(frame[required].to_numpy(dtype=float)).all(axis=1)].copy()
    frame = frame[
        frame["dip1"].between(0.0, 90.0)
        & frame["dip2"].between(0.0, 90.0)
        & frame["Depth"].between(args.depth_min_km, args.depth_max_km)
        & frame["Longitude"].between(args.lon_min, args.lon_max)
        & frame["Latitude"].between(args.lat_min, args.lat_max)
    ].copy()
    frame = frame.reset_index(drop=True)
    if len(frame) < args.nearest_events:
        raise ValueError(
            f"Only {len(frame)} usable mechanisms remain, fewer than --nearest-events={args.nearest_events}."
        )
    return frame


def build_grid(args: argparse.Namespace) -> Grid:
    km_per_degree_lon = KM_PER_DEGREE_LAT * math.cos(
        math.radians(0.5 * (args.lat_min + args.lat_max))
    )
    lon_step = args.grid_spacing_km / km_per_degree_lon
    lat_step = args.grid_spacing_km / KM_PER_DEGREE_LAT
    longitudes = np.arange(args.lon_min, args.lon_max + 1e-12, lon_step)
    latitudes = np.arange(args.lat_min, args.lat_max + 1e-12, lat_step)
    depth_start = args.depth_min_km + args.layer_spacing_km
    depths = np.arange(depth_start, args.depth_max_km, args.layer_spacing_km)
    if not len(longitudes) or not len(latitudes) or (not len(depths)):
        raise ValueError("The selected bounds and grid spacing produce an empty spatial grid.")
    return Grid(longitudes=longitudes, latitudes=latitudes, depths_km=depths)


def nearest_local_events(
    point_xyz: np.ndarray, event_xyz: np.ndarray, args: argparse.Namespace
) -> tuple[LocalSelection | None, str]:
    distances = np.linalg.norm(event_xyz - point_xyz, axis=1)
    local_order = np.argpartition(distances, args.nearest_events - 1)[: args.nearest_events]
    local_order = local_order[np.argsort(distances[local_order])]
    nearest_indices = local_order
    local_distances = distances[local_order]
    radius = float(local_distances[-1])
    if radius > args.max_radius_km:
        return (None, "large_nearest_event_radius")
    indices = np.sort(nearest_indices)
    local_distances = distances[indices]
    local_xyz = event_xyz[indices]
    depth_half_width_km = (
        0.5 * args.layer_spacing_km
        if args.depth_support_half_width_km is None
        else args.depth_support_half_width_km
    )
    depth_matched = int(
        np.count_nonzero(np.abs(local_xyz[:, 2] - point_xyz[2]) <= depth_half_width_km)
    )
    if depth_matched < args.min_depth_events:
        return (None, "poor_depth_coverage")
    dx = local_xyz[:, 0] - point_xyz[0]
    dy = local_xyz[:, 1] - point_xyz[1]
    sectors = np.unique((dx >= 0.0).astype(int) * 2 + (dy >= 0.0).astype(int))
    if len(sectors) < args.min_azimuth_sectors:
        return (None, "poor_horizontal_coverage")
    weighting = args.event_weighting
    if weighting == "gaussian_distance":
        weights = (
            np.exp(-np.square(local_distances / radius))
            if radius > 0.0
            else np.ones(len(indices), dtype=float)
        )
    elif weighting == "equal":
        weights = np.ones(len(indices), dtype=float)
    else:
        raise ValueError(f"Unknown event weighting: {weighting}")
    return (
        LocalSelection(
            indices=indices,
            weights=weights,
            distances_km=local_distances,
            nearest_radius_km=radius,
            depth_matched_event_count=depth_matched,
            horizontal_azimuth_sector_count=int(len(sectors)),
        ),
        "ok",
    )


def focal_mechanism_orientation_matrix(
    strike_deg: float, dip_deg: float, rake_deg: float
) -> np.ndarray:
    strike = math.radians(float(strike_deg))
    dip = math.radians(float(dip_deg))
    rake = math.radians(float(rake_deg))
    along_strike = np.array([math.sin(strike), math.cos(strike), 0.0])
    down_dip = np.array(
        [math.cos(strike) * math.cos(dip), -math.sin(strike) * math.cos(dip), -math.sin(dip)]
    )
    fault_normal = np.cross(down_dip, along_strike)
    slip = math.cos(rake) * along_strike - math.sin(rake) * down_dip
    tension = (fault_normal + slip) / math.sqrt(2.0)
    pressure = (fault_normal - slip) / math.sqrt(2.0)
    null = np.cross(pressure, tension)
    frame = np.column_stack((tension, null, pressure))
    u, _, vh = np.linalg.svd(frame)
    frame = u @ vh
    if np.linalg.det(frame) < 0.0:
        frame[:, 1] *= -1.0
    return frame


def rotation_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = math.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Cannot form a quaternion from the focal-mechanism rotation.")
    return quaternion / norm


def kagan_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=float).T @ np.asarray(second, dtype=float)
    quaternion = rotation_quaternion_wxyz(relative)
    cosine_half_angle = float(np.clip(np.max(np.abs(quaternion)), 0.0, 1.0))
    return float(math.degrees(2.0 * math.acos(cosine_half_angle)))


def mechanism_diversity_metrics(subset: pd.DataFrame) -> dict[str, Any]:
    frames = [
        focal_mechanism_orientation_matrix(strike, dip, rake)
        for strike, dip, rake in subset[["strike1", "dip1", "rake1"]].itertuples(
            index=False, name=None
        )
    ]
    angles = np.asarray(
        [
            kagan_angle_deg(frames[first], frames[second])
            for first in range(len(frames) - 1)
            for second in range(first + 1, len(frames))
        ],
        dtype=float,
    )
    mean_angle = float(np.mean(angles)) if len(angles) else float("nan")
    median_angle = float(np.median(angles)) if len(angles) else float("nan")
    if not np.isfinite(mean_angle) or mean_angle < MECHANISM_DIVERSITY_POOR_MAX_DEG:
        quality_index = 0
    elif mean_angle < MECHANISM_DIVERSITY_GOOD_MIN_DEG:
        quality_index = 1
    else:
        quality_index = 2
    return {
        "mechanism_diversity_mean_kagan_deg": mean_angle,
        "mechanism_diversity_median_kagan_deg": median_angle,
        "mechanism_diversity_pair_count": int(len(angles)),
        "mechanism_diversity_quality_index": quality_index,
        "mechanism_diversity_quality": MECHANISM_DIVERSITY_LABELS[quality_index],
    }


def regime_from_principal_stress(plunges: Iterable[float], phi: float) -> tuple[int, str, float]:
    vertical_axis = int(np.nanargmax(np.asarray(list(plunges), dtype=float)))
    regime_index = vertical_axis
    aphi = regime_index + 0.5 + (-1) ** regime_index * (float(phi) - 0.5)
    return (regime_index, REGIME_LABELS[regime_index], float(aphi))


def bootstrap_metrics(
    catalog: Any, config: Any, repeats: int, workers: int
) -> tuple[Any, dict[str, float]]:
    result = bootstrap(catalog, config, repeats=repeats, workers=workers)
    summary = result.summary.set_index("quantity")

    def quantile(quantity: str, field: str) -> float:
        return float(summary.loc[quantity, field]) if quantity in summary.index else float("nan")

    return (
        result.original,
        {
            "bootstrap_repeats": int(repeats),
            "bootstrap_rotation_median_deg": quantile("rotation_from_original_deg", "median"),
            "bootstrap_rotation_q975_deg": quantile("rotation_from_original_deg", "q975"),
            "bootstrap_sigma1_axis_q975_deg": quantile("sigma1_axis_angle_deg", "q975"),
            "bootstrap_sigma3_axis_q975_deg": quantile("sigma3_axis_angle_deg", "q975"),
        },
    )


def invert_local_catalogue(
    config: Any, frame: pd.DataFrame, selection: LocalSelection, args: argparse.Namespace
) -> dict[str, Any]:
    subset = frame.iloc[selection.indices]
    catalog = FocalMechanismCatalog.from_nodal_planes(
        subset.strike1.to_numpy(),
        subset.dip1.to_numpy(),
        subset.rake1.to_numpy(),
        subset.strike2.to_numpy(),
        subset.dip2.to_numpy(),
        subset.rake2.to_numpy(),
        weights=selection.weights,
    )
    if args.bootstrap_repeats:
        result, uncertainty = bootstrap_metrics(
            catalog, config, args.bootstrap_repeats, args.bootstrap_workers
        )
    else:
        result = invert(catalog, config)
        uncertainty = {
            "bootstrap_repeats": 0,
            "bootstrap_rotation_median_deg": float("nan"),
            "bootstrap_rotation_q975_deg": float("nan"),
            "bootstrap_sigma1_axis_q975_deg": float("nan"),
            "bootstrap_sigma3_axis_q975_deg": float("nan"),
        }
    result_values = result.to_dict()
    result_values.update(algorithm_version=ALGORITHM_VERSION, coordinate_system=COORDINATE_SYSTEM)
    regime_index, regime, aphi = regime_from_principal_stress(
        [result.sigma1_plunge_deg, result.sigma2_plunge_deg, result.sigma3_plunge_deg], result.phi
    )
    stress_type_index, _, stress_type = classify_zoback_1992(
        [result.sigma1_plunge_deg, result.sigma2_plunge_deg, result.sigma3_plunge_deg]
    )
    payload: dict[str, Any] = {
        "regime_index": regime_index,
        "stress_regime": regime,
        "Aphi": aphi,
        "stress_type_index": stress_type_index,
        "stress_type_code": ZOBACK_CODE_BY_INDEX[stress_type_index],
        "stress_type": stress_type,
        "stress_classification": "Zoback (1992) principal-axis plunge criteria",
        "sigma1_azimuth_axial_deg": float(result.sigma1_azimuth_deg % 180.0),
    }
    payload.update(result_values)
    payload.update(uncertainty)
    payload.update(mechanism_diversity_metrics(subset))
    return payload


def local_catalogue_signature(selection: LocalSelection) -> str:
    order = np.argsort(selection.indices)
    indices = np.asarray(selection.indices[order], dtype="<i8")
    weights = np.asarray(np.round(selection.weights[order], 6), dtype="<f4")
    payload = indices.tobytes() + weights.tobytes()
    return hashlib.sha256(payload).hexdigest()


def make_cell_row(
    payload: dict[str, Any],
    longitude: float,
    latitude: float,
    depth_km: float,
    grid_x: int,
    grid_y: int,
    grid_z: int,
    selection: LocalSelection,
    args: argparse.Namespace,
    catalogue_signature: str,
    source_inversion_id: int,
    reused_local_catalogue: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_z": grid_z,
        "longitude": float(longitude),
        "latitude": float(latitude),
        "depth_km": float(depth_km),
        "local_event_count": int(len(selection.indices)),
        "nearest_event_radius_km": selection.nearest_radius_km,
        "depth_matched_event_count": selection.depth_matched_event_count,
        "horizontal_azimuth_sector_count": selection.horizontal_azimuth_sector_count,
        "event_weighting": args.event_weighting,
        "event_weight_sum": float(np.sum(selection.weights)),
        "weight_effective_event_count": float(
            np.sum(selection.weights) ** 2 / np.sum(selection.weights**2)
        ),
        "local_catalogue_signature": catalogue_signature,
        "source_inversion_id": int(source_inversion_id),
        "reused_local_catalogue": bool(reused_local_catalogue),
    }
    row.update(payload)
    return row


def restore_local_catalogue_cache(
    rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, tuple[int, dict[str, Any]]], int]:
    cache: dict[str, tuple[int, dict[str, Any]]] = {}
    largest_id = 0
    for row in rows:
        signature = row.get("local_catalogue_signature")
        source_id = row.get("source_inversion_id")
        if signature is None or pd.isna(signature) or source_id is None or pd.isna(source_id):
            continue
        source_id_int = int(source_id)
        payload = {key: value for key, value in row.items() if key not in CELL_CONTEXT_FIELDS}
        cache[str(signature)] = (source_id_int, payload)
        largest_id = max(largest_id, source_id_int)
    return (cache, largest_id)


def result_key(grid_x: int, grid_y: int, grid_z: int) -> tuple[int, int, int]:
    return (int(grid_x), int(grid_y), int(grid_z))


def result_key_from_row(row: dict[str, Any] | pd.Series) -> tuple[int, int, int]:
    return result_key(row["grid_x"], row["grid_y"], row["grid_z"])


def write_result_table(rows: list[dict[str, Any]], result_path: Path) -> None:
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(columns=RESULT_KEY_COLUMNS)
    table.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(result_path)


def load_checkpoint(
    result_path: Path, resume: bool
) -> tuple[list[dict[str, Any]], set[tuple[int, int, int]]]:
    if not result_path.exists():
        return ([], set())
    if not resume:
        return ([], set())
    existing = pd.read_csv(result_path)
    missing = [column for column in RESULT_KEY_COLUMNS if column not in existing.columns]
    if missing:
        raise ValueError(
            f"Cannot resume {result_path}: missing cell-key columns {', '.join(missing)}. Use a new output directory."
        )
    rows = existing.to_dict("records")
    keys = {result_key_from_row(row) for row in rows}
    print(f"Resuming from {len(rows):,} completed cells in {result_path.name}.")
    return (rows, keys)


def validate_result_event_count(results: pd.DataFrame, nearest_events: int, source: str) -> None:
    if "local_event_count" not in results.columns:
        raise ValueError(
            f"Cannot verify the nearest-event count in {source} because local_event_count is missing. Regenerate the inversion."
        )
    stored_event_counts = sorted(
        pd.to_numeric(results["local_event_count"], errors="coerce")
        .dropna()
        .unique()
        .astype(int)
        .tolist()
    )
    if stored_event_counts != [nearest_events]:
        raise ValueError(
            f"{source} contains local_event_count={stored_event_counts}, but --nearest-events={nearest_events}. Rerun the inversion into a new output directory, or pass the original value only to reproduce old figures."
        )


def solver_config(args: argparse.Namespace) -> SLICConfig:
    return SLICConfig(
        friction_mu=args.friction_mu,
        perturbation_draws=args.perturbation_draws,
        random_seed=args.random_seed,
        mechanism_uncertainty_deg=args.mechanism_uncertainty_deg,
        search_mode=args.search_mode,
        one_stage_angle_step_deg=args.one_stage_angle_step_deg,
        one_stage_phi_step=args.one_stage_phi_step,
        coarse_angle_step_deg=args.coarse_angle_step_deg,
        coarse_phi_step=args.coarse_phi_step,
        refine_angle_half_width_deg=args.refine_angle_half_width_deg,
        refine_phi_half_width=args.refine_phi_half_width,
        refine_angle_step_deg=args.refine_angle_step_deg,
        refine_phi_step=args.refine_phi_step,
        numba_threads=args.numba_threads,
    ).validate()


def requested_plot_depths(args):
    values = args.plot_depths_km
    try:
        depths = tuple(
            (float(item) for item in (values.split(",") if isinstance(values, str) else values))
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Plot depths must be a list of numbers.") from error
    if not depths or any((not math.isfinite(depth) or depth < 0 for depth in depths)):
        raise ValueError("Plot depths must be finite and nonnegative.")
    if len(set(depths)) != len(depths):
        raise ValueError("Plot depths must not contain duplicates.")
    return depths


def inversion_signature(frame: pd.DataFrame, grid: Grid, args: argparse.Namespace) -> str:
    controls = solver_config(args).to_dict()
    controls.pop("numba_threads")
    for name in (
        "lon_min",
        "lon_max",
        "lat_min",
        "lat_max",
        "depth_min_km",
        "depth_max_km",
        "grid_spacing_km",
        "layer_spacing_km",
        "nearest_events",
        "max_radius_km",
        "min_depth_events",
        "depth_support_half_width_km",
        "min_azimuth_sectors",
        "event_weighting",
        "bootstrap_repeats",
    ):
        controls[name] = getattr(args, name)
    payload = {
        "version": 2,
        "algorithm_version": ALGORITHM_VERSION,
        "solver_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "controls": controls,
        "longitudes": grid.longitudes.tolist(),
        "latitudes": grid.latitudes.tolist(),
        "depths_km": grid.depths_km.tolist(),
        "catalogue": frame.to_csv(index=False, lineterminator="\n"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def validate_checkpoint_signature(rows: list[dict[str, Any]], signature: str) -> None:
    if any((not isinstance(row.get("inversion_signature"), str) for row in rows)):
        raise ValueError(
            "Cannot safely resume legacy stress.csv without an inversion_signature. Existing results remain usable for plotting; rerun into a new output directory."
        )
    if any((row["inversion_signature"] != signature for row in rows)):
        raise ValueError(
            "Cannot resume: algorithm, catalogue, grid or inversion parameters changed. Use the original inputs or a new output directory."
        )


def run_inversion(args: argparse.Namespace) -> int:
    validate_arguments(args)
    frame = read_catalog(Path(args.catalog).resolve(), args)
    grid = build_grid(args)
    run_signature = inversion_signature(frame, grid, args)
    theoretical_cells = grid.spatial_cell_count
    print(
        f"{len(frame):,} mechanisms after QC; {len(grid.longitudes)} x {len(grid.latitudes)} x {len(grid.depths_km)} spatial nodes; {theoretical_cells:,} candidate cells."
    )
    if args.dry_run:
        print("All retained mechanisms form one static catalogue.")
        return 0
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "stress.csv"
    thread_count = args.numba_threads or numba_config.NUMBA_NUM_THREADS
    set_num_threads(thread_count)
    print(f"Parallel SLIC search: {get_num_threads()} CPU threads.", flush=True)
    rows, completed_keys = load_checkpoint(result_path, args.resume)
    if rows:
        validate_checkpoint_signature(rows, run_signature)
        validate_result_event_count(pd.DataFrame(rows), args.nearest_events, str(result_path))
        print("Resume verified the catalogue, grid and inversion parameters.")
    local_catalogue_cache, next_source_inversion_id = restore_local_catalogue_cache(rows)
    projection = Projection.from_bounds(args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    event_x, event_y = projection.xy(frame.Longitude.to_numpy(), frame.Latitude.to_numpy())
    event_xyz = np.column_stack([event_x, event_y, frame.Depth.to_numpy(dtype=float)])
    print("Selecting and counting nodes for inversion...", flush=True)
    tasks = []
    qualified_keys = set()
    for grid_z, depth in enumerate(grid.depths_km):
        for grid_y, latitude in enumerate(grid.latitudes):
            x_values, y_values = projection.xy(
                grid.longitudes, np.full(len(grid.longitudes), latitude)
            )
            for grid_x, (longitude, x, y) in enumerate(zip(grid.longitudes, x_values, y_values)):
                selection, _ = nearest_local_events(
                    np.array([x, y, depth], dtype=float), event_xyz, args
                )
                if selection is None:
                    continue
                key = result_key(grid_x, grid_y, grid_z)
                qualified_keys.add(key)
                if key not in completed_keys:
                    tasks.append((grid_x, grid_y, grid_z, longitude, latitude, depth, selection))
    if completed_keys - qualified_keys:
        raise ValueError(
            "Saved results contain nodes outside the current selection. "
            "Resume with the original parameters."
        )
    if not args.resume:
        write_result_table([], result_path)
    total_nodes = len(qualified_keys)

    def show_progress():
        percent = 100.0 * len(completed_keys) / total_nodes if total_nodes else 100.0
        print(f"Inversion progress: {len(completed_keys)}/{total_nodes} ({percent:.1f}%)", flush=True)

    show_progress()
    config: Any | None = None
    completed_this_run = 0
    unique_inversions_this_run = 0
    reused_cells_this_run = 0
    attempts = theoretical_cells
    for grid_x, grid_y, grid_z, longitude, latitude, depth, selection in tasks:
        key = result_key(grid_x, grid_y, grid_z)
        signature = local_catalogue_signature(selection)
        cached_solution = local_catalogue_cache.get(signature)
        if cached_solution is None:
            if config is None:
                print(
                    "Loading the SLIC core for the first coverage-qualified catalogue...",
                    flush=True,
                )
                config = solver_config(args)
            next_source_inversion_id += 1
            payload = invert_local_catalogue(config, frame, selection, args)
            source_inversion_id = next_source_inversion_id
            local_catalogue_cache[signature] = (source_inversion_id, payload)
            unique_inversions_this_run += 1
            reused_local_catalogue = False
        else:
            source_inversion_id, payload = cached_solution
            reused_local_catalogue = True
            reused_cells_this_run += 1
        row = make_cell_row(
            payload,
            float(longitude),
            float(latitude),
            float(depth),
            grid_x,
            grid_y,
            grid_z,
            selection,
            args,
            signature,
            source_inversion_id,
            reused_local_catalogue,
        )
        row["inversion_signature"] = run_signature
        rows.append(row)
        completed_keys.add(key)
        completed_this_run += 1
        if completed_this_run % args.checkpoint_every == 0:
            write_result_table(rows, result_path)
        if completed_this_run % args.progress_every == 0:
            show_progress()
    results = pd.DataFrame(rows)
    write_result_table(results.to_dict("records"), result_path)
    if completed_this_run % args.progress_every:
        show_progress()
    print(
        f"Accepted {completed_this_run:,} new nodes ({len(rows):,} total) from {attempts:,} cells scanned; ran {unique_inversions_this_run:,} unique SLIC catalogues and reused {reused_cells_this_run:,} solutions."
    )
    print(f"Results written to {output}")
    return 0


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SLIC-3D stress inversion", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--catalog", help="CSV with focal mechanisms; Date and Time columns are ignored."
    )
    parser.add_argument("--output", help="Directory for this version of stress.csv and figures.")
    parser.add_argument("--lon-min", type=float)
    parser.add_argument("--lon-max", type=float)
    parser.add_argument("--lat-min", type=float)
    parser.add_argument("--lat-max", type=float)
    parser.add_argument("--depth-min-km", type=float)
    parser.add_argument(
        "--depth-max-km",
        type=float,
        help="Inclusive catalogue depth limit; default interior nodes are 5,...,55 km.",
    )
    parser.add_argument(
        "--grid-spacing-km", type=float, help="Horizontal output sampling, not physical resolution."
    )
    parser.add_argument(
        "--layer-spacing-km",
        type=float,
        help="Vertical sampling interval; depth support is controlled separately.",
    )
    parser.add_argument(
        "--nearest-events", type=int, help="Number of nearest mechanisms; the manuscript uses 45."
    )
    parser.add_argument(
        "--event-weighting",
        choices=("gaussian_distance", "equal"),
        help="Manuscript distance weights exp[-(k_i/d)^2], or equal weights for sensitivity checks.",
    )
    parser.add_argument(
        "--max-radius-km",
        type=float,
        help="Reject cells whose farthest selected mechanism exceeds this radius.",
    )
    parser.add_argument(
        "--min-depth-events",
        type=int,
        help="Minimum selected events inside the depth-support window.",
    )
    parser.add_argument(
        "--depth-support-half-width-km",
        type=float,
        help="Optional half-width for the depth-support count; defaults to half the layer spacing.",
    )
    parser.add_argument(
        "--min-azimuth-sectors",
        type=int,
        choices=(1, 2, 3, 4),
        help="Minimum populated horizontal quadrants around a cell.",
    )
    parser.add_argument("--friction-mu", type=float)
    parser.add_argument(
        "--mechanism-uncertainty-deg",
        type=float,
        help="Optional common mechanism uncertainty passed to SLIC.",
    )
    parser.add_argument(
        "--perturbation-draws",
        type=int,
        help="SLIC focal-mechanism perturbation draws per inversion.",
    )
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--search-mode", choices=("two_stage", "one_stage"))
    parser.add_argument(
        "--one-stage-angle-step-deg",
        type=float,
        help="One-stage orientation-grid increment. Use only for screening runs.",
    )
    parser.add_argument(
        "--one-stage-phi-step",
        type=float,
        help="One-stage Phi-grid increment. Use only for screening runs.",
    )
    parser.add_argument(
        "--coarse-angle-step-deg",
        type=float,
        help="Coarse orientation-grid increment for final two-stage inversion.",
    )
    parser.add_argument(
        "--coarse-phi-step",
        type=float,
        help="Coarse Phi-grid increment for final two-stage inversion.",
    )
    parser.add_argument(
        "--refine-angle-half-width-deg",
        type=float,
        help="Half-width around the coarse orientation optimum.",
    )
    parser.add_argument(
        "--refine-phi-half-width", type=float, help="Half-width around the coarse Phi optimum."
    )
    parser.add_argument(
        "--refine-angle-step-deg",
        type=float,
        help="Fine orientation-grid increment for final two-stage inversion.",
    )
    parser.add_argument(
        "--refine-phi-step",
        type=float,
        help="Fine Phi-grid increment for final two-stage inversion.",
    )
    parser.add_argument("--numba-threads", type=int)
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        help="Optional nonparametric local bootstrap repeats. Expensive for full maps.",
    )
    parser.add_argument("--bootstrap-workers", type=int)
    parser.add_argument(
        "--plot-depths-km",
        help="Comma-separated exact depths; all requested inversion layers must exist.",
    )
    parser.add_argument("--figure-dpi", type=int)
    parser.add_argument("--figure-formats", help="Comma-separated publication exports: png,pdf.")
    parser.add_argument("--progress-every", type=int)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        help="Write an atomic, resumable checkpoint after this many new accepted nodes.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing output directory, skipping completed grid cells.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the catalogue and grid design without running SLIC.",
    )
    parser.add_argument("--mode", choices=("all", "invert", "plot"), default="all")
    parser.add_argument("--relief", help="Relief grid used by the stress maps.")
    parser.set_defaults(**get_settings())
    return parser


def figure_formats(args):
    values = args.figure_formats
    formats = [
        str(item).strip().lower()
        for item in (values.split(",") if isinstance(values, str) else values)
    ]
    if not formats or any((item not in {"png", "pdf"} for item in formats)):
        raise ValueError("--figure-formats supports only png,pdf.")
    return list(dict.fromkeys(formats))


def validate_arguments(args: argparse.Namespace) -> None:
    for name, value in vars(args).items():
        if isinstance(value, float) and (not math.isfinite(value)):
            raise ValueError(f"--{name.replace('_', '-')} must be finite.")
    if args.bootstrap_repeats < 0 or args.bootstrap_workers < 1:
        raise ValueError("Bootstrap repeats must be nonnegative and workers must be positive.")
    if args.random_seed < 0:
        raise ValueError("--random-seed must be nonnegative.")
    if not 0 <= args.numba_threads <= numba_config.NUMBA_NUM_THREADS:
        raise ValueError(f"--numba-threads must be between 0 and {numba_config.NUMBA_NUM_THREADS}.")
    if args.figure_dpi < 1:
        raise ValueError("--figure-dpi must be positive.")
    solver_config(args)
    requested_plot_depths(args)
    for name in (
        "grid_spacing_km",
        "layer_spacing_km",
        "max_radius_km",
        "one_stage_angle_step_deg",
        "one_stage_phi_step",
        "coarse_angle_step_deg",
        "coarse_phi_step",
        "refine_angle_step_deg",
        "refine_phi_step",
    ):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.nearest_events < 4:
        raise ValueError("--nearest-events must be at least 4 for a local stress inversion.")
    if not 0 <= args.min_depth_events <= args.nearest_events:
        raise ValueError("--min-depth-events must lie between 0 and --nearest-events.")
    if args.depth_support_half_width_km is not None and args.depth_support_half_width_km <= 0.0:
        raise ValueError("--depth-support-half-width-km must be positive when provided.")
    if not 1 <= args.min_azimuth_sectors <= 4:
        raise ValueError("--min-azimuth-sectors must lie between 1 and 4.")
    if args.event_weighting not in {"gaussian_distance", "equal"}:
        raise ValueError("--event-weighting must be gaussian_distance or equal.")
    if args.depth_max_km <= args.depth_min_km:
        raise ValueError("--depth-max-km must exceed --depth-min-km.")
    if args.lon_max <= args.lon_min or args.lat_max <= args.lat_min:
        raise ValueError("Geographic bounds must have a positive extent.")
    if args.perturbation_draws < 1:
        raise ValueError("--perturbation-draws must be at least 1.")
    if args.refine_angle_half_width_deg < 0.0 or args.refine_phi_half_width < 0.0:
        raise ValueError("Refinement half-widths cannot be negative.")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1.")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least 1.")
    figure_formats(args)
    args.output = Path(args.output).expanduser().resolve()
    args.catalog = Path(args.catalog).expanduser().resolve()
    args.relief = Path(args.relief).expanduser().resolve() if args.relief else None


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    validate_arguments(args)
    if args.dry_run or args.mode in {"all", "invert"}:
        run_inversion(args)
    if not args.dry_run and args.mode in {"all", "plot"}:
        import Plot_figures

        Plot_figures.draw(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
