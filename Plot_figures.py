from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Iterable

sys.dont_write_bytecode = True
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Config import get_settings
from matplotlib.colors import Normalize

KM_PER_DEGREE_LAT = 111.32
JET = plt.get_cmap("jet")
APHI_INTERVAL_LABELS = ("Normal faulting", "Strike-slip", "Reverse faulting")
FIGURE_FIELDS = (
    ("Aphi", "$A\\phi$", 0.0, 3.0, "Aphi"),
    ("nearest_event_radius_km", "Coverage radius (km)", 0.0, None, "coverage"),
    ("Phi", "$\\Phi$", 0.0, 1.0, "phi"),
    ("sigma1_plunge_deg", "σ1 plunge (°)", 0.0, 90.0, "sigma1_plunge"),
    ("sigma1_azimuth_axial_deg", "σ1 axial azimuth (°)", 0.0, 180.0, "sigma1_azimuth"),
    ("sigma2_azimuth_axial_deg", "σ2 axial azimuth (°)", 0.0, 180.0, "sigma2_azimuth"),
    ("sigma2_plunge_deg", "σ2 plunge (°)", 0.0, 90.0, "sigma2_plunge"),
    ("sigma3_azimuth_axial_deg", "σ3 axial azimuth (°)", 0.0, 180.0, "sigma3_azimuth"),
    ("sigma3_plunge_deg", "σ3 plunge (°)", 0.0, 90.0, "sigma3_plunge"),
)


def load_relief(path):
    with h5py.File(path, "r") as relief:
        return (
            np.asarray(relief["lon"][::4], dtype=float),
            np.asarray(relief["lat"][::4], dtype=float),
            np.asarray(relief["z"][::4, ::4], dtype=float),
        )


def configure_plots():
    font = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": font,
            "font.sans-serif": [font],
            "mathtext.fontset": "custom",
            "mathtext.rm": font,
            "mathtext.it": font,
            "mathtext.bf": font + ":bold",
            "mathtext.fallback": None,
            "pdf.fonttype": 42,
            "font.size": 8.3,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "image.cmap": "jet",
        }
    )


def draw_map_background(axis, relief):
    longitude, latitude, elevation = relief
    land = np.ma.masked_where(~np.isfinite(elevation) | (elevation <= 0), elevation)
    axis.set_facecolor("#F2F2F2")
    axis.pcolormesh(
        longitude,
        latitude,
        land,
        shading="auto",
        cmap="Greys",
        vmin=0,
        vmax=4000,
        alpha=0.9,
        edgecolors="none",
        rasterized=True,
        zorder=0,
    )
    axis.contour(
        longitude,
        latitude,
        elevation,
        levels=(0, 800, 2000, 3000),
        colors=("#607D8B", "#B0B0B0", "#8A8A8A", "#666666"),
        linewidths=(0.65, 0.35, 0.4, 0.45),
        alpha=0.85,
        zorder=1,
    )


def parse_depths(values):
    depths = [float(value) for value in (values.split(",") if isinstance(values, str) else values)]
    if not depths or any((not math.isfinite(value) or value < 0 for value in depths)):
        raise ValueError("Plot depths must be finite and nonnegative.")
    if len(set(depths)) != len(depths):
        raise ValueError("Plot depths must not contain duplicates.")
    return depths


def figure_formats(args):
    values = args.figure_formats
    formats = [
        str(value).strip().lower()
        for value in (values.split(",") if isinstance(values, str) else values)
    ]
    if not formats or any((value not in {"png", "pdf"} for value in formats)):
        raise ValueError("Figure formats must be png or pdf.")
    return list(dict.fromkeys(formats))


def marker_area(axis, args):
    latitude_step = args.grid_spacing_km / KM_PER_DEGREE_LAT
    longitude_step = latitude_step / math.cos(math.radians((args.lat_min + args.lat_max) / 2))
    bbox = axis.get_window_extent()
    width = bbox.width * longitude_step / (axis.get_xlim()[1] - axis.get_xlim()[0])
    height = bbox.height * latitude_step / (axis.get_ylim()[1] - axis.get_ylim()[0])
    return (1.04 * max(width, height) * 72 / axis.figure.dpi) ** 2


def depth_panel_layout(count: int) -> tuple[int, int]:
    columns = min(3, count)
    return (math.ceil(count / columns), columns)


def choose_exact(
    values: np.ndarray, targets: Iterable[float], tolerance: float = 1e-06
) -> list[float]:
    available = np.unique(np.asarray(values, dtype=float))
    selected: list[float] = []
    missing: list[float] = []
    for target in targets:
        matches = available[np.isclose(available, target, atol=tolerance, rtol=0.0)]
        if len(matches):
            selected.append(float(matches[0]))
        else:
            missing.append(float(target))
    if missing:
        missing_text = ", ".join((f"{value:g}" for value in missing))
        available_text = ", ".join((f"{value:g}" for value in available))
        raise ValueError(
            f"Requested depth layers are absent: {missing_text} km. Available layers: {available_text} km. Rerun the inversion on the requested depth grid; nearby layers will not be substituted."
        )
    return selected


def automatic_map_extent(
    longitudes: Iterable[float], latitudes: Iterable[float], args: argparse.Namespace
) -> tuple[float, float, float, float]:
    longitude = np.asarray(list(longitudes), dtype=float)
    latitude = np.asarray(list(latitudes), dtype=float)
    longitude = longitude[np.isfinite(longitude)]
    latitude = latitude[np.isfinite(latitude)]
    if longitude.size == 0 or latitude.size == 0:
        return (args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    padding_km = max(10.0, float(args.grid_spacing_km))
    mean_latitude = float(np.mean(latitude))
    latitude_padding = padding_km / KM_PER_DEGREE_LAT
    longitude_padding = latitude_padding / max(math.cos(math.radians(mean_latitude)), 1e-06)
    rounding = 10.0
    west = math.floor((float(np.min(longitude)) - longitude_padding) * rounding) / rounding
    east = math.ceil((float(np.max(longitude)) + longitude_padding) * rounding) / rounding
    south = math.floor((float(np.min(latitude)) - latitude_padding) * rounding) / rounding
    north = math.ceil((float(np.max(latitude)) + latitude_padding) * rounding) / rounding
    return (
        max(float(args.lon_min), west),
        min(float(args.lon_max), east),
        max(float(args.lat_min), south),
        min(float(args.lat_max), north),
    )


def style_map_axis(
    axis: Any,
    depth: float,
    args: argparse.Namespace,
    panel_label: str,
    map_extent: tuple[float, float, float, float] | None = None,
) -> None:
    west, east, south, north = map_extent or (
        args.lon_min,
        args.lon_max,
        args.lat_min,
        args.lat_max,
    )
    axis.set_xlim(west, east)
    axis.set_ylim(south, north)
    axis.set_aspect(1.0 / math.cos(math.radians(0.5 * (south + north))))
    axis.tick_params(direction="out", length=2.5, width=0.65, labelsize=7.4, pad=1.8)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#4A4A4A")
        spine.set_linewidth(0.55)
    axis.text(
        0.035,
        0.965,
        f"({panel_label})  {depth:g} km",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=8.3,
        color="#202020",
        zorder=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.4},
    )
    axis.label_outer()


def create_slice_figure(
    nrows: int, ncols: int, *, map_extent: tuple[float, float, float, float]
) -> tuple[Any, np.ndarray]:
    figure_width = 6.2
    left_inches, right_inches, top_inches = (0.52, 0.09, 0.08)
    bottom_inches = 0.82
    wspace = hspace = 0.045
    west, east, south, north = map_extent
    geographic_aspect = 1.0 / math.cos(math.radians(0.5 * (south + north)))
    target_axis_ratio = (north - south) * geographic_aspect / (east - west)
    usable_width_inches = figure_width - left_inches - right_inches
    axis_width_inches = usable_width_inches / (ncols + wspace * max(ncols - 1, 0))
    axis_height_inches = target_axis_ratio * axis_width_inches
    axes_height_inches = axis_height_inches * (nrows + hspace * max(nrows - 1, 0))
    figure_height = axes_height_inches + bottom_inches + top_inches
    figure_height = max(2.25, figure_height)
    left = left_inches / figure_width
    right = 1.0 - right_inches / figure_width
    bottom = bottom_inches / figure_height
    top = 1.0 - top_inches / figure_height
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figure_width, figure_height), squeeze=False, sharex=True, sharey=True
    )
    fig.subplots_adjust(
        left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace
    )
    xlabel_y = 0.5 / figure_height
    fig.supxlabel("Longitude (°E)", x=0.53, y=xlabel_y, fontsize=9.0)
    fig.supylabel("Latitude (°N)", x=0.015, y=0.5 * (bottom + top), fontsize=9.0)
    return (fig, axes)


def add_slice_colorbar(fig: Any, mappable: Any, label: str) -> Any:
    figure_height = float(fig.get_figheight())
    colorbar_axis = fig.add_axes([0.185, 0.18 / figure_height, 0.69, 0.075 / figure_height])
    colorbar = fig.colorbar(mappable, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(label, fontsize=8.2, labelpad=3)
    colorbar.outline.set_linewidth(0.55)
    colorbar.ax.tick_params(labelsize=7.2, length=2.4, width=0.6, pad=1.8)
    return colorbar


def add_aphi_colorbar(fig):
    height = float(fig.get_figheight())
    cax = fig.add_axes([0.185, 0.38 / height, 0.69, 0.075 / height])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, 3), cmap=JET),
        cax=cax,
        orientation="horizontal",
        ticks=(0, 1, 2, 3),
    )
    cb.ax.tick_params(labelsize=7.2, length=3, width=0.7, pad=2)
    cb.outline.set_linewidth(0.7)
    cb.ax.annotate(
        "$A\\phi$",
        xy=(0, 0.5),
        xycoords="axes fraction",
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8.2,
        math_fontfamily="dejavusans",
        annotation_clip=False,
    )
    for boundary in (1, 2):
        cb.ax.axvline(boundary, color="#303030", linewidth=0.6)
    for midpoint, name in zip((0.5, 1.5, 2.5), APHI_INTERVAL_LABELS):
        cb.ax.annotate(
            name,
            xy=(midpoint / 3, 0),
            xycoords="axes fraction",
            xytext=(0, -17.2),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7.2,
            annotation_clip=False,
        )


def prepare_results(results: pd.DataFrame) -> pd.DataFrame:
    required = {"longitude", "latitude", "depth_km", "nearest_event_radius_km", "Phi"}
    required.update(
        (f"sigma{i}_{quantity}_deg" for i in (1, 2, 3) for quantity in ("azimuth", "plunge"))
    )
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Missing stress columns: {missing}")
    if results.empty:
        raise ValueError("stress.csv contains no inversion nodes.")
    results = results.copy()
    values = results[sorted(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Plot fields in stress.csv must be finite.")
    if "Aphi" not in results:
        vertical = np.argmax(
            results[[f"sigma{i}_plunge_deg" for i in (1, 2, 3)]].to_numpy(), axis=1
        )
        results["Aphi"] = vertical + 0.5 + (-1.0) ** vertical * (results.Phi.to_numpy() - 0.5)
    if not np.isfinite(results.Aphi).all() or not results.Aphi.between(-1e-08, 3 + 1e-08).all():
        raise ValueError("Aphi must be finite and lie in [0, 3].")
    for axis in (1, 2, 3):
        results[f"sigma{axis}_azimuth_axial_deg"] = np.mod(
            results[f"sigma{axis}_azimuth_deg"].to_numpy(dtype=float), 180.0
        )
        if not results[f"sigma{axis}_plunge_deg"].between(-1e-08, 90.0 + 1e-08).all():
            raise ValueError(f"σ{axis} plunge must lie in [0, 90].")
    if not results.Phi.between(-1e-08, 1.0 + 1e-08).all():
        raise ValueError("Phi must lie in [0, 1].")
    return results


def save_depth_maps(
    results: pd.DataFrame, output: Path, args: argparse.Namespace, relief
) -> list[Path]:
    results = prepare_results(results)
    depths = choose_exact(results.depth_km.to_numpy(), parse_depths(args.plot_depths_km))
    selected = results[results.depth_km.isin(depths)]
    extent = automatic_map_extent(selected.longitude, selected.latitude, args)
    nrows, ncols = depth_panel_layout(len(depths))
    layers = [selected[np.isclose(selected.depth_km, depth)] for depth in depths]
    generated = []
    for field, label, lower, upper, filename in FIGURE_FIELDS:
        if upper is None:
            upper = max(float(args.max_radius_km), float(selected[field].max()))
        norm = Normalize(lower, upper)
        fig, axes = create_slice_figure(nrows, ncols, map_extent=extent)
        for index, (axis, depth) in enumerate(zip(axes.ravel(), depths)):
            draw_map_background(axis, relief)
            style_map_axis(axis, depth, args, chr(ord("a") + index), extent)
        for axis in axes.ravel()[len(depths) :]:
            axis.axis("off")
        fig.canvas.draw()
        scatter = None
        for axis, layer in zip(axes.ravel(), layers):
            scatter = axis.scatter(
                layer.longitude,
                layer.latitude,
                c=layer[field],
                cmap=JET,
                norm=norm,
                s=marker_area(axis, args),
                marker="s",
                linewidths=0,
                edgecolors="none",
                antialiaseds=False,
                rasterized=True,
                zorder=2,
            )
            scatter.set_gid(field)
        if field == "Aphi":
            add_aphi_colorbar(fig)
        else:
            colorbar = add_slice_colorbar(fig, scatter, label)
            if upper == 180.0:
                colorbar.set_ticks(np.arange(0, 181, 30))
            elif upper == 90.0:
                colorbar.set_ticks(np.arange(0, 91, 15))
            elif field == "Phi":
                colorbar.set_ticks(np.linspace(0, 1, 6))
        generated.extend(save_publication_figure(fig, output / filename, args))
        print(f"Saved {filename}", flush=True)
    return generated


def save_publication_figure(fig: Any, base_path: Path, args: argparse.Namespace) -> list[Path]:
    generated: list[Path] = []
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for extension in figure_formats(args):
        path = base_path.with_suffix(f".{extension}")
        fig.savefig(path, dpi=args.figure_dpi, bbox_inches="tight", pad_inches=0.04)
        generated.append(path)
    plt.close(fig)
    return generated


def validate_arguments(args):
    parse_depths(args.plot_depths_km)
    figure_formats(args)
    if args.figure_dpi < 1 or args.grid_spacing_km <= 0 or args.max_radius_km <= 0:
        raise ValueError("Figure dpi, grid spacing and maximum radius must be positive.")
    if args.lon_max <= args.lon_min or args.lat_max <= args.lat_min:
        raise ValueError("Geographic bounds must have a positive extent.")
    if not args.relief or not Path(args.relief).is_file():
        raise FileNotFoundError(f"Relief grid not found: {args.relief}")


def draw(args):
    validate_arguments(args)
    output = Path(args.output)
    results = pd.read_csv(output / "stress.csv")
    relief = load_relief(args.relief)
    configure_plots()
    generated = save_depth_maps(results, output, args, relief)
    print(f"Saved {len(generated)} figure files to {output}")
    return generated


def parser():
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Draw SLIC-3D stress maps")
    parser.add_argument("--output", default=settings["output"])
    parser.add_argument("--relief", default=settings["relief"])
    for name in ("lon_min", "lon_max", "lat_min", "lat_max", "grid_spacing_km", "max_radius_km"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=settings[name])
    parser.add_argument("--plot-depths-km", default=settings["plot_depths_km"])
    parser.add_argument("--figure-dpi", type=int, default=settings["figure_dpi"])
    parser.add_argument("--figure-formats", default=settings["figure_formats"])
    return parser


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    draw(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
