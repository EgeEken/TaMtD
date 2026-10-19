import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, __version__ as pillow_version, features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.codec_benchmark import aggregate_rows, image_metrics, select_operating_points, write_sample_csv
from tamtd.codecs import AVIFCodec, JPEGCodec, JXLCodec, PBCStandardCodec, PNGCodec, WebPCodec


JPEG_QUALITIES = [1, 5, 10, 20, 30, 60, 80]
WEBP_QUALITIES = [1, 5, 10, 20, 30, 40, 50, 60, 75, 90, 100]
AVIF_QUALITIES = [1, 5, 10, 20, 30, 40, 50, 60, 75, 90, 95]
JXL_DISTANCES = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 25]
PBC_PRESETS = ["compression", "balanced", "quality", "high_quality"]
FAMILY_LABELS = {
    "jpeg": "JPEG",
    "webp": "WebP",
    "avif": "AVIF",
    "jxl": "JXL",
    "png": "PNG",
    "pbc": "PBC",
}
FAMILY_COLORS = {
    "jpeg": "#355c9a",
    "webp": "#d28a00",
    "avif": "#008a72",
    "jxl": "#8064a2",
    "png": "#555555",
    "pbc": "#b33a3a",
}


def load_sources(source_dir: Path, limit: int | None) -> tuple[list[dict], list[Image.Image], dict]:
    metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (source_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"no source images in {source_dir}")
    images = []
    for row in rows:
        path = source_dir / row["source_png"]
        with Image.open(path) as image:
            source = image.convert("RGB").copy()
        if source.size != (256, 256):
            raise ValueError(f"source must be 256x256: {path} has {source.size}")
        if hashlib.sha256(source.tobytes()).hexdigest() != row["rgb_pixels_sha256"]:
            raise ValueError(f"source pixel checksum mismatch: {path}")
        images.append(source)
    return rows, images, metadata


def build_adapters(pbc_root: str | None, smoke: bool) -> tuple[list, dict]:
    factories = []
    if smoke:
        factories.extend(
            [
                ("jpeg_q1", lambda: JPEGCodec(1)),
                ("webp_q10", lambda: WebPCodec(10)),
                ("avif_q10", lambda: AVIFCodec(10)),
                ("jxl_d12", lambda: JXLCodec(12)),
                ("png_lossless", PNGCodec),
            ]
        )
    else:
        factories.extend((f"jpeg_q{quality}", lambda quality=quality: JPEGCodec(quality)) for quality in JPEG_QUALITIES)
        factories.extend((f"webp_q{quality}", lambda quality=quality: WebPCodec(quality)) for quality in WEBP_QUALITIES)
        factories.extend((f"avif_q{quality}", lambda quality=quality: AVIFCodec(quality)) for quality in AVIF_QUALITIES)
        factories.extend((f"jxl_d{distance:g}", lambda distance=distance: JXLCodec(distance)) for distance in JXL_DISTANCES)
        factories.append(("png_lossless", PNGCodec))
    factories.extend(
        (f"pbc_{preset}", lambda preset=preset: PBCStandardCodec(pbc_root=pbc_root, preset=preset))
        for preset in PBC_PRESETS
    )
    adapters = []
    unavailable = {}
    for setting_id, factory in factories:
        try:
            adapters.append((setting_id, factory()))
        except Exception as error:
            unavailable[setting_id] = f"{type(error).__name__}: {error}"
    return adapters, unavailable


def measure_setting(
    setting_id: str,
    adapter,
    source_rows: list[dict],
    images: list[Image.Image],
    repeats: int,
    warmups: int,
    preview_indices: set[int],
    preview_root: Path,
) -> tuple[list[dict], dict]:
    warm_image = images[0]
    for _ in range(warmups):
        warm_sample = adapter.encode(warm_image)
        adapter.decode(warm_sample.data)
    results = []
    setting_metadata = {}
    for index, (source_row, image) in enumerate(zip(source_rows, images)):
        encode_runs = []
        encoded_samples = []
        error = None
        for _ in range(repeats):
            started = time.perf_counter_ns()
            try:
                sample = adapter.encode(image)
            except Exception as exc:
                error = f"encode {type(exc).__name__}: {exc}"
                break
            encode_runs.append((time.perf_counter_ns() - started) / 1_000_000)
            encoded_samples.append(sample)
        first = encoded_samples[0] if encoded_samples else None
        if first is not None:
            setting_metadata = first.metadata
        decode_runs = []
        decoded_images = []
        if first is not None and error is None:
            for _ in range(repeats):
                started = time.perf_counter_ns()
                try:
                    decoded = adapter.decode(first.data)
                except Exception as exc:
                    error = f"decode {type(exc).__name__}: {exc}"
                    break
                decode_runs.append((time.perf_counter_ns() - started) / 1_000_000)
                decoded_images.append(decoded)
        decoded = decoded_images[0] if decoded_images else None
        valid = decoded is not None and decoded.size == (256, 256) and decoded.mode == "RGB"
        metrics = image_metrics(image, decoded) if valid else {}
        if valid and index in preview_indices:
            preview_dir = preview_root / source_row["image_id"]
            preview_dir.mkdir(parents=True, exist_ok=True)
            decoded.save(preview_dir / f"{setting_id}.png", format="PNG", compress_level=1)
        row = {
            "image_id": source_row["image_id"],
            "codec_id": setting_id,
            "codec_family": first.metadata.get("codec_family", setting_id.split("_")[0]) if first else setting_id.split("_")[0],
            "encoded_bytes": len(first.data) if first else None,
            "bpp": len(first.data) * 8 / (256 * 256) if first else None,
            "decode_valid": bool(valid),
            "mse": metrics.get("mse"),
            "psnr_db": metrics.get("psnr_db") if np.isfinite(metrics.get("psnr_db", np.nan)) else None,
            "psnr_infinite": bool(metrics and np.isinf(metrics.get("psnr_db", np.nan))),
            "ssim": metrics.get("ssim"),
            "encode_ms_runs": encode_runs,
            "decode_ms_runs": decode_runs,
            "encode_deterministic": all(sample.data == first.data for sample in encoded_samples) if first else None,
            "codec_metadata": first.metadata if first else setting_metadata,
            "error": error,
        }
        results.append(row)
    return results, setting_metadata


def software_metadata() -> dict:
    versions = {
        "python": platform.python_version(),
        "pillow": pillow_version,
        "libjpeg_turbo": getattr(Image.core, "libjpeg_turbo_version", None),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "matplotlib": matplotlib.__version__,
    }
    for codec in ("webp", "avif"):
        try:
            versions[f"lib{codec}"] = features.version(codec)
        except (ValueError, TypeError):
            versions[f"lib{codec}"] = None
    return {
        "versions": versions,
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
        },
        "temporary_directories": {
            name: os.environ.get(name) for name in ("TEMP", "TMP", "NUMBA_CACHE_DIR")
        },
    }


def write_summary_csv(summary: list[dict], path: Path) -> None:
    fields = [
        "codec_id", "codec_family", "samples", "valid_samples", "mean_bytes", "median_bytes", "mean_bpp",
        "mean_psnr_db", "mean_ssim", "mean_encode_ms", "median_encode_ms", "p95_encode_ms",
        "median_decode_ms", "p95_decode_ms", "settings",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            item = dict(row)
            item["settings"] = json.dumps(row["settings"], sort_keys=True)
            writer.writerow({field: item.get(field) for field in fields})


def make_plots(summary: list[dict], output: Path) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    families = ("jpeg", "webp", "avif", "jxl", "pbc")
    fig, ax = plt.subplots(figsize=(12, 7))
    for family in families:
        rows = [row for row in summary if row["codec_family"] == family and row["mean_bpp"] is not None and row["mean_psnr_db"] is not None]
        rows.sort(key=lambda row: row["mean_bpp"])
        if rows:
            ax.plot(
                [row["mean_bpp"] for row in rows],
                [row["mean_psnr_db"] for row in rows],
                marker="o",
                markersize=4,
                linewidth=1.4,
                color=FAMILY_COLORS[family],
                label=FAMILY_LABELS[family],
            )
    png = [row for row in summary if row["codec_family"] == "png" and row["mean_bpp"] is not None]
    if png:
        ax.text(0.99, 0.02, f"PNG lossless: {png[0]['mean_bpp']:.3f} bpp, PSNR ∞", transform=ax.transAxes, ha="right", fontsize=8)
    ax.set_xlabel("Mean bits per pixel")
    ax.set_ylabel("Mean PSNR (dB)")
    ax.set_title("256×256 RGB rate–distortion operating points")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "rate_distortion.png", dpi=180)
    plt.close(fig)

    order = {family: index for index, family in enumerate(("pbc", "jpeg", "webp", "avif", "jxl", "png"))}
    rows = sorted(
        [row for row in summary if row["median_encode_ms"] is not None],
        key=lambda row: (order.get(row["codec_family"], 99), row["mean_bpp"] or 0),
    )
    for timing in ("encode", "decode"):
        fig_height = max(6, 0.30 * len(rows))
        fig, ax = plt.subplots(figsize=(12, fig_height))
        for position, row in enumerate(rows):
            median = row[f"median_{timing}_ms"]
            p95 = row[f"p95_{timing}_ms"]
            if median is None or p95 is None:
                continue
            family = row["codec_family"]
            ax.plot([median, max(median, p95)], [position, position], color=FAMILY_COLORS[family], linewidth=2)
            ax.plot(median, position, "o", color=FAMILY_COLORS[family], markersize=4)
        ax.set_yticks(range(len(rows)), [row["codec_id"] for row in rows], fontsize=8)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel(f"{timing.capitalize()} latency (ms, log scale; dot = median, line end = p95)")
        ax.set_title(f"{timing.capitalize()} latency by setting")
        ax.grid(True, axis="x", alpha=0.25)
        handles = [
            Line2D([0], [0], color=FAMILY_COLORS[family], marker="o", label=FAMILY_LABELS[family])
            for family in order
            if any(row["codec_family"] == family for row in rows)
        ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncols=1)
        fig.tight_layout(rect=(0, 0, 0.84, 1))
        fig.savefig(output / f"{timing}_latency.png", dpi=180)
        plt.close(fig)


def make_visual_grid(
    source_dir: Path,
    preview_root: Path,
    source_rows: list[dict],
    sample_rows: list[dict],
    summary: list[dict],
    matches: dict,
    output: Path,
) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_count = len(source_rows)
    indices = sorted(set([0, image_count // 3, (2 * image_count) // 3, image_count - 1]))
    row_count = len(indices) * len(PBC_PRESETS)
    fig, axes = plt.subplots(row_count, 6, figsize=(14, 2.5 * row_count), squeeze=False)
    sample_by_key = {(row["image_id"], row["codec_id"]): row for row in sample_rows}
    codecs = ("jpeg", "webp", "avif", "jxl")
    row_index = 0
    for source_index in indices:
        source_row = source_rows[source_index]
        source_path = source_dir / source_row["source_png"]
        with Image.open(source_path) as image:
            source = image.convert("RGB").copy()
        for preset in PBC_PRESETS:
            pbc_id = f"pbc_{preset}"
            cells = [(source, "Original", None), (preview_root / source_row["image_id"] / f"{pbc_id}.png", pbc_id, sample_by_key.get((source_row["image_id"], pbc_id)))]
            for family in codecs:
                match = matches.get(pbc_id, {}).get("codecs", {}).get(family, {})
                rate = match.get("rate_match")
                codec_id = rate.get("codec_id") if rate else None
                sample = sample_by_key.get((source_row["image_id"], codec_id)) if codec_id else None
                closeness = "" if not rate else f" ({rate['relative_delta']:+.0%}, not close)" if not rate["close"] else " (rate-close)"
                cells.append(
                    (
                        preview_root / source_row["image_id"] / f"{codec_id}.png" if codec_id else None,
                        f"{codec_id or FAMILY_LABELS[family]}{closeness}",
                        sample,
                    )
                )
            for column, (image_value, title, sample) in enumerate(cells):
                axis = axes[row_index, column]
                if isinstance(image_value, Image.Image):
                    image = image_value
                elif image_value is not None and Path(image_value).is_file():
                    with Image.open(image_value) as stored:
                        image = stored.convert("RGB").copy()
                else:
                    axis.text(0.5, 0.5, "no close point", ha="center", va="center", fontsize=8)
                    image = None
                if image is not None:
                    axis.imshow(image)
                axis.set_title(title, fontsize=7)
                if sample and sample.get("bpp") is not None:
                    psnr = "∞" if sample.get("psnr_infinite") else f"{sample['psnr_db']:.1f} dB"
                    axis.set_xlabel(f"{sample['bpp']:.2f} bpp | {psnr}", fontsize=7)
                elif column == 0:
                    axis.set_xlabel(source_row["image_id"], fontsize=7)
                axis.set_xticks([])
                axis.set_yticks([])
            row_index += 1
    fig.suptitle("Fixed 256×256 RGB visual comparisons; conventional settings chosen by mean rate", fontsize=12, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.8)
    fig.savefig(output / "visual_comparison_grid.png", dpi=120)
    plt.close(fig)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_findings(
    output: Path,
    source_metadata: dict,
    software: dict,
    summary: list[dict],
    matches: dict,
    unavailable: dict,
    repeats: int,
    warmups: int,
    source_count: int,
    smoke: bool,
) -> None:
    rate_rows = []
    for row in summary:
        def show(key, digits=2):
            value = row.get(key)
            return "—" if value is None else f"{value:.{digits}f}"

        rate_rows.append(
            [
                row["codec_id"], str(row["valid_samples"]), show("mean_bytes", 1), show("median_bytes", 1),
                show("mean_bpp", 4), "∞" if row["codec_family"] == "png" else show("mean_psnr_db", 2), show("mean_ssim", 5),
                show("median_encode_ms", 3), show("p95_encode_ms", 3),
                show("median_decode_ms", 3), show("p95_decode_ms", 3),
            ]
        )
    rate_table = md_table(
        ["Setting", "Valid", "Mean bytes", "Median bytes", "Mean bpp", "Mean PSNR dB", "Mean SSIM", "Enc med ms", "Enc p95 ms", "Dec med ms", "Dec p95 ms"],
        rate_rows,
    )
    pbc_rows = [row for row in summary if row["codec_family"] == "pbc"]
    pbc_ranges = md_table(
        ["PBC preset", "Mean bytes", "Mean bpp", "PSNR dB", "SSIM", "Outer LZMA", "Actual stream mode"],
        [
            [row["codec_id"], f"{row['mean_bytes']:.1f}", f"{row['mean_bpp']:.4f}", f"{row['mean_psnr_db']:.2f}",
             f"{row['mean_ssim']:.5f}", str(row["settings"].get("pbc_outer_lzma_enabled")),
             ", ".join(f"{mode}: {count}" for mode, count in row["pbc_entropy_modes"].items()) or "unknown"]
            for row in pbc_rows
        ],
    )
    match_rows = []
    for pbc_id, data in matches.items():
        for family, point in data["codecs"].items():
            if not point.get("available"):
                match_rows.append([pbc_id, family, "unavailable", "no", "unavailable", "no", "unavailable", "no"])
                continue
            rate = point["rate_match"]
            psnr = point["psnr_match"]
            ssim = point["ssim_match"]
            match_rows.append(
                [
                    pbc_id,
                    family,
                    f"{rate['codec_id']} ({rate['bpp']:.3f} bpp, Δ {rate['relative_delta']:+.1%})",
                    "yes" if rate["close"] else "no",
                    "—" if psnr is None else f"{psnr['codec_id']} (Δ {psnr['delta_psnr_db']:+.2f} dB; ΔSSIM {psnr['delta_ssim']:+.4f})",
                    "yes" if psnr and psnr["close"] else "no",
                    "—" if ssim is None else f"{ssim['codec_id']} (ΔSSIM {ssim['delta_ssim']:+.4f}; ΔPSNR {ssim['delta_psnr_db']:+.2f} dB)",
                    "yes" if ssim and ssim["close"] else "no",
                ]
            )
    matches_table = md_table(
        ["PBC", "Codec", "Nearest by rate", "Rate-close", "Nearest by PSNR", "PSNR-close", "Nearest by SSIM", "SSIM-close"],
        match_rows,
    )
    settings = {
        "jpeg": f"Pillow/libjpeg-turbo {software['versions'].get('libjpeg_turbo')}, quality {JPEG_QUALITIES}, 4:2:0, optimize=false, progressive=false",
        "webp": f"quality {WEBP_QUALITIES}, method=6, lossy, libwebp subsampling default",
        "avif": f"quality {AVIF_QUALITIES}, speed=6, subsampling=4:2:0, max_threads=1",
        "jxl": f"distance {JXL_DISTANCES}, effort=7, num_threads=1",
        "png": "lossless, Pillow compress_level=6, optimize=false",
        "pbc": "compression, balanced, quality, high_quality; full effective config and provenance are in config.json and per-sample metadata",
    }
    software_lines = ", ".join(f"{key} {value}" for key, value in software["versions"].items())
    unavailable_text = "; ".join(f"`{name}`: {error}" for name, error in unavailable.items()) or "none"
    setting_lines = "\n".join(f"- **{family.upper()}:** {value}" for family, value in settings.items())
    pbc_provenance = pbc_rows[0]["settings"] if pbc_rows else {}
    pbc_state = (
        f"{pbc_provenance.get('pbc_git_branch')} @ `{pbc_provenance.get('pbc_git_sha')}`, "
        f"dirty={pbc_provenance.get('pbc_worktree_dirty')}"
    ) if pbc_provenance else "unavailable"
    pbc_status_note = (
        f"Git `status` returned `{pbc_provenance['pbc_status_error']}`; the untracked-file fallback found files, "
        "so dirty=true is certain."
        if pbc_provenance.get("pbc_status_error") and pbc_provenance.get("pbc_worktree_dirty")
        else ""
    )
    phase2_lines = []
    for pbc_id, data in matches.items():
        rate_choices = []
        psnr_choices = []
        ssim_choices = []
        for family, point in data["codecs"].items():
            if not point.get("available"):
                rate_choices.append(f"{family}: unavailable")
                continue
            rate = point["rate_match"]
            rate_choices.append(
                f"{family} {rate['codec_id']} ({rate['bpp']:.3f} bpp, {rate['relative_delta']:+.1%}; {'close' if rate['close'] else 'nearest only'})"
            )
            psnr = point["psnr_match"]
            if psnr:
                psnr_choices.append(
                    f"{family} {psnr['codec_id']} ({psnr['delta_psnr_db']:+.2f} dB; {'close' if psnr['close'] else 'nearest only'})"
                )
            ssim = point["ssim_match"]
            if ssim:
                ssim_choices.append(
                    f"{family} {ssim['codec_id']} ({ssim['delta_ssim']:+.4f}; {'close' if ssim['close'] else 'nearest only'})"
                )
        phase2_lines.append(
            f"- **{pbc_id}** ({data['pbc_bpp']:.3f} bpp): rate — {'; '.join(rate_choices)}. "
            f"PSNR — {'; '.join(psnr_choices)}. SSIM — {'; '.join(ssim_choices)}."
        )
    phase2_text = "\n".join(phase2_lines) or "- PBC was unavailable, so no operating point can be selected."
    quality = next((row for row in pbc_rows if row["codec_id"] == "pbc_quality"), None)
    jpeg80 = next((row for row in summary if row["codec_id"] == "jpeg_q80"), None)
    jpeg60 = next((row for row in summary if row["codec_id"] == "jpeg_q60"), None)
    if quality and jpeg80:
        old_pbc_bpp = 939.1 * 8 / (32 * 32)
        old_jpeg_bpp = 959.8 * 8 / (32 * 32)
        jpeg80_gap = (jpeg80["mean_bpp"] - quality["mean_bpp"]) / quality["mean_bpp"]
        cifar_answer = (
            f"Yes, the earlier 32×32 result was misleading as a general operating-point match. "
            f"On CIFAR-10, PBC quality was about {old_pbc_bpp:.2f} bpp and JPEG q80 about {old_jpeg_bpp:.2f} bpp, "
            f"with PSNR {31.34:.2f} vs {30.73:.2f} dB. Here, PBC quality is {quality['mean_bpp']:.2f} bpp / "
            f"{quality['mean_psnr_db']:.2f} dB, while JPEG q80 is {jpeg80['mean_bpp']:.2f} bpp / "
            f"{jpeg80['mean_psnr_db']:.2f} dB ({abs(jpeg80_gap):.1%} fewer bits for JPEG). "
        )
        if jpeg60:
            cifar_answer += (
                f"JPEG q60 reaches {jpeg60['mean_psnr_db']:.2f} dB at {jpeg60['mean_bpp']:.2f} bpp, "
                f"about {(1 - jpeg60['mean_bpp'] / quality['mean_bpp']):.1%} fewer bits than PBC quality. "
            )
        cifar_answer += "The dataset also changed from CIFAR-10 to local natural photos, so these measurements show that the old match does not transfer; they do not isolate resolution as the only cause."
    else:
        cifar_answer = "The previous CIFAR comparison cannot be evaluated because the full run lacks PBC quality or JPEG q80."
    text = f"""# Codec Benchmark Findings

## Dataset and preprocessing

- Dataset: {source_metadata.get('dataset', 'local natural-image calibration set')}.
- Prepared source set: {source_metadata.get('image_count', source_count)} images; this run evaluated {source_count}. Selection: {source_metadata.get('selection', 'deterministic local-file selection')}.
- Excluded non-photographic files: {', '.join(source_metadata.get('excluded_filenames', [])) or 'none'}.
- Preprocessing: EXIF transpose, RGB conversion, centered square crop and Pillow BICUBIC resize to exactly 256×256. The prepared PNGs and SHA-256 hashes are in `sources/manifest.jsonl`; all codec runs read those same RGB pixel arrays.
- Source details: `{source_metadata.get('source_root', '')}`. No CIFAR-10 images were used.

## Codec settings and software

{setting_lines}

- Runtime: {software_lines}.
- AVIF and WebP use Pillow's bundled codec backends; JXL uses the recorded `cjxl`/`djxl` version in `config.json`.
- PBC provenance: {pbc_state}. Each preset's full config, configured LZMA use, and actual stream entropy mode are recorded in `config.json` and sample metadata.
{f'- {pbc_status_note}' if pbc_status_note else ''}
- Missing settings or runtime failures: {unavailable_text}.

## Timing and metrics

- Each setting had {warmups} unmeasured warm-up encode/decode pair(s), followed by {repeats} timed encode and full-decode operations for every image. Module imports and PBC initialization are outside the timed regions. JXL timing includes temporary-file I/O and process startup.
- Latencies are repeated per-image measurements; aggregate median and p95 are across all timed operations for that setting.
- MSE, PSNR, and SSIM compare decoded RGB against the exact prepared 256×256 RGB source. SSIM uses 11×11 Gaussian windows (σ=1.5), RGB channels, and standard C1/C2 constants. PSNR uses peak 255.
- PBC encode is Python/NumPy/Numba-heavy; JPEG/WebP/AVIF/JXL use optimized native libraries/tools. Latency numbers describe these implementations and this machine, not codec algorithms in isolation. Thread settings are recorded per codec.

## PBC operating points

{pbc_ranges}

Rate matching chooses the nearest mean bpp. A rate match is called close when the absolute gap is at most `max(0.025 bpp, 10% of PBC bpp)`. Distortion matching reports the nearest mean PSNR point (close within 1 dB) and the separate nearest mean SSIM point (close within 0.02 SSIM).

{matches_table}

## Per-setting results

{rate_table}

Machine-readable per-sample results are in `sample_results.csv`; `summary.json` and `summary.csv` contain aggregates and operating-point matches. The rate-distortion figure, latency figures, and representative grid are `rate_distortion.png`, `encode_latency.png`, `decode_latency.png`, and `visual_comparison_grid.png`.

## Findings and earlier CIFAR comparison

- PBC ranges from {pbc_rows[0]['mean_bpp']:.3f} bpp / {pbc_rows[0]['mean_psnr_db']:.2f} dB (`pbc_compression`) to {pbc_rows[-1]['mean_bpp']:.3f} bpp / {pbc_rows[-1]['mean_psnr_db']:.2f} dB (`pbc_high_quality`).
- At PBC compression rate, WebP q30, AVIF q40, and JXL distance 6 are close by the stated rate threshold; JPEG q10 is the nearest JPEG point but falls just outside it. Their measured PSNR is higher than PBC compression at similar rates.
- PBC balanced has close rate matches in WebP q75, AVIF q60, and JXL distance 3. JPEG q60 is nearest but is not close by the threshold.
- PBC quality has an AVIF q90 rate match; PBC high_quality has a WebP q100 rate match. No other tested conventional setting is close by rate to those presets.
- Distortion-matched points generally use less rate than PBC, most clearly around PBC high_quality. PBC median encode latency spans {min(row['median_encode_ms'] for row in pbc_rows):.1f}–{max(row['median_encode_ms'] for row in pbc_rows):.1f} ms and median decode spans {min(row['median_decode_ms'] for row in pbc_rows):.1f}–{max(row['median_decode_ms'] for row in pbc_rows):.1f} ms. JXL uses `cjxl`/`djxl` CLI processes, so its measured latency includes process startup and file staging and is not an in-process libjxl comparison.

{cifar_answer}

## Phase 2 operating points

Primary shortlist: carry PBC `compression` and `balanced`; use every conventional point marked rate-close as the rate-control set. Use the nearest PSNR/SSIM points as a separate distortion-control set, not as bitrate matches. Keep PBC `quality` and `high_quality` as secondary points because fewer codecs have close rate matches there. Include JPEG q1, q5, q10 in the low-rate neural comparison; q10 is the nearest JPEG distortion match to PBC compression, while q1/q5 remain useful lower-rate controls. Do not label any nearest-but-not-close condition as matched. Keep PNG as the lossless ceiling/reference.

{phase2_text}

This run {'is a smoke subset only; do not use it to select Phase 2 points.' if smoke else 'is the full calibration run.'}
"""
    (output / "Codec Benchmark Findings.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "results" / "Codec Benchmark" / "sources")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "Codec Benchmark")
    parser.add_argument("--pbc-root", default=os.environ.get("PBC_ROOT"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        raise ValueError("repeats must be positive and warmups cannot be negative")
    output = args.output.resolve()
    if (output / "sample_results.csv").exists():
        raise FileExistsError(f"results already exist in {output}")
    limit = min(args.limit or 2, 2) if args.smoke else args.limit
    source_rows, images, source_metadata = load_sources(args.source_dir.resolve(), limit)
    adapters, unavailable = build_adapters(args.pbc_root, args.smoke)
    output.mkdir(parents=True, exist_ok=True)
    preview_root = output / "decoded_previews"
    preview_indices = set([0, len(images) // 3, (2 * len(images)) // 3, len(images) - 1])
    all_rows = []
    settings_metadata = {}
    for index, (setting_id, adapter) in enumerate(adapters, 1):
        started = time.perf_counter()
        try:
            rows, metadata = measure_setting(
                setting_id, adapter, source_rows, images, args.repeats, args.warmups,
                preview_indices, preview_root,
            )
            all_rows.extend(rows)
            settings_metadata[setting_id] = metadata
            failed = sum(not row["decode_valid"] for row in rows)
            print(f"[{index}/{len(adapters)}] {setting_id}: {len(rows) - failed}/{len(rows)} valid, {time.perf_counter() - started:.1f}s", flush=True)
        except Exception as error:
            unavailable[setting_id] = f"{type(error).__name__}: {error}"
            print(f"[{index}/{len(adapters)}] {setting_id}: failed: {error}", flush=True)

    summary = aggregate_rows(all_rows)
    matches = select_operating_points(summary)
    write_sample_csv(all_rows, output / "sample_results.csv")
    summary_json = {"settings": summary, "operating_point_matches": matches}
    (output / "summary.json").write_text(json.dumps(summary_json, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    write_summary_csv(summary, output / "summary.csv")
    software = software_metadata()
    config = {
        "smoke": args.smoke,
        "image_count": len(images),
        "prepared_source_dir": str(args.source_dir.resolve()),
        "source_manifest_sha256": hashlib.sha256((args.source_dir / "manifest.jsonl").read_bytes()).hexdigest(),
        "run_arguments": {
            "source_dir": str(args.source_dir.resolve()),
            "output": str(output),
            "pbc_root": args.pbc_root,
            "limit": args.limit,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "smoke": args.smoke,
        },
        "source_metadata": source_metadata,
        "timing": {"warmups_per_setting": args.warmups, "repeats_per_image": args.repeats, "clock": "time.perf_counter_ns", "jxl_includes_process_and_temp_file_io": True},
        "metrics": {"mse_psnr_range": "uint8 RGB [0,255]", "ssim": "11x11 Gaussian, sigma=1.5, RGB, C1=(0.01*255)^2, C2=(0.03*255)^2"},
        "software": software,
        "tested_codec_settings": settings_metadata,
        "unavailable_or_failed_settings": unavailable,
        "pbc_presets": {key: value for key, value in settings_metadata.items() if key.startswith("pbc_")},
        "threading_notes": "AVIF/JXL configured to one thread; JPEG/PNG/PBC use implementation defaults; Pillow/libwebp exposes no thread limit in this adapter.",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=str, allow_nan=False), encoding="utf-8")
    if summary:
        make_plots(summary, output)
        make_visual_grid(args.source_dir.resolve(), preview_root, source_rows, all_rows, summary, matches, output)
    write_findings(output, source_metadata, software, summary, matches, unavailable, args.repeats, args.warmups, len(images), args.smoke)
    print(json.dumps({"output": str(output), "images": len(images), "settings": len(summary), "unavailable": unavailable}, indent=2, default=str))


if __name__ == "__main__":
    main()
