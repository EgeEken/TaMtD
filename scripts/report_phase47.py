from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.analysis.phase47 import (
    descriptive_correlation,
    rate_distortion_residuals,
    semantic_exposure_auc,
    semantic_exposure_depth,
    stage_operation_metadata,
    stage_utility_gain,
    summarize_timing,
)


RESULTS = PROJECT_ROOT / "results/Phase 4.7 Standard Codec Stages"
PHASE45B = PROJECT_ROOT / "results/Phase 4.5B Analysis"
CACHE = PROJECT_ROOT / "data/cache/neural_accessibility/imagenette_320"
CONDITIONS = ("jxl_d6", "jxl_d3", "webp_q30", "webp_q75", "avif_q40", "avif_q60")
FAMILY_COLORS = {"jxl": "#4c78a8", "webp": "#f58518", "avif": "#54a24b"}
STAGE_ORDER = {"raw_bytes": 0, "jxl_dc_progressive": 1, "pre_reconstruction_coefficients": 1, "decoded_yuv": 2, "decoded_rgb": 3}


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _csv_number(value: str | float | int | None) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _build_probe_rows() -> list[dict]:
    rgb = _read_csv(RESULTS / "rgb_probe_per_seed.csv")
    structured = _read_csv(RESULTS / "structured_probe_per_seed.csv")
    raw = [row for row in _read_csv(PHASE45B / "probe_utility_summary.csv") if row["representation"] == "codec_file_bytes" and row["condition"] in CONDITIONS]
    rgb_by_seed = {(row["condition"], int(row["seed"])): row for row in rgb if row["stage"] == "decoded_rgb"}
    output = []
    for row in raw:
        condition, seed = row["condition"], int(row["seed"])
        endpoint = rgb_by_seed[(condition, seed)]
        full_accuracy = float(endpoint["accuracy"])
        full_u = float(endpoint["u_probe_bits"])
        accuracy = float(row["accuracy"])
        u_bits = float(row["u_probe_bits"])
        output.append({
            "condition": condition, "codec_family": condition.split("_", 1)[0], "stage": "raw_bytes", "stage_order": 0,
            "representation": "compressed file bytes", "probe_family": row["model_family"], "seed": seed,
            "n_samples": int(row["n_test"]), "accuracy": accuracy, "cross_entropy_nats": float(row["cross_entropy_nats"]),
            "class_prior_cross_entropy_nats": float(row["class_prior_cross_entropy_nats"]), "u_probe_bits": u_bits,
            "decoded_rgb_accuracy": full_accuracy, "decoded_rgb_u_probe_bits": full_u,
            "relative_u_probe_utility": u_bits / full_u if full_u > 0 else None,
            "relative_accuracy": (accuracy - 0.1) / (full_accuracy - 0.1) if full_accuracy > 0.1 else None,
            "comparison_type": "Phase 4.5B raw-byte probe paired by seed; different probe family",
        })
    for row in rgb:
        output.append({
            **row,
            "codec_family": row["condition"].split("_", 1)[0], "stage_order": STAGE_ORDER.get(row["stage"], 3),
            "probe_family": "RGBResNet18", "comparison_type": "same frozen RGB checkpoint on current same-harness decode",
        })
    for row in structured:
        output.append({
            **row,
            "codec_family": row["condition"].split("_", 1)[0], "stage_order": STAGE_ORDER["pre_reconstruction_coefficients"],
            "probe_family": "BlockStageTransformer", "comparison_type": "representation + shared structured probe",
        })
    return output


def _timing_summary(rows: list[dict]) -> list[dict]:
    output = []
    for item in summarize_timing(rows):
        matching = [row for row in rows if (row["condition"], row["stage"]) == (item["condition"], item["stage"])]
        item["thread_count"] = max(int(row["thread_count"]) for row in matching)
        for source, target in (
            ("capture_adapter_ms", "capture_adapter"),
            ("model_adapter_ms", "model_adapter"),
            ("cumulative_decoder_ms", "cumulative_decoder"),
            ("cumulative_decoder_adapter_ms", "cumulative_decoder_adapter"),
        ):
            values = [float(row[source]) for row in matching]
            item[f"median_{target}_ms"] = float(np.median(values))
            item[f"p95_{target}_ms"] = float(np.percentile(values, 95))
        item["timing_boundary"] = matching[0]["timing_boundary"]
        output.append(item)
    return output


def _stage_manifest(conditions: list[str], timed_stages: set[tuple[str, str]], scored_stages: set[tuple[str, str]]) -> list[dict]:
    rows = []
    for condition in conditions:
        family = condition.split("_", 1)[0]
        stages = ["raw_bytes"]
        if family == "jxl":
            stages += ["jxl_dc_progressive", "decoded_rgb"]
        else:
            stages += ["pre_reconstruction_coefficients", "decoded_yuv", "decoded_rgb"]
        for stage in stages:
            metadata = stage_operation_metadata(condition, stage)
            if stage == "raw_bytes":
                representation = "compressed file byte sequence"
            elif stage == "jxl_dc_progressive":
                representation = "native progressive RGB buffer if emitted by the codestream"
            elif stage == "pre_reconstruction_coefficients":
                representation = "canonical 16x16 token grid with mode, quantizer and coefficient summaries"
            elif stage == "decoded_yuv":
                representation = "fully reconstructed native YUV planes"
            else:
                representation = "fully decoded RGB pixels"
            rows.append({
                **metadata,
                "codec_family": family,
                "representation": representation,
                "timing_available": (condition, stage) in timed_stages,
                "probe_available": (condition, stage) in scored_stages,
                "test_samples": 3925 if (condition, stage) in scored_stages else (256 if (condition, stage) in timed_stages else 0),
                "stage_order": STAGE_ORDER[stage],
            })
    return rows


def _training_resources() -> list[dict]:
    records_dir = CACHE / "phase47" / "training" / "results"
    rows = []
    structured_rows = _read_csv(RESULTS / "structured_probe_per_seed.csv")
    for result in structured_rows:
        checkpoint_stem = Path(result["checkpoint"].replace("\\", "/")).stem
        checkpoint_stem = checkpoint_stem.replace("__block_stage_transformer", "")
        path = records_dir / f"{checkpoint_stem}.json"
        if not path.is_file():
            raise FileNotFoundError(f"training resource record missing for {result['condition']}/seed{result['seed']}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "complete" or record.get("stage") != "pre_reconstruction_coefficients":
            raise ValueError(f"training resource record is not a completed structured run: {path}")
        resources = record.get("resources", {})
        peaks = resources.get("peaks", {})
        minima = resources.get("minima", {})
        start = resources.get("start", {})
        end = resources.get("end", {})
        commit_limits = [int(snapshot["system"]["commit_limit_bytes"]) for snapshot in (start, end) if snapshot.get("system", {}).get("commit_limit_bytes") is not None]
        minimum_commit_limit = min(commit_limits) if commit_limits else None
        peak_commit_used = peaks.get("commit_used_bytes")
        rows.append({
            "condition": record["condition"], "seed": record["seed"], "device": record.get("device"),
            "training_seconds": record.get("training_seconds"), "peak_rss_bytes": peaks.get("peak_rss_bytes"),
            "peak_process_pagefile_bytes": peaks.get("peak_process_pagefile_bytes"),
            "minimum_memory_available_bytes": minima.get("memory_available_bytes"),
            "peak_system_commit_used_bytes": peaks.get("commit_used_bytes"),
            "minimum_system_commit_free_bytes": minimum_commit_limit - int(peak_commit_used) if minimum_commit_limit is not None and peak_commit_used is not None else None,
            "minimum_c_drive_free_bytes": minima.get("disk_free_bytes"),
            "minimum_vram_free_mb": minima.get("vram_free_mb"),
            "peak_vram_used_mb": peaks.get("vram_used_mb"),
            "cuda_allocated_peak_bytes": peaks.get("cuda_allocated_peak_bytes"),
            "cuda_reserved_peak_bytes": peaks.get("cuda_reserved_peak_bytes"),
        })
    return rows


def _format(value, digits: int = 3) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.{digits}f}"


def _write_findings(manifest: list[dict], timing: list[dict], probe_summary: list[dict], sed: list[dict], auc: list[dict], residuals: list[dict], correlations: list[dict], resources: list[dict], parity: list[dict]) -> None:
    versions = json.loads((PROJECT_ROOT / "patches/phase47/versions.json").read_text(encoding="utf-8"))
    lines = [
        "# Phase 4.7 Findings",
        "",
        "## Why this phase exists",
        "",
        "Earlier staged evidence centered on project-owned PBC and JPEG's late, relatively simple DCT representation. Phase 4.7 measured exact cached JPEG XL, lossy WebP and AVIF streams to find where standard-codec representations expose Imagenette semantics as decoding proceeds.",
        "",
        "## Decoder provenance",
        "",
        "| Library | Tag / commit | Build and instrumentation |",
        "|---|---|---|",
    ]
    build_notes = {
        "libjxl": "Release shared decoder; tools/tests/examples off; pinned Brotli/Highway/skcms submodules; runtime uses the resizable runner at one thread.",
        "libwebp": "Release shared decoder; animation, mux, tools and tests off; patch `libwebp-pre-reconstruction.patch`.",
        "libaom": "Release decoder-only shared build; encoder, multithreading, tests/examples/docs off; patch `libaom-pre-reconstruction.patch`.",
        "libavif": "Release shared libavif with system libaom decoder only; encoder and libyuv off.",
    }
    for name, source in versions["sources"].items():
        lines.append(f"| {name} | `{source['tag']}` / `{source['commit']}` | {build_notes[name]} |")
    lines += [
        "",
        "Windows x64 build used CMake 3.29.2, Ninja 1.12.0 and GCC/G++ 13.2.0. Builds and source checkouts are external to the repository. Rebuild with `scripts/build_phase47_codecs.ps1` or `scripts/build_phase47_codecs.sh`; source/commit pins are in `patches/phase47/versions.json`.",
        "",
        "## Exact stage definitions",
        "",
        "The manifest records the operation state and boundary for every condition. A pre-reconstruction feature grid contains no RGB/YUV pixels. The WebP callback stops after the last VP8 residual row; the AV1 hook returns after the tile parse-only pass and before frame reconstruction/filter wrapup.",
        "",
        "| Condition | Stage | Available | Boundary | Entropy / coeffs | Inverse transform / prediction | Pixels / filters / RGB |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in manifest:
        if row["stage"] == "raw_bytes":
            continue
        available = "yes" if row["available"] else "no"
        coeff = "yes" if row["transform_coefficients_decoded"] else "no"
        entropy = "yes" if row["entropy_syntax_decoded"] else "no"
        inv_pred = "yes" if row["inverse_transform_complete"] or row["intra_prediction_applied"] else "no"
        pixels = "yes" if row["pixel_reconstruction_complete"] else "no"
        filters = "yes" if row["loop_filters_complete"] else "no"
        rgb = "yes" if row["rgb_conversion_complete"] else "no"
        lines.append(f"| {row['condition']} | {row['stage']} | {available} | {row['boundary']} | syntax {entropy}; coeffs {coeff} | {inv_pred} | pixels {pixels}; filters {filters}; RGB {rgb} |")
    lines += [
        "",
        "## Timing methodology",
        "",
        "All six operating points use the exact Phase 4.5B in-memory streams and the fixed 8,522/947/3,925 train/validation/test split. Stream manifests and every sample payload hash/length were checked before decoding. Timing warmed each codec once and used 256 systematically spaced held-out samples, three repeats and one decoder thread; filesystem reads and serialization are outside the timed call. `perf_counter_ns` boundaries are described per row in `decoder_timing_per_sample.csv`.",
        "",
        "Pre-reconstruction decoder time ends at the instrumented callback. The capture adapter time covers copying/transformation after that boundary, and the model-adapter time covers conversion to the shared token inputs. The timing boundary includes callback work needed to gather the IR; `extra_capture_work` marks this instrumentation cost. Native YUV and RGB endpoints are separately timed on the same sample/repeat harness. Stage timings are implementation measurements, not codec-algorithm constants.",
        "",
        "| Condition | Stage | n | Decoder median / p95 ms | Capture adapter median ms | Model adapter median ms | Decoder + adapters median ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in timing:
        lines.append(
            f"| {row['condition']} | {row['stage']} | {row['n']} | {_format(row['median_decoder_ms'])} / {_format(row['p95_decoder_ms'])} | {_format(row.get('median_capture_adapter_ms'))} | {_format(row.get('median_model_adapter_ms'))} | {_format(row.get('median_cumulative_decoder_adapter_ms'))} |"
        )
    lines += [
        "",
        "## Probe methodology",
        "",
        "Full RGB endpoints were evaluated with the three frozen Phase 4 RGBResNet18 checkpoints (seeds 42, 123 and 777). The shared BlockStageTransformer has 539,274 parameters, three 128-wide encoder layers and no codec-ID embedding; all four WebP/AV1 conditions used the same AdamW settings (learning rate 3e-4, weight decay 1e-4, batch size 32, at most 20 epochs, patience 6, CUDA mixed precision, workers 0, pin memory off). Checkpoints were selected on validation; test was evaluated once. After correcting WebP skip-byte initialization, the 120-image gate passed at 96.7% training accuracy after 62 epochs. Structured-stage results are **representation + shared structured probe**, not architecture-independent accessibility.",
        "",
        "`U_probe_bits = (CE_prior - CE_model) / ln(2)`. It is a probe utility score and is not Shannon mutual information. The available raw-byte endpoint is the single existing Phase 4.5B byte-probe result per condition; it is context only, uses a different probe family and is not repeated across the three structured-probe seeds.",
        "",
        "| Condition | Stage | Probe | Test accuracy mean ± SD | U-probe bits mean ± SD | Relative U mean | Relative accuracy mean |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in probe_summary:
        lines.append(
            f"| {row['condition']} | {row['stage']} | {row['probe_family']} | {_format(row['accuracy_mean'])} ± {_format(row['accuracy_std'])} | {_format(row['u_probe_bits_mean'])} ± {_format(row['u_probe_bits_std'])} | {_format(row['relative_u_probe_utility_mean'])} | {_format(row['relative_accuracy_mean'])} |"
        )
    for family, title in (("jxl", "JXL findings"), ("webp", "WebP findings"), ("avif", "AVIF/AV1 findings")):
        lines += ["", f"## {title}", ""]
        conditions = [condition for condition in CONDITIONS if condition.startswith(family)]
        for condition in conditions:
            scored = [row for row in probe_summary if row["condition"] == condition and row["stage"] in ("jxl_dc_progressive", "pre_reconstruction_coefficients", "decoded_rgb")]
            timings = [row for row in timing if row["condition"] == condition and row["stage"] in ("jxl_dc_progressive", "pre_reconstruction_coefficients", "decoded_yuv", "decoded_rgb")]
            stage_text = "; ".join(f"{row['stage']}: accuracy {_format(row['accuracy_mean'])}, U {_format(row['u_probe_bits_mean'])} bits" for row in scored)
            time_text = "; ".join(f"{row['stage']} median/p95 {_format(row['median_decoder_ms'])}/{_format(row['p95_decoder_ms'])} ms" for row in timings)
            lines.append(f"- **{condition}:** {stage_text}. {time_text}.")
        if family == "jxl":
            lines.append("The exact cached JXL streams emitted zero `JXL_DEC_FRAME_PROGRESSION` events in the held-out samples; their encoder configuration has no progressive option. They therefore have no genuine pre-full RGB stage here. The pinned libjxl v0.12.0 one-shot RGB path is stable with an explicit one-thread resizable runner. The upstream progressive example crashes in `N_AVX2::DecodeGroupImpl` when fed the exact cached codestream in 1,024-byte chunks; a minimal C reproducer and sample are preserved in the ignored `phase47` cache. The partial-input path was excluded from results. No image was synthesized from full RGB.")
        elif family == "webp":
            lines.append("The opt-in libwebp patch returns at `VP8ProcessRow` after all 16 macroblock rows have passed `ParseResiduals`, before `ReconstructRow` and `FinishRow`. The IR preserves coefficient summaries, prediction-mode histograms, segment and quantizer information. Y2 WHT/DC handling has occurred; spatial IDCT and prediction application have not. The patch initializes the VP8 skip byte before its optional parse so the IR is deterministic; all six WebP probe seeds were rerun from rebuilt exact-stream feature maps. Sixteen held-out full RGB decodes matched the pre-fix build byte-for-byte.")
        else:
            lines.append("The libaom patch runs the actual parse-only visitor through AVIF/libavif, gathers AV1 block modes, qindex/segment/skip and low-frequency coefficient summaries, then exits before inverse transform/reconstruction and frame filters. Native YUV and RGB are decoded through the unmodified full path. This is a real early boundary, not the post-reconstruction inspect callback.")
    lines += [
        "",
        "## Semantic exposure depth",
        "",
        "SED is reported separately for U-probe and chance-corrected accuracy at τ = 0.50, 0.75 and 0.90. It uses only observed pre-RGB stages; no threshold is interpolated. `NR` means not reached before RGB. A fraction above 1 means the feature stage's measured decoder-plus-adapter cost exceeded the full RGB path even though its boundary precedes pixel reconstruction. The normalized AUC uses each seed's observed points; because only one historical raw-byte probe exists per condition, only its matching curve includes that raw-byte point. Structured and byte probes differ, so AUC is descriptive.",
        "",
        "| Condition | Metric | τ | Mean ms among reaching seeds | Fraction of full RGB | Seeds reaching |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in sed:
        if row["seed"] == "mean":
            lines.append(f"| {row['condition']} | {row['metric']} | {float(row['threshold']):.2f} | {_format(row.get('milliseconds'))} | {_format(row.get('fraction_of_full_rgb'))} | {row.get('reached_seed_count', 0)}/3 |")
    lines += ["", "Normalized semantic-exposure AUC (trapezoidal utility vs measured normalized decode cost; descriptive only):", "", "| Condition | U-probe AUC mean ± SD | Accuracy AUC mean ± SD |", "|---|---:|---:|"]
    for condition in CONDITIONS:
        row_u = next((row for row in auc if row["condition"] == condition and row["seed"] == "mean" and row.get("u_probe_auc") is not None), None)
        row_a = next((row for row in auc if row["condition"] == condition and row["seed"] == "mean" and row.get("accuracy_auc") is not None), None)
        lines.append(f"| {condition} | {_format(row_u.get('u_probe_auc') if row_u else None)} ± {_format(row_u.get('u_probe_auc_std') if row_u else None)} | {_format(row_a.get('accuracy_auc') if row_a else None)} ± {_format(row_a.get('accuracy_auc_std') if row_a else None)} |")
    lines += [
        "",
        "## Rate-distortion efficiency vs accessibility",
        "",
        "The six exact standard-codec points use full-Imagenette bpp/PSNR/SSIM from Phase 4.5B. Positive PSNR residual means observed PSNR exceeds the OLS baseline `PSNR ~ log₂(bpp)`; SSIM has an analogous separate OLS residual. Pearson and Spearman values below are descriptive with n=6; no significance claim is made.",
        "",
        "| x | y | n | Pearson r | Spearman ρ |",
        "|---|---|---:|---:|---:|",
    ]
    for row in correlations:
        lines.append(f"| {row['x']} | {row['y']} | {row['n']} | {_format(row.get('pearson_r'))} | {_format(row.get('spearman_rho'))} |")
    lines += ["", "## Rate-distortion efficiency vs exposure depth", "", "SED/AUC correlations use each condition's mean over the three structured-probe seeds where a threshold was reached. The effective sample size can be below six because JXL has no progressive pre-full observation. These associations are exploratory and cannot establish that compression efficiency causes semantic exposure to move later.", ""]
    sed50_correlation = next(row for row in correlations if row["y"] == "sed50_u_fraction")
    auc_correlation = next(row for row in correlations if row["y"] == "u_probe_auc")
    lines.append(
        f"In these points, RD residual versus SED50 fraction is Pearson {_format(sed50_correlation.get('pearson_r'))} "
        f"and Spearman {_format(sed50_correlation.get('spearman_rho'))} (n={sed50_correlation['n']}); "
        f"RD residual versus normalized U-probe AUC is Pearson {_format(auc_correlation.get('pearson_r'))} "
        f"and Spearman {_format(auc_correlation.get('spearman_rho'))} (n={auc_correlation['n']}). "
        "Higher AUC means utility appears earlier, so the observed direction does not support the idea that better RD efficiency pushes semantics deeper. The small sample and differing codec/probe behavior make this a weakening observation, not a general rejection."
    )
    lines.append("| Condition | bpp | PSNR dB | SSIM | PSNR RD residual dB | raw U bits | SED50 ms / fraction | U-probe AUC |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in residuals:
        lines.append(f"| {row['condition']} | {_format(row['bpp'])} | {_format(row['psnr'])} | {_format(row['ssim'])} | {_format(row['rd_efficiency_residual'])} | {_format(row['raw_u_probe_bits'])} | {_format(row.get('sed50_u_ms'))} / {_format(row.get('sed50_u_fraction'))} | {_format(row.get('u_probe_auc'))} |")
    lines += [
        "",
        "## Historical JPEG and PBC comparators",
        "",
        "`historical_jpeg_pbc_comparators.csv` carries the existing JPEG q10 and PBC balanced Phase 4.5B frontier rows. Their decoder implementations, test timing subsets and probe families differ; they are context only and are not overlaid on the same-harness standard-codec plots.",
        "",
        "## Systems interpretation",
        "",
        "The early structured points include instrumentation and adapter costs, while full RGB points use the native decoder paths. The report therefore compares measured implementation paths and probe utility, not a codec-invariant computational lower bound. Raw-byte and structured probes also differ. No single global Pareto claim is made across the historical JPEG/PBC measurements.",
        "",
        "## Strongest supported claims",
        "",
        "- Exact WebP and AVIF streams yield deterministic structured stages before pixel reconstruction, with same-harness native YUV and RGB endpoints.",
        "- The exact cached JXL operating points do not yield a valid pre-full stage through the pinned public API and are not counted as staged results.",
        "- AVIF/AV1 exposes a measured early point with useful but incomplete utility; WebP's capture adapter dominates its online cost, while JXL is full-decode-only at these operating points.",
        "- All codec-to-codec interpretation remains conditioned on the shared but non-identical probe families.",
        "",
        "## Claims that remain unsupported",
        "",
        "The six-point exploratory analysis does not establish that better rate-distortion efficiency pushes semantics deeper, that any probe score is an architecture-independent accessibility measure, or that these timings generalize to other implementations, images, hardware or video.",
        "",
        "## Relevance to upcoming video work",
        "",
        "The AV1 hook sits in libaom's codec-level parse-only tile path and its features describe a frame grid, so the hook itself can be reused behind a future video-frame/OBU adapter. The tested path is AVIF still-image decode only; frame references, inter-frame state, timestamps and video container behavior were not validated. No video or VLM experiment was run.",
        "",
        "## Portfolio-safe summary",
        "",
        "Instrumented open-source WebP/VP8 and AV1/libaom decoders to expose structured representations before pixel reconstruction, then measured classification utility and same-harness decode cost on exact Imagenette codec streams. Tested the cached JPEG XL files through the pinned public progression API; those streams exposed no progressive stage, so JXL is reported as full-decode-only at these operating points.",
        "",
        "## RGB endpoint recheck",
        "",
        f"All 18 current full-RGB checkpoint evaluations used 3,925 held-out samples. Against the Phase 4.5B endpoint table, the maximum absolute accuracy delta was {_format(max(abs(float(row['accuracy_delta'])) for row in parity), 5)} and the maximum absolute U-probe delta was {_format(max(abs(float(row['u_probe_delta_bits'])) for row in parity), 5)} bits. Eight repeated RGB decodes per condition were byte-identical. Small cross-library endpoint differences are retained in `full_decode_rechecks.csv` rather than silently treated as exact parity.",
        "",
        f"The structured training resource artifact contains {len(resources)} completed child rows, including process RSS/pagefile peaks, system commit headroom, disk/RAM/VRAM minima and CUDA allocation peaks. Parent-process disk/memory recovery snapshots are in `training_resource_recovery.csv`.",
    ]
    claim_lines = []
    for condition in ("avif_q40", "avif_q60"):
        structured = next(row for row in probe_summary if row["condition"] == condition and row["stage"] == "pre_reconstruction_coefficients")
        early = next(row for row in timing if row["condition"] == condition and row["stage"] == "pre_reconstruction_coefficients")
        full = next(row for row in timing if row["condition"] == condition and row["stage"] == "decoded_rgb")
        fraction = float(early["median_cumulative_decoder_adapter_ms"]) / float(full["median_decoder_ms"])
        claim_lines.append(
            f"- **{condition}:** pre-reconstruction test accuracy {_format(structured['accuracy_mean'])}, "
            f"relative U-probe utility {_format(structured['relative_u_probe_utility_mean'])}; decoder plus adapters "
            f"{_format(early['median_cumulative_decoder_adapter_ms'])} ms vs full RGB {_format(full['median_decoder_ms'])} ms "
            f"({fraction:.3f} of RGB median)."
        )
    for condition in ("webp_q30", "webp_q75"):
        structured = next(row for row in probe_summary if row["condition"] == condition and row["stage"] == "pre_reconstruction_coefficients")
        early = next(row for row in timing if row["condition"] == condition and row["stage"] == "pre_reconstruction_coefficients")
        full = next(row for row in timing if row["condition"] == condition and row["stage"] == "decoded_rgb")
        claim_lines.append(
            f"- **{condition}:** pre-reconstruction test accuracy {_format(structured['accuracy_mean'])}, "
            f"relative U-probe utility {_format(structured['relative_u_probe_utility_mean'])}; capture plus model adapters "
            f"raise cumulative median to {_format(early['median_cumulative_decoder_adapter_ms'])} ms vs full RGB "
            f"{_format(full['median_decoder_ms'])} ms."
        )
    claims_index = lines.index("## Claims that remain unsupported")
    lines[claims_index:claims_index] = claim_lines + [""]
    (RESULTS / "Phase 4.7 Findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    timing_rows = _read_csv(RESULTS / "decoder_timing_per_sample.csv")
    probe_rows = _build_probe_rows()
    if len([row for row in probe_rows if row["stage"] == "decoded_rgb"]) != 18:
        raise ValueError("expected all 18 current frozen RGB endpoint results")
    structured_rows = [row for row in probe_rows if row["stage"] == "pre_reconstruction_coefficients"]
    if len(structured_rows) != 12:
        raise ValueError("expected 12 structured-probe seed results")
    expected_timing_rows = sum(256 * 3 * (1 if condition.startswith("jxl") else 3) for condition in CONDITIONS)
    if len(timing_rows) != expected_timing_rows:
        raise ValueError(f"unexpected timing row count: {len(timing_rows)}")

    timing_summary = _timing_summary(timing_rows)
    _write_csv(RESULTS / "decoder_timing_summary.csv", timing_summary)
    _write_csv(RESULTS / "stage_probe_per_seed.csv", probe_rows)
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in probe_rows:
        grouped[(row["condition"], row["stage"], row["probe_family"])].append(row)
    probe_summary = []
    for (condition, stage, family), values in sorted(grouped.items()):
        summary = {"condition": condition, "stage": stage, "probe_family": family, "seed_count": len(values)}
        for field in ("accuracy", "cross_entropy_nats", "u_probe_bits", "relative_u_probe_utility", "relative_accuracy"):
            mean, std = _mean_std([float(row[field]) for row in values if _csv_number(row.get(field)) is not None])
            summary[f"{field}_mean"] = mean
            summary[f"{field}_std"] = std
        probe_summary.append(summary)
    _write_csv(RESULTS / "stage_probe_summary.csv", probe_summary)

    timed_stages = {(row["condition"], row["stage"]) for row in timing_summary}
    scored_stages = {(row["condition"], row["stage"]) for row in probe_rows}
    manifest = _stage_manifest(list(CONDITIONS), timed_stages, scored_stages)
    _write_csv(RESULTS / "decoder_stage_manifest.csv", manifest)

    timing_lookup = {(row["condition"], row["stage"]): row for row in timing_summary}
    curve_rows = []
    for probe in probe_rows:
        condition, stage = probe["condition"], probe["stage"]
        timing = timing_lookup.get((condition, stage), {})
        if stage == "raw_bytes":
            decoder_ms = adapter_ms = cumulative_ms = 0.0
            decoder_only_ms = 0.0
        elif stage.startswith("jxl_dc_progressive_"):
            decoder_ms = float(timing.get("median_decoder_ms", 0.0))
            adapter_ms = float(timing.get("median_model_adapter_ms", 0.0) or 0.0)
            cumulative_ms = float(timing.get("median_cumulative_decoder_adapter_ms", decoder_ms) or decoder_ms)
            decoder_only_ms = float(timing.get("median_cumulative_decoder_ms", decoder_ms) or decoder_ms)
        elif stage == "pre_reconstruction_coefficients":
            decoder_ms = float(timing.get("median_decoder_ms", 0.0))
            adapter_ms = float(timing.get("median_capture_adapter_ms", 0.0) or 0.0) + float(timing.get("median_model_adapter_ms", 0.0) or 0.0)
            cumulative_ms = float(timing.get("median_cumulative_decoder_adapter_ms", decoder_ms) or decoder_ms)
            decoder_only_ms = float(timing.get("median_cumulative_decoder_ms", decoder_ms) or decoder_ms)
        else:
            decoder_ms = float(timing.get("median_decoder_ms", 0.0))
            adapter_ms = 0.0
            cumulative_ms = float(timing.get("median_cumulative_decoder_adapter_ms", decoder_ms) or decoder_ms)
            decoder_only_ms = float(timing.get("median_cumulative_decoder_ms", decoder_ms) or decoder_ms)
        curve_rows.append({
            "condition": condition, "codec_family": probe["codec_family"], "stage": stage,
            "stage_order": int(probe["stage_order"]), "probe_family": probe["probe_family"], "seed": probe["seed"],
            "accuracy": float(probe["accuracy"]), "cross_entropy_nats": float(probe["cross_entropy_nats"]),
            "u_probe_bits": float(probe["u_probe_bits"]),
            "relative_accuracy": float(probe["relative_accuracy"]),
            "relative_u_probe_utility": float(probe["relative_u_probe_utility"]),
            "decoder_only_ms": decoder_only_ms, "decoder_ms_at_stage": decoder_ms,
            "model_adapter_ms": adapter_ms, "cumulative_ms": cumulative_ms,
        })
    _write_csv(RESULTS / "semantic_exposure_curve_per_seed.csv", curve_rows)

    curve_groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in curve_rows:
        curve_groups[(row["condition"], int(row["seed"]))].append(row)
    sed_rows = []
    auc_rows = []
    gain_rows = []
    for (condition, seed), stages in sorted(curve_groups.items()):
        full_ms = next(row["cumulative_ms"] for row in stages if row["stage"] == "decoded_rgb")
        for row in semantic_exposure_depth(stages, full_ms):
            sed_rows.append({"condition": condition, "seed": seed, "full_rgb_ms": full_ms, **row})
        auc_rows.append({"condition": condition, "seed": seed, "full_rgb_ms": full_ms, **semantic_exposure_auc(stages, full_ms)})
        for row in stage_utility_gain(stages):
            gain_rows.append({"condition": condition, "seed": seed, **row})
    for condition in CONDITIONS:
        condition_sed = [row for row in sed_rows if row["condition"] == condition]
        for metric, threshold in sorted({(row["metric"], row["threshold"]) for row in condition_sed}):
            matches = [row for row in condition_sed if row["metric"] == metric and row["threshold"] == threshold]
            reached = [row for row in matches if row["status"] == "reached"]
            sed_rows.append({
                "condition": condition, "seed": "mean", "metric": metric, "threshold": threshold,
                "status": "reached" if reached else "not_reached_before_rgb", "reached_seed_count": len(reached),
                "milliseconds": float(np.mean([row["milliseconds"] for row in reached])) if reached else None,
                "fraction_of_full_rgb": float(np.mean([row["fraction_of_full_rgb"] for row in reached])) if reached else None,
            })
        values = [row for row in auc_rows if row["condition"] == condition]
        if values:
            for field in ("u_probe_auc", "accuracy_auc"):
                mean, std = _mean_std([float(row[field]) for row in values])
                auc_rows.append({"condition": condition, "seed": "mean", field: mean, f"{field}_std": std, "n_seeds": len(values)})
    _write_csv(RESULTS / "semantic_exposure_depth.csv", sed_rows)
    _write_csv(RESULTS / "semantic_exposure_auc.csv", auc_rows)
    _write_csv(RESULTS / "stage_utility_gain.csv", gain_rows)

    raw_rd = [row for row in _read_csv(PHASE45B / "raw_byte_accessibility_table.csv") if row["condition"] in CONDITIONS and row["representation"] == "codec_file_bytes"]
    rd_input = [{
        "condition": row["condition"], "codec_family": row["codec_family"],
        "bpp": float(row["imagenette_bpp_mean"]), "psnr": float(row["imagenette_psnr_mean_db"]),
        "ssim": float(row["imagenette_ssim_mean"]), "raw_accuracy": float(row["raw_test_accuracy_mean"]),
        "raw_u_probe_bits": float(row["raw_u_probe_bits_mean"]),
    } for row in raw_rd]
    residuals = rate_distortion_residuals(rd_input)
    sed_means = [row for row in sed_rows if row["seed"] == "mean" and row["metric"] == "relative_u_probe_utility"]
    sed_by_condition = {(row["condition"], float(row["threshold"])): row for row in sed_means}
    auc_means = {row["condition"]: row for row in auc_rows if row["seed"] == "mean" and row.get("u_probe_auc") is not None}
    for row in residuals:
        for threshold, name in ((0.5, "sed50_u"), (0.75, "sed75_u"), (0.9, "sed90_u")):
            item = sed_by_condition.get((row["condition"], threshold), {})
            row[f"{name}_ms"] = item.get("milliseconds")
            row[f"{name}_fraction"] = item.get("fraction_of_full_rgb")
        row["u_probe_auc"] = auc_means.get(row["condition"], {}).get("u_probe_auc")
    _write_csv(RESULTS / "rd_efficiency_stage_analysis.csv", residuals)
    correlations = [
        descriptive_correlation(residuals, "rd_efficiency_residual", "raw_u_probe_bits"),
        descriptive_correlation(residuals, "rd_efficiency_residual", "raw_accuracy"),
    ]
    for y_name in ("sed50_u_fraction", "sed75_u_fraction", "sed90_u_fraction", "u_probe_auc"):
        correlations.append(descriptive_correlation(residuals, "rd_efficiency_residual", y_name))

    historical = [row for row in _read_csv(PHASE45B / "absolute_frontier.csv") if row["point_id"].startswith(("jpeg_q10::", "pbc_balanced::"))]
    _write_csv(RESULTS / "historical_jpeg_pbc_comparators.csv", historical)
    resource_rows = _training_resources()
    _write_csv(RESULTS / "training_resource_per_seed.csv", resource_rows)

    mean_curve_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in curve_rows:
        mean_curve_groups[(row["condition"], row["stage"])].append(row)
    mean_curves = []
    for (condition, stage), values in sorted(mean_curve_groups.items()):
        mean_curves.append({
            "condition": condition, "codec_family": condition.split("_", 1)[0], "stage": stage,
            "stage_order": STAGE_ORDER[stage],
            "cumulative_ms": float(np.mean([row["cumulative_ms"] for row in values])),
            "full_rgb_ms": float(np.mean([row["cumulative_ms"] for row in curve_rows if row["condition"] == condition and row["stage"] == "decoded_rgb"])),
            "relative_u_probe_utility": float(np.mean([row["relative_u_probe_utility"] for row in values])),
            "relative_accuracy": float(np.mean([row["relative_accuracy"] for row in values])),
        })

    for filename, normalized in (("standard_codec_semantic_exposure_absolute_ms.png", False), ("standard_codec_semantic_exposure_normalized.png", True)):
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        for condition in CONDITIONS:
            points = sorted((row for row in mean_curves if row["condition"] == condition), key=lambda row: row["stage_order"])
            xs = [row["cumulative_ms"] / row["full_rgb_ms"] if normalized else row["cumulative_ms"] for row in points]
            ys = [row["relative_u_probe_utility"] for row in points]
            color = FAMILY_COLORS[condition.split("_", 1)[0]]
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=condition, color=color, alpha=0.88)
            for x, y, row in zip(xs, ys, points, strict=True):
                ax.annotate(row["stage"].replace("pre_reconstruction_coefficients", "pre-recon").replace("decoded_rgb", "RGB").replace("raw_bytes", "bytes"), (x, y), xytext=(4, 5), textcoords="offset points", fontsize=7)
        for threshold in (0.5, 0.75, 0.9):
            ax.axhline(threshold, color="#777777", linewidth=0.7, linestyle="--", alpha=0.55)
        ax.set_xlabel("Cumulative decoder + required model-adapter time (fraction of RGB decode)" if normalized else "Cumulative decoder + required model-adapter time (ms)")
        ax.set_ylabel("U-probe utility relative to matched decoded RGB")
        ax.set_title("Standard-codec semantic exposure; points are measured stages")
        ax.grid(alpha=0.2)
        ax.legend(ncol=2, fontsize=8)
        fig.savefig(RESULTS / filename, dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    thresholds = (0.5, 0.75, 0.9)
    x_positions = np.arange(len(CONDITIONS))
    width = 0.23
    for offset, threshold in enumerate(thresholds):
        values = []
        for condition in CONDITIONS:
            row = next((item for item in sed_rows if item["condition"] == condition and item["seed"] == "mean" and item["metric"] == "relative_u_probe_utility" and float(item["threshold"]) == threshold), None)
            values.append(float(row["fraction_of_full_rgb"]) if row and row.get("fraction_of_full_rgb") not in (None, "") else np.nan)
        bars = ax.bar(x_positions + (offset - 1) * width, values, width, label=f"U-probe {threshold:.0%}")
        for i, bar in enumerate(bars):
            if math.isnan(values[i]):
                ax.text(x_positions[i] + (offset - 1) * width, 0.03, "NR", ha="center", va="bottom", rotation=90, fontsize=7)
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_xticks(x_positions, CONDITIONS, rotation=20, ha="right")
    ax.set_ylabel("SED / same-harness full RGB decode time")
    ax.set_title("Semantic Exposure Depth at fixed utility thresholds")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(RESULTS / "semantic_exposure_depth_thresholds.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for row in residuals:
        ax.scatter(row["rd_efficiency_residual"], row["raw_u_probe_bits"], s=55, color=FAMILY_COLORS[row["codec_family"]])
        ax.annotate(row["condition"], (row["rd_efficiency_residual"], row["raw_u_probe_bits"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    correlation = correlations[0]
    ax.set_xlabel("PSNR residual from PSNR ~ log₂(bpp) (dB)")
    ax.set_ylabel("Raw-byte U-probe bits (Phase 4.5B)")
    ax.set_title(f"Rate-distortion residual vs raw-byte accessibility (n={correlation['n']})")
    ax.grid(alpha=0.2)
    fig.savefig(RESULTS / "rd_efficiency_vs_raw_accessibility.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    for row in residuals:
        color = FAMILY_COLORS[row["codec_family"]]
        if row.get("sed50_u_fraction") is not None:
            axes[0].scatter(row["rd_efficiency_residual"], row["sed50_u_fraction"], color=color, s=55)
            axes[0].annotate(row["condition"], (row["rd_efficiency_residual"], row["sed50_u_fraction"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
        if row.get("u_probe_auc") is not None:
            axes[1].scatter(row["rd_efficiency_residual"], row["u_probe_auc"], color=color, s=55)
            axes[1].annotate(row["condition"], (row["rd_efficiency_residual"], row["u_probe_auc"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axes[0].set_ylabel("U-probe SED₅₀ / full RGB time")
    axes[1].set_ylabel("Normalized semantic exposure AUC")
    for ax in axes:
        ax.set_xlabel("PSNR RD-efficiency residual (dB)")
        ax.grid(alpha=0.2)
    fig.suptitle("Descriptive RD-efficiency vs semantic-exposure relationships")
    fig.savefig(RESULTS / "rd_efficiency_vs_semantic_exposure.png", dpi=200)
    plt.close(fig)

    gains = [row for row in gain_rows if row["delta_u_probe_per_ms"] is not None]
    labels = [f"{row['condition']} s{row['seed']}: {row['from_stage']} → {row['to_stage']}" for row in gains]
    values = [float(row["delta_u_probe_per_ms"]) for row in gains]
    if gains:
        fig, ax = plt.subplots(figsize=(10, max(4, len(gains) * 0.22)), constrained_layout=True)
        colors = [FAMILY_COLORS[row["condition"].split("_", 1)[0]] for row in gains]
        ax.barh(np.arange(len(gains)), values, color=colors, alpha=0.8)
        ax.set_yticks(np.arange(len(gains)), labels, fontsize=6)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_xlabel("Δ U-probe bits per ms of measured stage cost")
        ax.set_title("Stage utility gain per decode work (per seed; diagnostic only)")
        ax.grid(axis="x", alpha=0.2)
        fig.savefig(RESULTS / "stage_utility_gain_per_ms.png", dpi=200)
        plt.close(fig)

    schematic = [
        ("JXL cached streams", "full decode only; no progressive event observed"),
        ("WebP early", "VP8 entropy/mode/residual parse; Y2 WHT/DC handling; no spatial IDCT, prediction application, reconstruction, or filters"),
        ("WebP YUV", "inverse transforms, prediction, pixel reconstruction, deblocking; native YUV output"),
        ("AV1 early", "AVIF/frame/tile syntax + entropy/coefficient parse; no inverse transform, prediction application, reconstruction, or loop/restoration filters"),
        ("AVIF YUV", "AV1 reconstruction and loop/restoration filters; native YUV output"),
        ("RGB endpoints", "codec full decode + RGB conversion where required"),
    ]
    fig, ax = plt.subplots(figsize=(13, 3.2), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=schematic, colLabels=("Measured stage", "Operations completed / boundary"), cellLoc="left", colLoc="left", loc="center", colWidths=(0.20, 0.80))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    ax.set_title("Exact decoder-stage boundaries used in Phase 4.7", pad=16)
    fig.savefig(RESULTS / "decoder_stage_operations.png", dpi=200)
    plt.close(fig)

    parity = _read_csv(RESULTS / "full_decode_rechecks.csv")
    max_accuracy_delta = max(abs(float(row["accuracy_delta"])) for row in parity)
    max_u_delta = max(abs(float(row["u_probe_delta_bits"])) for row in parity)
    summary = {
        "conditions": list(CONDITIONS), "test_samples_per_condition": 3925,
        "timing_sample_count_per_condition": 256, "timing_repeats": 3, "thread_count": 1,
        "rgb_probe_seeds": [42, 123, 777], "structured_probe_seeds": [42, 123, 777],
        "jxl_progression_samples": sum(int(row["n_samples"]) for row in probe_rows if str(row["stage"]).startswith("jxl_dc_progressive")),
        "max_rgb_parity_accuracy_delta": max_accuracy_delta,
        "max_rgb_parity_u_probe_delta_bits": max_u_delta,
        "rd_efficiency_vs_raw_accessibility_correlations": correlations[:2],
        "rd_efficiency_vs_semantic_exposure_correlations": correlations[2:],
        "rd_point_count": len(residuals), "rd_residual_model": "OLS: PSNR and SSIM independently regressed on log2(Imagenette bpp)",
        "training_resource_rows": len(resource_rows),
        "resource_monitoring_note": "per-child peak/process telemetry is stored in training_resource_per_seed.csv; parent recovery snapshots are in training_resource_recovery.csv",
        "limitations": [
            "JXL exact cached streams emitted no progressive API event, so the JXL pre-full stage is unavailable.",
            "Structured-stage metrics are representation + shared structured probe and are not architecture-independent.",
            "Raw-byte metrics are one historical Phase 4.5B byte-probe result per condition; raw-byte and structured probes differ.",
            "Six codec operating points make all RD/accessibility correlations descriptive and exploratory.",
        ],
    }
    _write_findings(manifest, timing_summary, probe_summary, sed_rows, auc_rows, residuals, correlations, resource_rows, parity)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "timing_stages": len(timing_summary), "probe_rows": len(probe_rows), "structured_rows": len(structured_rows)}, indent=2))


if __name__ == "__main__":
    main()
