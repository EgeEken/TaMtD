import numpy as np
import pytest
import torch
import json
from pathlib import Path

from tamtd.analysis.phase47 import (
    descriptive_correlation,
    rate_distortion_residuals,
    semantic_exposure_auc,
    semantic_exposure_depth,
    semantic_exposure_pareto,
    stage_operation_metadata,
    stage_utility_gain,
    summarize_timing,
)
from tamtd.codec_stage_ir import (
    FEATURE_FIELDS,
    SCHEMA,
    SCHEMA_VERSION,
    adapt_av1_tokens,
    adapt_webp_tokens,
    serialize_ir_metadata,
    validate_ir,
    verify_stream_alignment,
)
from tamtd.models.block_stage import BlockStageTransformer
from tamtd.phase47.decoder_bindings import _webp_rows_to_features
from scripts.extract_phase47_stages import _test_indices


def test_ir_metadata_and_stream_alignment_are_deterministic(tmp_path):
    metadata = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "pre_reconstruction": True, "stage": "coefficients"}
    features = np.zeros((256, len(FEATURE_FIELDS)), dtype=np.float32)
    validate_ir(metadata, features)
    assert serialize_ir_metadata(metadata) == serialize_ir_metadata(metadata)
    assert serialize_ir_metadata(metadata) == b'{"pre_reconstruction":true,"schema":"tamtd.codec-stage-ir","schema_version":1,"stage":"coefficients"}\n'
    path = tmp_path / "streams.bin"
    first, second = b"abc", b"defg"
    path.write_bytes(first + second)
    base = [
        {"cache_index": 0, "image_id": "a", "label": 1, "split": "train", "rgb_sha256": "x"},
        {"cache_index": 1, "image_id": "b", "label": 2, "split": "test", "rgb_sha256": "y"},
    ]
    import hashlib

    stream_rows = [
        {**base[0], "stream": {"offset": 0, "length": len(first), "sha256": hashlib.sha256(first).hexdigest()}},
        {**base[1], "stream": {"offset": len(first), "length": len(second), "sha256": hashlib.sha256(second).hexdigest()}},
    ]
    verify_stream_alignment(base, stream_rows, path)
    stream_rows[1]["stream"]["offset"] += 1
    with pytest.raises(ValueError, match="contiguous"):
        verify_stream_alignment(base, stream_rows, path)


def test_pre_reconstruction_ir_rejects_pixel_payload():
    features = np.zeros((256, len(FEATURE_FIELDS)), dtype=np.float32)
    with pytest.raises(ValueError, match="pixels"):
        validate_ir(
            {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "pre_reconstruction": True, "rgb": []},
            features,
        )
    assert not any("rgb" in field or "pixel" in field for field in FEATURE_FIELDS)


def test_shared_structured_adapters_and_probe():
    features = np.zeros((256, len(FEATURE_FIELDS)), dtype=np.float32)
    features[:, 34] = 1
    values, modes, mask = adapt_webp_tokens(features)
    av1_values, av1_modes, av1_mask = adapt_av1_tokens(features)
    np.testing.assert_array_equal(values, av1_values)
    np.testing.assert_array_equal(modes, av1_modes)
    np.testing.assert_array_equal(mask, av1_mask)
    model = BlockStageTransformer()
    output = model(
        torch.from_numpy(values).unsqueeze(0),
        torch.from_numpy(modes).unsqueeze(0),
        torch.from_numpy(mask).unsqueeze(0),
    )
    assert output.shape == (1, 10)
    assert 300_000 <= sum(parameter.numel() for parameter in model.parameters()) <= 1_000_000
    again = adapt_webp_tokens(features)
    for left, right in zip((values, modes, mask), again, strict=True):
        np.testing.assert_array_equal(left, right)


def test_extraction_timing_sample_selection_is_deterministic():
    rows = [{"split": "train"}, *({"split": "test"} for _ in range(20)), {"split": "validation"}]
    first = _test_indices(rows, 5)
    assert first == [1, 5, 10, 15, 20]
    assert _test_indices(rows, 5) == first


def test_webp_adapter_rejects_uninitialized_skip_metadata():
    rows = np.zeros(16 * 16 * 800, dtype=np.uint8)
    quant = np.zeros(4 * 8 * 4, dtype=np.uint8)
    rows[797] = 1
    rows[798] = 2
    first = _webp_rows_to_features(rows, quant)
    second = _webp_rows_to_features(rows, quant)
    np.testing.assert_array_equal(first, second)
    assert first[0, 44] == 1
    assert first[0, 45] == 2
    rows[797] = 172
    with pytest.raises(ValueError, match="invalid skip or segment"):
        _webp_rows_to_features(rows, quant)


def test_pre_reconstruction_operation_metadata_has_no_pixel_decode():
    webp = stage_operation_metadata("webp_q30", "pre_reconstruction_coefficients")
    av1 = stage_operation_metadata("avif_q40", "pre_reconstruction_coefficients")
    assert webp["available"] and av1["available"]
    for row in (webp, av1):
        assert row["entropy_syntax_decoded"] is True
        assert row["transform_coefficients_decoded"] is True
        assert row["inverse_transform_complete"] is False
        assert row["intra_prediction_applied"] is False
        assert row["pixel_reconstruction_complete"] is False
        assert row["loop_filters_complete"] is False
        assert row["rgb_conversion_complete"] is False
        assert row["early_exit"] is True
    assert stage_operation_metadata("jxl_d6", "jxl_dc_progressive")["available"] is False
    with pytest.raises(ValueError, match="unknown Phase 4.7 stage"):
        stage_operation_metadata("avif_q40", "invented_stage")


def test_timing_sed_auc_and_stage_gain():
    timings = [
        {"condition": "webp_q30", "stage": "coefficients", "decoder_ms": 2.0},
        {"condition": "webp_q30", "stage": "coefficients", "decoder_ms": 4.0},
    ]
    assert summarize_timing(timings)[0]["median_decoder_ms"] == 3.0
    with_adapter = summarize_timing([
        {"condition": "webp_q30", "stage": "coefficients", "decoder_ms": 2.0, "capture_adapter_ms": 0.2, "model_adapter_ms": 0.1, "cumulative_decoder_ms": 2.0, "cumulative_decoder_adapter_ms": 2.3},
        {"condition": "webp_q30", "stage": "coefficients", "decoder_ms": 4.0, "capture_adapter_ms": 0.4, "model_adapter_ms": 0.1, "cumulative_decoder_ms": 4.0, "cumulative_decoder_adapter_ms": 4.5},
    ])[0]
    assert with_adapter["median_capture_adapter_ms"] == pytest.approx(0.3)
    assert with_adapter["p95_cumulative_decoder_adapter_ms"] > 4.3
    stages = [
        {"stage": "bytes", "cumulative_ms": 0.0, "relative_u_probe_utility": 0.1, "relative_accuracy": 0.2, "u_probe_bits": 0.2, "accuracy": 0.2},
        {"stage": "coefficients", "cumulative_ms": 2.0, "relative_u_probe_utility": 0.7, "relative_accuracy": 0.8, "u_probe_bits": 1.4, "accuracy": 0.6},
        {"stage": "decoded_rgb", "cumulative_ms": 4.0, "relative_u_probe_utility": 1.0, "relative_accuracy": 1.0, "u_probe_bits": 2.0, "accuracy": 0.75},
    ]
    sed = semantic_exposure_depth(stages, 4.0)
    assert sed[0]["stage"] == "coefficients"
    assert sed[1]["status"] == "not_reached_before_rgb"
    auc = semantic_exposure_auc(stages[1:], 4.0, ("coefficients", "decoded_rgb"))
    assert auc["u_probe_auc"] == pytest.approx(0.425)
    assert auc["status"] == "available"
    gain = stage_utility_gain(stages)
    assert gain[0]["delta_u_probe_per_ms"] == pytest.approx(0.6)
    nonmonotonic = [
        {**stages[0], "stage_order": 0},
        {**stages[1], "stage_order": 1, "cumulative_ms": 5.0},
        {**stages[2], "stage_order": 2, "cumulative_ms": 4.0},
    ]
    assert stage_utility_gain(nonmonotonic)[1]["delta_ms"] == -1.0


def test_phase47_auc_requires_consistent_observed_endpoints():
    rgb = {
        "stage": "decoded_rgb", "cumulative_ms": 2.0,
        "relative_u_probe_utility": 1.0, "relative_accuracy": 1.0,
    }
    only_rgb = semantic_exposure_auc([rgb], 2.0, ("pre_reconstruction_coefficients", "decoded_rgb"))
    assert only_rgb["u_probe_auc"] is None
    assert only_rgb["reason"] == "missing_required_stages"

    webp = [
        {"stage": "pre_reconstruction_coefficients", "cumulative_ms": 3.0, "relative_u_probe_utility": 0.7, "relative_accuracy": 0.7},
        rgb,
    ]
    reordered = semantic_exposure_auc(webp, 2.0, ("pre_reconstruction_coefficients", "decoded_rgb"))
    assert reordered["u_probe_auc"] is None
    assert reordered["reason"] == "nonmonotonic_stage_cost"

    missing_endpoint = semantic_exposure_auc(webp[:1], 2.0, ("raw_bytes", "decoded_rgb"))
    assert missing_endpoint["u_probe_auc"] is None
    assert missing_endpoint["reason"] == "missing_required_stages"


def test_semantic_pareto_keeps_only_measured_nondominated_points():
    stages = [
        {"stage": "raw_bytes", "cumulative_ms": 0.0, "accuracy": 0.3},
        {"stage": "pre_reconstruction_coefficients", "cumulative_ms": 3.0, "accuracy": 0.6},
        {"stage": "decoded_rgb", "cumulative_ms": 2.0, "accuracy": 0.8},
    ]
    rows = semantic_exposure_pareto(stages, "accuracy")
    assert [row["nondominated"] for row in rows] == [True, False, True]
    assert [row["stage"] for row in rows] == ["raw_bytes", "pre_reconstruction_coefficients", "decoded_rgb"]


def test_rate_distortion_residual_and_descriptive_correlations():
    rows = [
        {"condition": "a", "bpp": 0.2, "psnr": 20.0, "ssim": 0.5},
        {"condition": "b", "bpp": 0.4, "psnr": 25.0, "ssim": 0.6},
        {"condition": "c", "bpp": 0.8, "psnr": 31.0, "ssim": 0.8},
        {"condition": "d", "bpp": 1.6, "psnr": 33.0, "ssim": 0.82},
    ]
    residuals = rate_distortion_residuals(rows)
    assert len(residuals) == len(rows)
    assert abs(sum(row["rd_efficiency_residual"] for row in residuals)) < 1e-9
    correlation = descriptive_correlation(
        [{"x": row["bpp"], "y": row["psnr"]} for row in rows], "x", "y"
    )
    assert correlation["n"] == 4
    assert correlation["spearman_rho"] == pytest.approx(1.0)


def test_build_manifest_pins_sources_and_committed_patches():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "patches/phase47/versions.json").read_text(encoding="utf-8"))
    assert set(manifest["sources"]) == {"libjxl", "libwebp", "libaom", "libavif"}
    for source in manifest["sources"].values():
        assert len(source["commit"]) == 40
        assert all(character in "0123456789abcdef" for character in source["commit"])
        if source["patch"]:
            patch = root / "patches/phase47" / source["patch"]
            assert patch.is_file()
            assert patch.stat().st_size > 100
    assert "checkout --detach" in (root / "scripts/build_phase47_codecs.sh").read_text(encoding="utf-8")
