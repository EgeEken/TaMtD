import pytest

from tamtd.analysis.phase48 import exact_calibration_duplicate, select_calibration_rows, within_image_correlation


def test_calibration_selection_is_stratified_deterministic_and_aligned():
    rows = [
        {"cache_index": index, "image_id": f"train/class{label}/{index}.jpg", "label": label}
        for label in range(3)
        for index in range(10)
    ]
    first = select_calibration_rows(rows, per_class=3)
    second = select_calibration_rows(list(reversed(rows)), per_class=3)
    assert [row["image_id"] for row in first] == [row["image_id"] for row in second]
    assert len(first) == 9
    assert [sum(row["label"] == label for row in first) for label in range(3)] == [3, 3, 3]
    with pytest.raises(ValueError, match="only 3 samples"):
        select_calibration_rows(rows[:3], per_class=4)


def test_exact_duplicate_requires_same_stream_for_every_image():
    left = [{"image_id": "a", "stream_sha256": "1"}, {"image_id": "b", "stream_sha256": "2"}]
    right = [{"image_id": "a", "stream_sha256": "1"}, {"image_id": "b", "stream_sha256": "2"}]
    assert exact_calibration_duplicate(left, right)
    right[1]["stream_sha256"] = "different"
    assert not exact_calibration_duplicate(left, right)
    assert not exact_calibration_duplicate(left, right[:1])


def test_within_image_correlation_centers_image_difficulty_and_bootstraps_images():
    rows = []
    for image_index, base in enumerate((0.1, 0.3, 0.8, 1.4, 2.0)):
        for setting, delta in enumerate((0.0, 1.0, 2.0)):
            rows.append({"image_id": str(image_index), "bpp": base + delta, "correct": 0.2 * image_index + 0.1 * setting})
    result = within_image_correlation(rows, "bpp", "correct", replicates=200)
    assert result["n_images"] == 5
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["pearson_ci95_low"] == pytest.approx(1.0)
    assert result["centering"] == "within image identity"
