# AWS experiment runner

This is an environment setup guide. It does not create AWS resources or start a billable job. Use the same training and result-writing code on Windows and Linux through `scripts/run_experiment.py`; the runner starts one fresh Python child per condition, waits for its result, records resource diagnostics, then checks resources before proceeding.

## Local Linux setup

On Ubuntu 24.04, run:

```bash
bash scripts/bootstrap_ubuntu.sh
```

The script installs Python 3.12, PDM 2.26.9, the locked project dependencies, and common command-line codec tools. It does not download Imagenette or clone the external PBC repository. Verify the GPU environment after installing the matching NVIDIA driver:

```bash
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
nvidia-smi
```

Install the PBC checkout separately at the commit recorded in the Phase 3 metadata. Set `PBC_ROOT` to its root directory, which must contain `PBC3.py` and the recorded `patch_policy.npz`. Existing saved Windows checkpoint paths are resolved against the PBC checkout root.

## Paths and device

All path variables accept absolute paths. Relative paths are resolved from the repository root. Defaults are `data/`, `data/cache/`, `results/`, and `tmp/tamtd/`.

```bash
export DATA_ROOT=/mnt/tamtd-data
export CACHE_ROOT=/mnt/tamtd-cache
export RESULTS_ROOT=/mnt/tamtd-results
export TEMP_ROOT=/mnt/tamtd-tmp
export PBC_ROOT=/opt/PBC
export CUDA_DEVICE=cuda
```

`CUDA_DEVICE=auto` selects CUDA when available and CPU otherwise. `EXPERIMENT_CONFIG` can be set to a YAML config path; otherwise pass `--config` directly.

## Preview and run

Preview pending Phase 4 entries without launching training:

```bash
.venv/bin/python scripts/run_experiment.py --phase phase4_confirmation --dry-run
```

Run only after reviewing the printed pending list and resource thresholds:

```bash
.venv/bin/python scripts/run_experiment.py --phase phase4_confirmation
```

The runner skips result files already present and Phase 3 seed-42 rows listed for reuse in the YAML. A single condition can be selected explicitly with `--condition`, `--representation`, and `--seed`. The runner requires at least 20 GB free before each child and stops below 12 GB. Windows runs also guard system commit; Linux runs guard available RAM and cgroup headroom. After each child exits, the runner waits 60 seconds and records disk/memory again so delayed Windows pagefile recovery is visible. Do not disable or modify the operating system pagefile or swap configuration to bypass these checks.

Byte-stream conditions can use length buckets and a padded-token budget. For example, the following options apply the new sampler to one explicitly selected pending byte condition:

```bash
.venv/bin/python scripts/run_experiment.py \
  --condition pbc_quality --representation pbc_store_bytes --seed 123 \
  --byte-batch-mode token_budget --max-padded-tokens 1048576 \
  --max-samples-per-batch 32 --effective-batch-size 32
```

The sampler does not truncate streams; gradient accumulation targets the requested effective sample count. Length-based microbatches can overshoot that target, and their sample grouping/order differs from fixed batches, so the result is not numerically equivalent to the Phase 4 protocol. The run records physical batch and optimizer-update size distributions; fixed batching remains available for historical reproduction. DataLoader workers and pinned memory are disabled in the per-condition trainer. Diagnostic fields record padding, throughput, process/system memory where available, disk minima, and GPU memory. Windows process RSS is explicitly unavailable because the runner does not trust working-set counters there; system commit and disk telemetry remain available.

Run the bounded sampler diagnostic separately from benchmark conditions:

```bash
.venv/bin/python scripts/run_experiment.py --diagnostic pbc_quality_store --device cuda
```

It uses a deterministic subset of cached PBC quality STORE streams, runs one training epoch, writes no checkpoint, and does not replace or append a Phase 4 result row. The same safety checks apply.

## Storage, results, and checkpoints

Set `CACHE_ROOT` and `TEMP_ROOT` to volumes with enough headroom. Dataset cache, memmaps, checkpoints, and temporary files stay outside Git. Per-condition JSON results are written after evaluation with an atomic replacement; checkpoint saves also use a temporary file and replacement. Runner sidecars are written next to results and contain pre-run, minimum, and post-exit resource snapshots.

## AWS instance sizing

Do not choose an instance from GPU memory alone. Host RAM matters for memmaps, data decoding, and model cleanup. AWS currently lists `gr6.4xlarge` with 16 vCPUs, 128 GiB host RAM, and one 24 GB L4; `g6e.4xlarge` has 16 vCPUs, 128 GiB host RAM, and one 48 GB L40S. The first is a reasonable one-GPU option for the existing benchmark conditions; the second offers more GPU memory for later large-model inference or tuning. Check instance availability in the target region before planning.

Estimate the full run in the [AWS Pricing Calculator](https://docs.aws.amazon.com/pricing-calculator/latest/userguide/ec2-estimates.html), including region, expected running hours, EBS volume size/type, snapshots, data transfer, and any stopped-but-retained storage. Instance rates vary by region and purchase mode; this guide intentionally gives no fixed cost estimate. See the [EC2 accelerated instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html) and [AWS accelerated-computing instance list](https://aws.amazon.com/ec2/instance-types/accelerated-computing/).

## Artifact sync

The optional `scripts/sync_artifacts_s3.sh` uploads a selected local directory with the AWS CLI. It uses the active AWS profile or instance role, excludes common cache/checkpoint/tensor files, and never deletes destination objects. Review the source and destination before running it:

```bash
bash scripts/sync_artifacts_s3.sh results/Phase\ 4\ Confirmation s3://YOUR-BUCKET/tamtd/phase4
bash scripts/sync_artifacts_s3.sh --include-checkpoints results/Phase\ 4\ Confirmation s3://YOUR-BUCKET/tamtd/phase4-with-checkpoints
bash scripts/sync_artifacts_s3.sh --include-cache --include-checkpoints "$CACHE_ROOT" s3://YOUR-BUCKET/tamtd/cache
```

The first command uploads compact result files only. Checkpoints and regenerable cache files require the explicit flags shown above; raw dataset paths remain excluded. The companion download-only helper can fetch source data to a selected local directory:

```bash
bash scripts/sync_data_s3.sh s3://YOUR-BUCKET/imagenette "$DATA_ROOT/raw/imagenette"
```

Both helpers rely on the AWS CLI profile or instance role and do not delete destination files. Install/configure the AWS CLI separately. Do not place access keys in scripts, YAML, or result metadata.
