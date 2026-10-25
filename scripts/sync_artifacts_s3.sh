#!/usr/bin/env bash
set -euo pipefail

include_cache=false
include_checkpoints=false
while [[ $# -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --include-cache) include_cache=true ;;
    --include-checkpoints) include_checkpoints=true ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 [--include-cache] [--include-checkpoints] <local-artifact-directory> <s3://bucket/prefix>" >&2
  exit 2
fi

source_dir="$(realpath -e "$1")"
destination="$2"

if [[ ! -d "$source_dir" ]]; then
  echo "Source must be a directory: $source_dir" >&2
  exit 2
fi
if [[ "$destination" != s3://* ]]; then
  echo "Destination must use the s3:// scheme" >&2
  exit 2
fi

filters=(
  --exclude '.aws/*'
  --exclude '*/.aws/*'
  --exclude '.git/*'
  --exclude '*/.git/*'
  --exclude '.env*'
  --exclude '*/.env*'
  --exclude '*.key'
  --exclude '*.crt'
  --exclude '*.p12'
  --exclude '*.pfx'
  --exclude '*.pem'
  --exclude '*credentials*'
  --exclude 'id_rsa*'
  --exclude 'id_ed25519*'
  --exclude '*.tmp'
  --exclude '__pycache__/*'
  --exclude '*/__pycache__/*'
  --exclude '.venv/*'
  --exclude '*/.venv/*'
  --exclude '*/logs/*'
  --exclude '*/raw/*'
)

if [[ "$include_cache" != true ]]; then
  filters+=(
    --exclude 'cache/*'
    --exclude '*/cache/*'
    --exclude '*/features/*'
    --exclude '*/streams/*'
    --exclude '*.bin'
    --exclude '*.mp4'
    --exclude '*.npy'
    --exclude '*.npz'
  )
fi
if [[ "$include_checkpoints" != true ]]; then
  filters+=(--exclude '*/checkpoints/*' --exclude '*.pt' --exclude '*.pth')
fi

aws s3 sync "$source_dir/" "$destination" "${filters[@]}"
