#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <s3://bucket/source-prefix> <local-destination-directory>" >&2
  exit 2
fi

source_uri="$1"
destination="$(realpath -m "$2")"

if [[ "$source_uri" != s3://* ]]; then
  echo "Source must use the s3:// scheme" >&2
  exit 2
fi

mkdir -p "$destination"
aws s3 sync "$source_uri" "$destination/" \
  --exclude '.aws/*' \
  --exclude '*/.aws/*' \
  --exclude '.env*' \
  --exclude '*/.env*' \
  --exclude '*.key' \
  --exclude '*.crt' \
  --exclude '*.p12' \
  --exclude '*.pfx' \
  --exclude '*.pem' \
  --exclude '*credentials*' \
  --exclude 'id_rsa*' \
  --exclude 'id_ed25519*'
