#!/usr/bin/env bash
set -euo pipefail

required=(
  FLOSSWARE_STORAGE_POSTGRES_DSN
  FLOSSWARE_STORAGE_EMBEDDING_DIMENSIONS
  FLOSSWARE_STORAGE_ORIENTDB_URL
  FLOSSWARE_STORAGE_ORIENTDB_DATABASE
  FLOSSWARE_STORAGE_ORIENTDB_USERNAME
  FLOSSWARE_STORAGE_ORIENTDB_PASSWORD
)

for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    printf 'Missing required environment variable: %s\n' "$name" >&2
    exit 2
  fi
done

python -m pytest -q tests/test_database_adapters.py -k live
