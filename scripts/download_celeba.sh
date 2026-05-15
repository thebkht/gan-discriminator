#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/celeba"
ZIP_PATH="${DATA_DIR}/img_align_celeba.zip"
EXTRACT_DIR="${DATA_DIR}/img_align_celeba"
FINAL_IMG_DIR="${EXTRACT_DIR}/img_align_celeba"
DATASET_REF="jessicali9530/celeba-dataset"

mkdir -p "${DATA_DIR}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Error: kaggle CLI is not installed." >&2
  echo "Install it with: pip install kaggle" >&2
  exit 1
fi

if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
  echo "Error: Kaggle credentials not found at ~/.kaggle/kaggle.json" >&2
  exit 1
fi

if ! kaggle datasets status "${DATASET_REF}" >/dev/null 2>&1; then
  echo "Error: Kaggle credentials appear invalid or the Kaggle API is unreachable." >&2
  exit 1
fi

if [[ ! -f "${ZIP_PATH}" ]]; then
  echo "Downloading CelebA archive from Kaggle..."
  kaggle datasets download \
    --dataset "${DATASET_REF}" \
    --file img_align_celeba.zip \
    --path "${DATA_DIR}"
else
  echo "Archive already exists: ${ZIP_PATH}"
fi

if [[ -d "${FINAL_IMG_DIR}" ]]; then
  echo "Extracted dataset already exists: ${FINAL_IMG_DIR}"
  exit 0
fi

echo "Extracting archive..."
unzip -q -o "${ZIP_PATH}" -d "${EXTRACT_DIR}"

if [[ ! -d "${FINAL_IMG_DIR}" ]]; then
  echo "Error: extraction finished but ${FINAL_IMG_DIR} was not created." >&2
  exit 1
fi

echo "CelebA is ready at: ${FINAL_IMG_DIR}"
