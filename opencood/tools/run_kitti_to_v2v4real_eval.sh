#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <kitti_train_config_yaml> <model_dir> [v2v4real_test_dir]" >&2
  exit 1
fi

KITTI_CONFIG="$1"
MODEL_DIR="$2"
V2V4REAL_TEST_DIR="${3:-./V2V4Real/test}"

python opencood/tools/train.py --hypes_yaml "${KITTI_CONFIG}"

python - <<PY
import os
import yaml

config_path = os.path.join("${MODEL_DIR}", "config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("fusion", {})["core_method"] = "IntermediateFusionDataset"
cfg["validate_dir"] = "${V2V4REAL_TEST_DIR}"

with open(config_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"Updated {config_path} for V2V4Real evaluation")
PY

python opencood/tools/inference.py \
  --model_dir "${MODEL_DIR}" \
  --fusion_method intermediate
