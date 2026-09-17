#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
CODES_DIR="${ROOT_DIR}/CLIP-pretrained-model/codes"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[1/4] 运行文本-图像检索..."
"${PYTHON_BIN}" "${CODES_DIR}/text_image_retrieval.py" \
  --device cpu \
  --strategy manual_compressed \
  --caption_path "${ROOT_DIR}/dataset1/caption.json" \
  --image_dir "${ROOT_DIR}/dataset1/images" \
  --weight_path "${CODES_DIR}/ViT-B-32.pt" \
  --output_path "${CODES_DIR}/result/caption_result.json" \
  --cache_path "${CODES_DIR}/result/image_features.pt"

echo "[2/4] 评估文本-图像检索准确率..."
(
  cd "${CODES_DIR}"
  "${PYTHON_BIN}" "${CODES_DIR}/evaluate_result.py" \
    --result_path "${CODES_DIR}/result/caption_result.json" \
    --ground_truth_path "${ROOT_DIR}/dataset1/caption_with_keywords_and_image.json"
)

echo "[3/4] 运行车牌识别扩展测试..."
"${PYTHON_BIN}" "${CODES_DIR}/plate_recognition.py" \
  --device cpu \
  --plate_dir "${ROOT_DIR}/车牌识别/License_plate" \
  --template_dir "${ROOT_DIR}/车牌识别/Character_templates" \
  --weight_path "${CODES_DIR}/ViT-B-32.pt" \
  --output_path "${CODES_DIR}/result/plate_result.json" \
  --evaluate_by_filename

echo "[4/4] 测试完成。"
echo "文本检索结果: ${CODES_DIR}/result/caption_result.json"
echo "文本检索准确率: ${CODES_DIR}/accuracy.txt"
echo "车牌识别结果: ${CODES_DIR}/result/plate_result.json"
