#!/bin/bash

# bash examples/chair_mscoco_test.sh

# set -x
set -euo pipefail

export PYTHONUNBUFFERED=1

MODEL_PATH=/share/liyilin-nfs/models/Qwen2.5-VL-3B-Instruct

PORT=5000
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../CHAIR" && pwd)
CHAIR_PY=${SCRIPT_DIR}/chair.py
TEST_PY=${SCRIPT_DIR}/test_chair.py

echo "Starting chair.py on port ${PORT}..."
python3 "${CHAIR_PY}" \
      --cap_file example_inputs.jsonl \
      --image_id_key image_id \
      --caption_key caption \
      --cache chair.pkl \
      --save_path outputs.json \
      --coco_path /share/liyilin-nfs/datasets/MSCOCO/annotations \
      --serve --port ${PORT} > /dev/null 2>&1 &
CHAIR_PID=$!

# call test client using a batch file
BATCH_FILE=$(mktemp --suffix=.json)
cat > ${BATCH_FILE} <<EOF
{
  "captions": [
    {
      "caption": "There are two bracelets on the boy's wrist, one closer to his hand and another a bit above it. There's also a residual segment of what looks like a tennis racket visible on the left side of the image. In the background, another tennis racket splits the frame vertically. The boy is wearing blue short socks that are visible on both of his legs.",
      "image_id": 0
    },
    {
      "caption": "The wedding scene showcases a beautifully designed wedding cake as the focal point. Surrounding the cake are multiple boxes. Various buttons on their clothing add details on their outfits, likely signifying their enjoying the special occasion. One man is wearing a watch on his left wrist, while the woman is adorned in earrings, and another man sports a tie and a ring, possibly on his hand, which may be on the cake.",
      "image_id": 1
    }
  ]
}
EOF
python3 "${TEST_PY}" --server "http://127.0.0.1:${PORT}" --batch_file "${BATCH_FILE}" --timeout 60 || true
rm -f ${BATCH_FILE}

echo "Stopping CHAIR server. (pid=${CHAIR_PID})"
kill ${CHAIR_PID} || true
wait ${CHAIR_PID} 2>/dev/null || true
