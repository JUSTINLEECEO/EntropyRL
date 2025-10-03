# TODO: make this script more formal
# NOTE: experiment_name
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

# DEBUG: call test client using a batch file
# BATCH_FILE=$(mktemp --suffix=.json)
# cat > ${BATCH_FILE} <<EOF
# {
#   "captions": [
#     {
#       "caption": "The image features a young man sitting in a low table, reaching for a birthday cake placed on a dining table. The cake has a single pink candle on it, indicating that it is a birthday celebration. \n\nAnother person, a man, is present in the scene, sitting behind the woman and holding out their hand towards the cake. A bowl is also visible on the table, close to the cake. The scene captures a joyful moment of a child's birthday celebration.",
#       "image_id": 1290
#     },
#     {
#       "caption": "The image features a young child sitting in a high chair, reaching for a birthday cake placed on a dining table. The cake has a single pink candle on it, indicating that it is a birthday celebration. \n\nAnother person, possibly a woman, is present in the scene, standing behind the child and holding out their hand towards the cake. A cup is also visible on the table, close to the cake. The scene captures a joyful moment of a child's birthday celebration.",
#       "image_id": 1290
#     }
#   ]
# }
# EOF
# python3 "${TEST_PY}" --server "http://127.0.0.1:${PORT}" --batch_file "${BATCH_FILE}" --timeout 60 || true
# rm -f ${BATCH_FILE}

echo "Starting main program..."
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=JustinLeeCEO/MSCOCO2014@train \
    data.val_files=JustinLeeCEO/MSCOCO2014@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_3b_mscoco_grpo_with_dynamic_epsilon_baseline2 \
    trainer.n_gpus_per_node=8 \
    trainer.save_debug_path=null \
    worker.reward.reward_function=./examples/reward_function/coco_cider.py:compute_score

echo "Stopping CHAIR server. (pid=${CHAIR_PID})"
kill ${CHAIR_PID} || true
wait ${CHAIR_PID} 2>/dev/null || true
