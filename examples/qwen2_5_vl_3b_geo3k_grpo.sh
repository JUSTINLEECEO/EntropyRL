#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/share/liyilin-nfs/models/Qwen2.5-VL-3B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=JustinLeeCEO/geometry3k@train \
    data.val_files=JustinLeeCEO/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_3b_geo_grpo \
    trainer.n_gpus_per_node=8 \
    trainer.save_debug_path=./debug
