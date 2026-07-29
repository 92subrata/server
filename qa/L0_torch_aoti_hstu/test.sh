#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# HSTU (Generative Recommenders) Torch AOTI test, step 2 (runs inside the
# recsys-examples tritonserver image). Assembles the `platform: "torch_aoti"`
# model repository from the model exported in step 1 (export.sh), starts the
# FlexKV KV-cache server, launches tritonserver, and runs the HSTU client. The
# CI job launches this via `docker run`; this script itself does not use Docker.
#
# The overall flow mirrors the DevTech reference test
# (Devtech-Compute/distributed-recommender: ci/tritonserver_test.sh).

set -e

RECSYS_DIR=${RECSYS_DIR:="/workspace/recsys-examples/examples/hstu"}
EXPORTED_DIR=${EXPORTED_DIR:="/exported_hstu_model"}
MODEL_REPO=${MODEL_REPO:="/triton_model_repo"}

COLOR_ERROR="\033[31m"
COLOR_RESET="\033[0m"
COLOR_SUCCESS="\033[32m"
RET=0

# Assemble the torch_aoti model repository: config from the recsys tree, the
# exported AOTI package as model version 1.
mkdir -p "${MODEL_REPO}"
cp -apr "${RECSYS_DIR}/inference_aoti/triton_aoti/hstu_gr_ranking_kvcache" "${MODEL_REPO}/"
cp -apr "${EXPORTED_DIR}/hstu_gr_ranking_kvcache_model" "${MODEL_REPO}/hstu_gr_ranking_kvcache/1"
cp -apr "${EXPORTED_DIR}/export_test_dump" "${RECSYS_DIR}/inference_aoti"

cd "${RECSYS_DIR}/inference_aoti"
export FLEXKV_LOG_LEVEL=WARNING
export KVCACHE_MANAGER_CONFIG_FILE=${PWD}/kvcache_cpp_runtime.yaml

python3 start_flexkv_server_for_kvcache_cpp.py \
  --config_file "${KVCACHE_MANAGER_CONFIG_FILE}" 2>&1 &
kvserver_pid=$!
sleep 10
kill -0 ${kvserver_pid}

tritonserver --model-repository="${MODEL_REPO}/" &
triton_pid=$!
sleep 30

python3 test_tritonserver_aoti_hstu_model.py > test_benchmark.log || RET=1
cat test_benchmark.log

kill ${triton_pid} || true
kill -9 ${triton_pid} || true
kill ${kvserver_pid} || true

if [[ ${RET} -eq 0 ]]; then
    echo -e "${COLOR_SUCCESS}\n***\n*** HSTU Torch AOTI Test Passed\n***${COLOR_RESET}"
else
    echo -e "${COLOR_ERROR}\n***\n*** HSTU Torch AOTI Test FAILED\n***${COLOR_RESET}"
fi

exit ${RET}
