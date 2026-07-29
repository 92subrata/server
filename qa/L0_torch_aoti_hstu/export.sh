#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# HSTU Torch AOTI test, step 1 (runs inside the recsys-examples inference
# image). Exports the AOTI model from the KuaiRand-1K checkpoint and runs the
# C++ KV-cache inference to produce a reference dump, then copies both to
# /exported_hstu_model for step 2 (test.sh). The CI job launches this via
# `docker run`; this script itself does not use Docker.

set -e

RECSYS_DIR=${RECSYS_DIR:="/workspace/recsys-examples/examples/hstu"}
HSTU_CKPT_NAME=${HSTU_CKPT_NAME:="fused_kuairand_1k_ckpt"}
EXPORTED_DIR=${EXPORTED_DIR:="/exported_hstu_model"}

export FLEXKV_LOG_LEVEL=WARNING
export DYNAMICEMB_OPS_LIB_DIR=/workspace/recsys-examples/corelib/dynamicemb/torch_binding_build/
export PYTHONPATH=${PYTHONPATH}:/workspace/recsys-examples/examples/

cd "${RECSYS_DIR}"
export KVCACHE_MANAGER_CONFIG_FILE=./inference_aoti/kvcache_cpp_runtime.yaml

python3 ./inference_aoti/export_inference_gr_ranking_kvcache.py \
  --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin \
  --checkpoint_dir "${RECSYS_DIR}/ckpt/${HSTU_CKPT_NAME}" \
  --max_bs 2 --kvcache_config_file "${KVCACHE_MANAGER_CONFIG_FILE}"

python3 ./inference_aoti/start_flexkv_server_for_kvcache_cpp.py \
  --config_file "${KVCACHE_MANAGER_CONFIG_FILE}" > cpp_kvcache_server.log 2>&1 &
kvserver_pid=$!
sleep 10
kill -0 ${kvserver_pid}

./inference_aoti/cpp_inference/build/inference_hstu_gr_ranking_kvcache_exported_model \
  ./inference_aoti/hstu_gr_ranking_kvcache_model \
  ./inference_aoti/export_test_dump
kill ${kvserver_pid} || true

mkdir -p "${EXPORTED_DIR}"
cp -apr "${RECSYS_DIR}/inference_aoti/hstu_gr_ranking_kvcache_model" "${EXPORTED_DIR}/"
cp -apr "${RECSYS_DIR}/inference_aoti/export_test_dump" "${EXPORTED_DIR}/"
