#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# End-to-end HSTU (Generative Recommenders) Torch AOTI test.
#
# Exports the HSTU ranking model to an AOTI package from the KuaiRand-1K
# checkpoint, produces a reference output dump with the C++ KV-cache runtime,
# then serves the package with `platform: "torch_aoti"` and runs the HSTU
# client against it. The command sequence mirrors the DevTech reference test
# (Devtech-Compute/distributed-recommender: ci/tritonserver_test.sh), which
# splits these phases across two container images.
#
# The test image must provide the recsys-examples stack (export tooling,
# dynamicemb, the C++ KV-cache runtime, FlexKV, the torch_aoti model
# configuration, and the HSTU client), with the HSTU dataset and checkpoint
# mounted into RECSYS_DIR. tritonserver itself comes from TRITON_DIR, which CI
# points at a build from the pipeline under test.

source ../common/util.sh

if [[ "${DEBUG}" == "true" ]]; then
    set -x
else
    set +x
fi

# The CI harness runs this with `bash -ex`. Errexit is disabled because the test
# checks exit codes itself and has to reach the FlexKV and tritonserver teardown
# on failure. util.sh helpers re-enable it, so it is cleared again after each one.
set +e

COLOR_DARK="\033[90m"
COLOR_ERROR="\033[31m"
COLOR_INFO="\033[94m"
COLOR_RESET="\033[0m"
COLOR_SUCCESS="\033[32m"
RET=0

export CUDA_VISIBLE_DEVICES=0

TESTDIR=`pwd`

# TRITON_DIR selects which Triton is exercised. The recsys image ships a released
# tritonserver with a pinned PyTorch backend, so CI points TRITON_DIR at a mounted
# tree built by this pipeline in order to catch regressions in Triton itself.
TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${BACKEND_DIR:=${TRITON_DIR}/backends}
SERVER_TIMEOUT=${SERVER_TIMEOUT:=300}
export LD_LIBRARY_PATH=${TRITON_DIR}/lib:${LD_LIBRARY_PATH}

# recsys-examples HSTU tree shipped in the test image. inference_aoti holds the
# export script, the C++ KV-cache runtime, the FlexKV launcher, the torch_aoti
# model configuration, and the client.
RECSYS_DIR=${RECSYS_DIR:="/workspace/recsys-examples/examples/hstu"}
AOTI_DIR=${AOTI_DIR:=${RECSYS_DIR}/inference_aoti}

MODEL_NAME=${MODEL_NAME:="hstu_gr_ranking_kvcache"}
HSTU_CKPT_NAME=${HSTU_CKPT_NAME:="fused_kuairand_1k_ckpt"}
HSTU_CKPT_DIR=${HSTU_CKPT_DIR:=${RECSYS_DIR}/ckpt/${HSTU_CKPT_NAME}}
GIN_CONFIG=${GIN_CONFIG:=${RECSYS_DIR}/inference/configs/kuairand_1k_inference_ranking.gin}
MAX_BS=${MAX_BS:=2}

# Everything the test produces is kept under the test directory so CI collects it
# and reruns start from a clean slate.
MODELDIR=${MODELDIR:=${TESTDIR}/models}
EXPORTED_MODEL=${EXPORTED_MODEL:=${TESTDIR}/${MODEL_NAME}_model}
DUMP_DIR=${DUMP_DIR:=${TESTDIR}/export_test_dump}

CLIENT_LOG="${TESTDIR}/${MODEL_NAME}-client.log"
SERVER_LOG="${TESTDIR}/${MODEL_NAME}-server.log"
EXPORT_LOG="${TESTDIR}/${MODEL_NAME}-export.log"
KVCACHE_LOG="${TESTDIR}/${MODEL_NAME}-kvcache.log"

BACKENDS=${BACKENDS:="pytorch"}
export BACKENDS

# dynamicemb ops lib and the recsys examples package are needed by the export
# tooling and the client.
export FLEXKV_LOG_LEVEL=${FLEXKV_LOG_LEVEL:="WARNING"}
export DYNAMICEMB_OPS_LIB_DIR=${DYNAMICEMB_OPS_LIB_DIR:="/workspace/recsys-examples/corelib/dynamicemb/torch_binding_build/"}
export PYTHONPATH=${PYTHONPATH}:/workspace/recsys-examples/examples/
export KVCACHE_MANAGER_CONFIG_FILE=${KVCACHE_MANAGER_CONFIG_FILE:=${AOTI_DIR}/kvcache_cpp_runtime.yaml}

KVCACHE_PID=0

# The KV-cache runtime and the torch_aoti model both talk to a FlexKV server.
# It is restarted between the reference run and serving so Triton sees a clean
# cache, matching the reference test's two-phase flow.
function start_kvcache_server () {
    KVCACHE_PID=0
    python3 ${AOTI_DIR}/start_flexkv_server_for_kvcache_cpp.py \
        --config_file ${KVCACHE_MANAGER_CONFIG_FILE} >> ${KVCACHE_LOG} 2>&1 &
    local pid=$!
    sleep 10
    if ! kill -0 ${pid} > /dev/null 2>&1; then
        echo -e "${COLOR_ERROR}\n***\n*** Failed to start FlexKV KV-cache server\n***${COLOR_RESET}" 1>&2
        cat ${KVCACHE_LOG} 1>&2
        return 1
    fi
    KVCACHE_PID=${pid}
    echo -e "${COLOR_DARK}FlexKV KV-cache server running (pid: ${KVCACHE_PID})${COLOR_RESET}"
}

function stop_kvcache_server () {
    if [[ "${KVCACHE_PID}" -ne 0 ]]; then
        echo -e "${COLOR_DARK}Killing FlexKV KV-cache server (pid: ${KVCACHE_PID})${COLOR_RESET}"
        kill ${KVCACHE_PID} > /dev/null 2>&1 || true
        wait ${KVCACHE_PID} > /dev/null 2>&1 || true
        KVCACHE_PID=0
    fi
}

rm -rf ${MODELDIR} ${EXPORTED_MODEL} ${DUMP_DIR}

# The export tooling and the client resolve dataset and config paths relative to
# the recsys tree.
cd ${RECSYS_DIR}

# Export the AOTI package from the ranking checkpoint. The export script writes
# it under AOTI_DIR; it is relocated to the test directory below.
echo -e "${COLOR_DARK}Exporting ${MODEL_NAME} from ${HSTU_CKPT_DIR}${COLOR_RESET}"
python3 ${AOTI_DIR}/export_inference_gr_ranking_kvcache.py \
    --gin_config_file ${GIN_CONFIG} \
    --checkpoint_dir ${HSTU_CKPT_DIR} \
    --max_bs ${MAX_BS} \
    --kvcache_config_file ${KVCACHE_MANAGER_CONFIG_FILE} > ${EXPORT_LOG} 2>&1
EXIT_CODE=$?
if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo -e "${COLOR_ERROR}\n***\n*** AOTI export failed with exit code ${EXIT_CODE}\n***${COLOR_RESET}" 1>&2
    cat ${EXPORT_LOG} 1>&2
    echo -e "${COLOR_ERROR}\n***\n*** Test Suite FAILED\n***${COLOR_RESET}" 1>&2
    exit 1
fi
if [[ ! -d ${AOTI_DIR}/${MODEL_NAME}_model ]]; then
    echo -e "${COLOR_ERROR}\n***\n*** Export did not produce ${AOTI_DIR}/${MODEL_NAME}_model\n***${COLOR_RESET}" 1>&2
    cat ${EXPORT_LOG} 1>&2
    echo -e "${COLOR_ERROR}\n***\n*** Test Suite FAILED\n***${COLOR_RESET}" 1>&2
    exit 1
fi
mv ${AOTI_DIR}/${MODEL_NAME}_model ${EXPORTED_MODEL}

# Generate the reference output dump the client compares Triton against. The C++
# KV-cache runtime needs a FlexKV server to talk to.
echo -e "${COLOR_DARK}Generating reference dump with the C++ KV-cache runtime${COLOR_RESET}"
start_kvcache_server || exit 1
${AOTI_DIR}/cpp_inference/build/inference_hstu_gr_ranking_kvcache_exported_model \
    ${EXPORTED_MODEL} \
    ${DUMP_DIR} >> ${EXPORT_LOG} 2>&1
EXIT_CODE=$?
stop_kvcache_server
if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo -e "${COLOR_ERROR}\n***\n*** Reference inference failed with exit code ${EXIT_CODE}\n***${COLOR_RESET}" 1>&2
    cat ${EXPORT_LOG} 1>&2
    echo -e "${COLOR_ERROR}\n***\n*** Test Suite FAILED\n***${COLOR_RESET}" 1>&2
    exit 1
fi

# The client reads the dump from inference_aoti, so link it to the copy held with
# the test artifacts. The link name is removed first: `ln -sfn` would otherwise
# create the link inside a pre-existing directory of that name.
rm -rf ${AOTI_DIR}/export_test_dump
ln -sfn ${DUMP_DIR} ${AOTI_DIR}/export_test_dump

# Assemble the model repository: torch_aoti configuration from the recsys tree,
# exported package as version 1.
echo -e "${COLOR_DARK}Setting up model repository in ${MODELDIR}${COLOR_RESET}"
mkdir -p ${MODELDIR}
cp -r ${AOTI_DIR}/triton_aoti/${MODEL_NAME} ${MODELDIR}/${MODEL_NAME}
cp -r ${EXPORTED_MODEL} ${MODELDIR}/${MODEL_NAME}/1
echo -e "${COLOR_DARK}ls ${MODELDIR}/${MODEL_NAME}${COLOR_RESET}"
ls -lha ${MODELDIR}/${MODEL_NAME}

start_kvcache_server || exit 1

SERVER_ARGS="--model-repository=${MODELDIR} --backend-directory=${BACKEND_DIR} --log-verbose=1"
echo -e "${COLOR_DARK}Running ${SERVER} (backends: ${BACKEND_DIR})${COLOR_RESET}"
# Reports a dynamic-linker mismatch between the overlaid tritonserver and the
# image's libraries directly, instead of as an opaque startup timeout.
if ! ${SERVER} --version; then
    echo -e "${COLOR_ERROR}\n***\n*** ${SERVER} failed to run\n***${COLOR_RESET}" 1>&2
    stop_kvcache_server
    echo -e "${COLOR_ERROR}\n***\n*** Test Suite FAILED\n***${COLOR_RESET}" 1>&2
    exit 1
fi
run_server
set +e
if [[ "${SERVER_PID}" -eq 0 ]]; then
    echo -e "${COLOR_ERROR}\n***\n*** Failed to start ${SERVER}\n***${COLOR_RESET}" 1>&2
    cat ${SERVER_LOG} 1>&2
    stop_kvcache_server
    echo -e "${COLOR_ERROR}\n***\n*** Test Suite FAILED\n***${COLOR_RESET}" 1>&2
    exit 1
fi

# The client resolves the reference dump relative to inference_aoti.
cd ${AOTI_DIR}
TEST_NAME="test_tritonserver_aoti_hstu_model"
python3 ./${TEST_NAME}.py > ${CLIENT_LOG} 2>&1
EXIT_CODE=$?
cat ${CLIENT_LOG}
if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo -e "${COLOR_ERROR}\n***\n*** Test '${TEST_NAME}' Failed with exit code ${EXIT_CODE}\n***${COLOR_RESET}" 1>&2
    RET=1
else
    echo -e "${COLOR_INFO}\n***\n*** Test '${TEST_NAME}' Passed\n***${COLOR_RESET}"
fi

echo -e "${COLOR_DARK}Killing server (pid: ${SERVER_PID})${COLOR_RESET}"
kill -s SIGINT ${SERVER_PID}
wait ${SERVER_PID} || true
stop_kvcache_server

if [[ ${RET} -ne 0 ]]; then
    echo -e "${COLOR_ERROR}\n***\n*** Test Suite FAILED\n***${COLOR_RESET}" 1>&2
else
    echo -e "${COLOR_SUCCESS}\n***\n*** Test Suite PASSED\n***${COLOR_RESET}"
fi

exit ${RET}
