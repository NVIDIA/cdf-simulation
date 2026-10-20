#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

isaac_sim_root="${CDF_ISAAC_SIM_ROOT:-/isaac-sim}"

# Every route can end up in `capture.render`, which launches an isolated Isaac
# child through ISAAC_SIM_PATH. This used to be exported for `job-script` only,
# because the legacy router injected it for the `config` route on its way past;
# retiring the router took that with it and left `config --execute` failing with
# "capture.render requires ISAAC_SIM_PATH". Export it once, for every route.
export ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-${isaac_sim_root}}"

# Python JobOrder workflow, explicitly selected with `job-script`:
#   docker run ... IMAGE job-script path/to/job_script.py --execute
# A script can boot `SimulationApp` for `render_async()` or call synchronous
# `render()`, which launches an isolated Isaac child through `ISAAC_SIM_PATH`.
if [ "${1:-}" = "job-script" ]; then
    shift
    if [ "$#" -eq 0 ]; then
        echo "error: job-script requires a Python script path" >&2
        exit 2
    fi

    exec "${isaac_sim_root}/python.sh" "$@"
fi

# Versioned JobConfig YAML workflow:
#   docker run ... IMAGE config path/to/job.yaml [--execute]
if [ "${1:-}" = "config" ]; then
    shift
    if [ "$#" -eq 0 ]; then
        echo "error: config requires a YAML path" >&2
        exit 2
    fi

    exec "${isaac_sim_root}/python.sh" -m cdf.config "$@"
fi

# Direct command strings reach Kit through `--exec` unchanged.
# lean `isaacsim.exp.base.kit` includes the extensions required by headless SDG
# while avoiding the full app's extension and ROS2 startup chain.
exec "${isaac_sim_root}/kit/kit" \
    "${isaac_sim_root}/apps/isaacsim.exp.base.kit" \
    --no-window \
    --/rtx/hydra/supportMultiTickRate=false \
    --exec "$@"
