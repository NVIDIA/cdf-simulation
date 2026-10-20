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

"""Engine-side visualization output: USD export of solved solder scenes.

``usd_export.export_scene`` writes solder meshes + solid parts as a standalone
USD with PBR materials -- the ``--capture-dir`` debug artifact of the
fill-sweep CLI (``python -m pcb_aoi.solder_evolver.solve``), viewable in
usdview or any USD tool. The engine renders nothing itself.
"""
