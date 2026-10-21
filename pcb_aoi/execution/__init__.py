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

"""PCB-AOI execution adapters."""

from .isaac_adapter import (
    PIN_TIP_ORDER_TYPE,
    PIN_VARIATION_ORDER_TYPE,
    handle_pin_tip,
    handle_pin_variation,
    isaac_handlers,
    register,
)
from .pin_tip_contract import PinTipParameters
from .pin_variation_contract import PinVariationGroup, PinVariationParameters

__all__ = [
    "PIN_TIP_ORDER_TYPE",
    "PIN_VARIATION_ORDER_TYPE",
    "PinTipParameters",
    "PinVariationGroup",
    "PinVariationParameters",
    "handle_pin_tip",
    "handle_pin_variation",
    "isaac_handlers",
    "register",
]
