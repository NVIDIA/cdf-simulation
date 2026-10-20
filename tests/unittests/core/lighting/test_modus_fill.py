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

"""Verify optional neutral fill without importing the Isaac Sim runtime."""

import ast
import unittest
from pathlib import Path
from unittest.mock import Mock

import pcb_aoi  # noqa: F401 -- register the PCB domain
from cdf import JobConfig, CDFJob, build_job_from_config


class ModusFillTest(unittest.TestCase):
    """Check the declaration and actual batch lighting function."""

    def test_job_config_matches_python_declaration(self):
        """Keep versioned YAML and Python authoring behavior equivalent."""
        python_job = CDFJob(domain="pcb-aoi")
        python_job.scene.configure(usd_path="C:/tmp/board.usda", root_path="/Board")
        python_job.appearance.lighting(
            mode="modus_light",
            modus={},
            dome_fill={"intensity": (600.0, 700.0)},
        )
        yaml_job = build_job_from_config(
            JobConfig.model_validate(
                {
                    "api_version": "1.0.0",
                    "domain": "pcb-aoi",
                    "scene": {"usd": "C:/tmp/board.usda", "root": "/Board"},
                    "appearance": {
                        "lighting": {
                            "mode": "modus_light",
                            "modus": {},
                            "dome_fill": {"intensity": [600.0, 700.0]},
                        }
                    },
                }
            )
        )
        self.assertEqual(
            yaml_job.appearance.lighting_spec,
            python_job.appearance.lighting_spec,
        )

    def test_fill_is_preserved_in_compiled_orders(self):
        """Preserve the optional fill in the immutable execution plan."""
        job = CDFJob(domain="pcb-aoi")
        job.scene.configure(usd_path="/tmp/board.usda", root_path="/Board")
        job.appearance.lighting(
            mode="modus_light", modus={}, dome_fill={"intensity": (600, 600)}
        )
        plan = job.compile_orders(serialization_path="/tmp/final_stage.usda")
        order = next(
            order
            for order in plan.orders
            if order.order_type == "appearance.lighting.configure"
        )
        self.assertEqual(tuple(order.parameters["dome_fill"]["intensity"]), (600, 600))

    def test_optional_fill_authoring_and_metadata(self):
        """Author and report only an explicitly configured neutral fill."""
        # Compile the production function alone to avoid Kit-only module imports.
        path = Path(__file__).resolve().parents[4] / "scripts/sdg/standalone/common.py"
        source = ast.parse(path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_randomize_modus_lights"
        )
        for intensity in (None, 0.0, 600.0):
            with self.subTest(intensity=intensity):
                stage = Mock()
                stage.GetPrimAtPath.return_value.IsValid.return_value = False
                ensure_dome = Mock()
                namespace = {
                    "Any": object,
                    "logger": Mock(),
                    "Gf": Mock(Vec3f=lambda *values: values),
                    "Sdf": Mock(),
                    "_sample_range": lambda value: float(value[0]),
                    "_ensure_dome_light": ensure_dome,
                }
                exec(  # noqa: S102 -- execute a fixed, repository-owned function
                    compile(
                        ast.Module(body=[function], type_ignores=[]), str(path), "exec"
                    ),
                    namespace,
                )
                config = {"exposure_range": [1.0, 1.0]}
                if intensity is not None:
                    config["dome_fill"] = {"intensity": [intensity, intensity]}
                metadata = namespace["_randomize_modus_lights"](stage, {}, config)
                if intensity is None:
                    ensure_dome.assert_not_called()
                    self.assertNotIn("dome_fill", metadata)
                else:
                    ensure_dome.assert_called_once_with(
                        stage, intensity, (1.0, 1.0, 1.0), 1.0, visible=True
                    )
                    stage.GetPrimAtPath.return_value.CreateAttribute.assert_called_once_with(
                        "visibleInPrimaryRay", namespace["Sdf"].ValueTypeNames.Bool
                    )
                    stage.GetPrimAtPath.return_value.CreateAttribute.return_value.Set.assert_called_once_with(
                        False
                    )
                    self.assertEqual(metadata["dome_fill"]["intensity"], intensity)
                    self.assertEqual(metadata["dome_fill"]["color"], [1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
