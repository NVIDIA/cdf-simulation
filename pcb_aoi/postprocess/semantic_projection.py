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

"""Project component semantics for registered shared-prototype PCB panels.

Ordinary board stages receive Replicator semantics directly. Native USD panel
instances require a different path because authoring semantics through an
instance proxy can affect its shared prototype; observed panel stages lost
their authored material appearance and rendered white after that mutation.
This module implements the post-capture half of the instance-safe path used
only when ``scene.instances(...)`` has registered and validated a board cohort.

The capture runtime temporarily requests BasicWriter instance-ID images and
composed-prim-path mappings instead of native semantic segmentation. It passes
catalog-derived component records to :func:`project_instance_semantics`, which
maps component roots, package geometry, contacts, and associated board pads
across every registered placement. The resulting
``semantic_segmentation_*.png`` and label JSON files follow the same artifact
contract as direct semantic capture. Associated board pads are included so
bottom-side THT geometry can remain labeled when a pad occludes its package
lead.

This is not the default annotation implementation and it does not infer panel
instances by inspecting filenames or USD hierarchy. Standalone, baked, and
otherwise unregistered board stages must remain on the direct Replicator path.
The capture runtime owns that selection boundary and invokes this module only
after instance-ID capture has completed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from cdf.errors import ValidationError
from cdf.postprocess.semantic_colors import semantic_color

_BACKGROUND = {"class": "BACKGROUND"}
_UNLABELLED = {"class": "UNLABELLED"}


def _parse_mapping_id(value: str) -> int:
    """Parse a raw or colorized Replicator segmentation identifier."""
    text = value.strip()
    if not text.startswith("("):
        try:
            return int(text)
        except ValueError as error:
            raise ValidationError(
                f"invalid instance segmentation identifier {value!r}",
            ) from error
    if not text.endswith(")"):
        raise ValidationError(
            f"invalid colorized instance identifier {value!r}",
        )
    try:
        channels = tuple(int(part.strip()) for part in text[1:-1].split(","))
    except ValueError as error:
        raise ValidationError(
            f"invalid colorized instance identifier {value!r}",
        ) from error
    if len(channels) != 4 or any(channel < 0 or channel > 255 for channel in channels):
        raise ValidationError(
            f"colorized instance identifier must contain four bytes: {value!r}",
        )
    red, green, blue, alpha = channels
    return red | green << 8 | blue << 16 | alpha << 24


def _packed_color(color: Sequence[int]) -> int:
    """Pack one RGBA color into Replicator's little-endian integer form."""
    red, green, blue, alpha = color
    return int(red) | int(green) << 8 | int(blue) << 16 | int(alpha) << 24


def _component_path_semantics(
    projection: Mapping[str, object],
    semantic_types: frozenset[str],
) -> dict[str, dict[str, str]]:
    """Expand source component semantics across registered board instances."""
    board_root = projection.get("board_root")
    instance_paths = projection.get("instance_paths")
    components = projection.get("components")
    if not isinstance(board_root, str) or not board_root.startswith("/"):
        raise ValidationError("semantic projection requires an absolute board_root")
    if not isinstance(instance_paths, (list, tuple)) or not instance_paths:
        raise ValidationError("semantic projection requires board instance paths")
    if not isinstance(components, (list, tuple)) or not components:
        raise ValidationError("semantic projection requires component records")

    expanded = {}
    for component in components:
        if not isinstance(component, Mapping):
            raise ValidationError("semantic projection component must be a mapping")
        source_path = component.get("source_path")
        semantics = component.get("semantics")
        if not isinstance(source_path, str) or not source_path.startswith(
            f"{board_root.rstrip('/')}/",
        ):
            raise ValidationError(
                f"semantic projection component is outside {board_root!r}",
            )
        if not isinstance(semantics, Mapping):
            raise ValidationError("semantic projection component has no semantics")
        selected = {
            str(name): str(value)
            for name, value in semantics.items()
            if str(name) in semantic_types
        }
        if not selected:
            continue
        member_paths = component.get("member_paths") or (source_path,)
        if not isinstance(member_paths, (list, tuple)):
            raise ValidationError(
                "semantic projection component member_paths must be a sequence",
            )
        for member_path in member_paths:
            if not isinstance(member_path, str) or not member_path.startswith(
                f"{board_root.rstrip('/')}/",
            ):
                raise ValidationError(
                    f"semantic projection member is outside {board_root!r}",
                )
            relative_path = member_path.removeprefix(board_root)
            for instance_path in instance_paths:
                if not isinstance(instance_path, str) or not instance_path.startswith(
                    "/",
                ):
                    raise ValidationError(
                        "semantic projection instance paths must be absolute",
                    )
                expanded[f"{instance_path.rstrip('/')}{relative_path}"] = selected
    return expanded


def _semantics_for_prim_path(
    prim_path: str,
    semantics_by_component: Mapping[str, dict[str, str]],
) -> dict[str, str] | None:
    """Return semantics inherited from the nearest component ancestor."""
    candidate = prim_path.rstrip("/")
    while candidate:
        semantics = semantics_by_component.get(candidate)
        if semantics is not None:
            return semantics
        candidate, separator, _ = candidate.rpartition("/")
        if not separator:
            break
    return None


def _load_source_ids(image_path: Path, *, colorized: bool):
    """Load raw or RGBA instance IDs into a uint32 array."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise ValidationError(
            "instance semantic projection requires NumPy and Pillow",
        ) from error

    with Image.open(image_path) as image:
        if not colorized:
            return np.asarray(image, dtype=np.uint32).copy()
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return (
        rgba[..., 0].astype(np.uint32)
        | rgba[..., 1].astype(np.uint32) << 8
        | rgba[..., 2].astype(np.uint32) << 16
        | rgba[..., 3].astype(np.uint32) << 24
    )


def _validate_source_ids(
    source_ids,
    source_to_output: Mapping[int, int],
    image_path: Path,
    *,
    colorized: bool,
) -> None:
    """Reject instance-ID frames that cannot carry the mapping's identifiers.

    Both failures below yield a plausible-looking but wrong mask rather than an
    error, which is the worst outcome for ground truth: an empty frame reads as
    "nothing is here" and a saturated identifier silently relabels one instance
    as another.

    Args:
        source_ids: Instance identifiers decoded from the frame.
        source_to_output: Identifiers that map to a semantic class.
        image_path: Frame being validated, for the error message.
        colorized: Whether identifiers were decoded from RGBA.

    Raises:
        ValidationError: If the frame is empty, or raw identifiers exceed what
            its pixel format can represent.

    """
    import numpy as np

    if source_to_output and not source_ids.any():
        raise ValidationError(
            f"instance-ID frame {image_path.name} is empty while its mapping "
            f"carries {len(source_to_output)} labeled instances; the instance "
            "segmentation annotator produced no identifiers for this frame",
        )
    if colorized or not source_to_output:
        return
    representable = int(np.iinfo(source_ids.dtype).max)
    largest = max(source_to_output)
    if largest > representable:
        raise ValidationError(
            f"instance-ID frame {image_path.name} stores identifiers as "
            f"{source_ids.dtype}, which cannot represent identifier {largest}; "
            "enable colorize_instance_id_segmentation so identifiers are "
            "written as RGBA",
        )


def _remap_source_ids(source_ids, source_to_output: Mapping[int, int]):
    """Map instance IDs to semantic IDs with vectorized sorted lookup."""
    import numpy as np

    output_ids = np.where(source_ids == 0, 0, 1).astype(np.uint32)
    if not source_to_output:
        return output_ids
    source_keys = np.asarray(sorted(source_to_output), dtype=np.uint32)
    output_values = np.asarray(
        [source_to_output[int(key)] for key in source_keys],
        dtype=np.uint32,
    )
    indices = np.searchsorted(source_keys, source_ids)
    matches = indices < len(source_keys)
    matches &= source_keys[np.minimum(indices, len(source_keys) - 1)] == source_ids
    output_ids[matches] = output_values[indices[matches]]
    return output_ids


def _write_semantic_image(
    output_path: Path,
    output_ids,
    semantics_by_id: Mapping[int, Mapping[str, str]],
    *,
    colorized: bool,
) -> dict[str, dict[str, str]]:
    """Write one BasicWriter-compatible semantic PNG and label mapping."""
    import numpy as np
    from PIL import Image

    if not colorized:
        if int(output_ids.max()) > 65535:
            raise ValidationError(
                "raw semantic projection exceeds PNG's 16-bit identifier limit",
            )
        Image.fromarray(output_ids.astype(np.uint16)).save(output_path)
        labels = {"0": _BACKGROUND, "1": _UNLABELLED}
        labels.update(
            {
                str(identifier): dict(semantics)
                for identifier, semantics in sorted(semantics_by_id.items())
            },
        )
        return labels

    max_identifier = max((1, *semantics_by_id))
    palette = np.zeros((max_identifier + 1, 4), dtype=np.uint8)
    palette[1] = (0, 0, 0, 255)
    labels = {
        "(0, 0, 0, 0)": _BACKGROUND,
        "(0, 0, 0, 255)": _UNLABELLED,
    }
    for identifier, semantics in sorted(semantics_by_id.items()):
        color = semantic_color(semantics)
        palette[identifier] = color
        labels[str(color)] = dict(semantics)
    Image.fromarray(palette[output_ids]).save(output_path)
    return labels


def project_instance_semantics(
    output_dir: str | Path,
    projection: Mapping[str, object],
    *,
    semantic_types: Sequence[str] = ("class",),
    colorized: bool = True,
    remove_instance_artifacts: bool = False,
) -> tuple[dict[str, object], ...]:
    """Create semantic frames from BasicWriter instance-ID path mappings.

    Use this function for the post-capture phase of a validated native-instance
    panel. The projection records must already have been built from the selected
    ``BoardMetadataCatalog`` and expanded through the registered board cohort.
    This function does not open the USD stage, infer instances, or choose the
    annotation strategy. Standalone boards should use BasicWriter's native
    semantic output instead.

    Args:
        output_dir: BasicWriter directory containing instance-ID artifacts.
        projection: Catalog-derived component and board-instance mapping.
        semantic_types: Semantic namespaces retained in generated labels.
        colorized: Whether to write RGBA colors instead of raw uint32 IDs.
        remove_instance_artifacts: Whether intermediate instance artifacts are
            deleted after successful projection.

    Returns:
        Generated semantic image and label artifact records.

    Raises:
        ValidationError: If required mappings, images, or projection metadata
            are unavailable.

    Example:

    .. code-block:: python

        artifacts = project_instance_semantics(
            "output/trigger_0000",
            projection,
        )
        print(artifacts[0]["path"])

    """
    root = Path(output_dir)
    selected_types = frozenset(str(value) for value in semantic_types)
    if not selected_types:
        raise ValidationError("semantic projection requires semantic_types")
    semantics_by_component = _component_path_semantics(
        projection,
        selected_types,
    )
    mapping_paths = sorted(root.glob("instance_id_segmentation_mapping_*.json"))
    if not mapping_paths:
        raise ValidationError(
            f"semantic projection found no instance-ID mappings in {root}",
        )

    artifacts = []
    cleanup_paths = []
    for mapping_path in mapping_paths:
        suffix = mapping_path.stem.removeprefix(
            "instance_id_segmentation_mapping_",
        )
        image_path = root / f"instance_id_segmentation_{suffix}.png"
        if not image_path.is_file():
            raise ValidationError(
                f"semantic projection is missing {image_path.name}",
            )
        try:
            instance_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValidationError(
                f"could not read instance-ID mapping {mapping_path}: {error}",
            ) from error
        if not isinstance(instance_mapping, dict):
            raise ValidationError(
                f"instance-ID mapping must be an object: {mapping_path}",
            )

        source_semantics = {}
        mapping_is_colorized = False
        for raw_identifier, prim_path in instance_mapping.items():
            if not isinstance(prim_path, str):
                continue
            identifier_text = str(raw_identifier)
            mapping_is_colorized |= identifier_text.strip().startswith("(")
            semantics = _semantics_for_prim_path(
                prim_path,
                semantics_by_component,
            )
            if semantics is not None:
                source_semantics[_parse_mapping_id(identifier_text)] = semantics

        semantic_groups = sorted(
            {tuple(sorted(values.items())) for values in source_semantics.values()},
        )
        group_ids = {
            group: identifier for identifier, group in enumerate(semantic_groups, 2)
        }
        source_to_output = {
            source_identifier: group_ids[tuple(sorted(semantics.items()))]
            for source_identifier, semantics in source_semantics.items()
        }
        semantics_by_id = {
            identifier: dict(group) for group, identifier in group_ids.items()
        }
        source_ids = _load_source_ids(
            image_path,
            colorized=mapping_is_colorized,
        )
        _validate_source_ids(
            source_ids,
            source_to_output,
            image_path,
            colorized=mapping_is_colorized,
        )
        output_ids = _remap_source_ids(source_ids, source_to_output)
        semantic_path = root / f"semantic_segmentation_{suffix}.png"
        labels = _write_semantic_image(
            semantic_path,
            output_ids,
            semantics_by_id,
            colorized=colorized,
        )
        labels_path = root / f"semantic_segmentation_labels_{suffix}.json"
        labels_path.write_text(
            json.dumps(labels, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts.append(
            {
                "frame": int(suffix.rsplit("_", 1)[-1]),
                "path": str(semantic_path),
                "labels_path": str(labels_path),
                "semantic_groups": len(semantic_groups),
            },
        )
        cleanup_paths.extend((image_path, mapping_path))

    if remove_instance_artifacts:
        for path in cleanup_paths:
            path.unlink()
    return tuple(artifacts)
