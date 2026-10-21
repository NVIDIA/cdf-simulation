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

"""Find through-hole leads on a board and pair each with its own barrel.

**The board metadata catalog is the source of truth wherever it has an answer.**
Reading prim attributes directly is double maintenance against a moving USD
schema, and this module has already paid for that once: it read the previous
``pcb:`` namespace and silently found nothing on a board authoring ``pcba:``.

What the catalog answers, and what it does not, on a panelised THT board:

===========================  ===========================================
from the catalog             refdes and package type (``ComponentRecord``),
                             terminal number (``TerminalRecord``), plating
                             and the authored terminal-to-hole association
                             (``HoleRecord``)
still read from the prim     the meshes' POINTS, which are geometry rather
                             than metadata; the bore radius, because
                             ``HoleRecord`` carries no radius field and the
                             board authors no ``holeDimsMm``; and the lead
                             type, because ``TerminalRecord.process_class``
                             may be unset on every terminal
===========================  ===========================================

The authored association may cover only part of a board's terminals, so the
geometric pairing stays as a fallback rather than being deleted -- dropping it
would silently stop varying every lead without an association. Every lead
records which route found its barrel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..geometry.transform import transform_points
from pcb_aoi.geometry.tht_pin_pose import Lead

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Fraction of a barrel's height a lead must span to count as passing through
#: it. This is an OVERLAP FRACTION and not a strict containment test: the
#: plated-barrel mesh includes its annular caps and so stands slightly proud of
#: the board on each face, while leads stop flush at the board surface.
#: Requiring the lead to exceed the barrel at both ends rejects every genuine
#: pin on a real export.
COVERAGE_FRACTION = 0.9

#: Slack, in millimetres, added to a bore radius when matching a lead to the
#: nearest hole. Wide enough for placement tolerance, narrow enough that a
#: surface-mount termination sitting beside a via does not claim it.
MATCH_SLACK_MM = 0.5

#: Attribute names per field, canonical first. The current board schema is the
#: ``pcba:`` namespace, which is what the metadata catalog and
#: `annotations.label_geometry` already read. The ``pcb:`` names are an EARLIER
#: export of the same board and are kept as a fallback rather than a synonym:
#: reading only those found nothing at all on the current panel while looking
#: exactly like a board that carries no schema.
_KEYS = {
    "role": ("pcba:mesh:role", "pcb:mesh:role"),
    "pin": ("pcba:pin:number", "pcb:pin:number"),
    "hole_radius": ("pcba:board:holeRadiusMm", "pcb:hole:radiusMm"),
    "plated": ("pcba:board:plated", "pcb:hole:plated"),
    "package_type": ("pcba:component:packageType", "pcb:component:packageType"),
    "process_class": ("pcba:component:leadType", "pcb:component:processClass"),
    "refdes": (
        "pcba:component:refdes",
        "pcba:board:refdes",
        "pcb:component:refdes",
    ),
}

_LEAD_ROLE = "pkg_lead"
#: Mesh roles that carry a drilled, plated bore this module can pair a lead to.
#:
#: `board_hole` came first and is what `pcb_aoi.hole_plane` still authors.
#: `board_barrel` is what the input contract settled on: `board_hole` was
#: deliberately removed by the current schema ("a hole is an absence of
#: material"), leaving the plating mesh as the only thing a board authors for a
#: through hole. Boards exported against the current contract therefore carry
#: no `board_hole` at all, and every lead on them was being rejected.
#:
#: Both name the same geometry -- this module's own docstrings already call it
#: "the barrel mesh" -- so both are accepted. The radius is read from
#: `hole_radius` either way, which matters: a barrel mesh's XY extent is the
#: OUTSIDE of the copper, not the bore.
_HOLE_ROLES = ("board_hole", "board_barrel")


@dataclass(frozen=True)
class CatalogIndex:
    """What the board metadata catalog can say about leads and barrels.

    Attributes:
        terminal_by_path: Terminal records keyed by BOARD-RELATIVE prim path.
        component_by_path: Component records keyed by board-relative path.
        hole_by_path: Hole records keyed by board-relative path.
        hole_path_by_terminal: Board-relative hole path keyed by
            (board-relative component path, terminal number).

    Paths are relative to their own board root because the two sides disagree
    on the prefix: the catalog resolves a panel through its instance proxies
    (``.../StepRepeats/SR_0_0/Components/<refdes>/pkg_lead_1``) while discovery
    works on the prototype the placements share
    (``/World/_class_Boards/<board>/Components/<refdes>/pkg_lead_1``). Everything
    after the board root is identical, and that is what is keyed on.
    """

    terminal_by_path: dict[str, object] = field(default_factory=dict)
    component_by_path: dict[str, object] = field(default_factory=dict)
    hole_by_path: dict[str, object] = field(default_factory=dict)
    hole_path_by_terminal: dict[tuple[str, str], str] = field(
        default_factory=dict,
    )

    def __bool__(self) -> bool:
        """Report whether the index has anything to contribute.

        Returns:
            Whether any record was indexed.
        """
        return bool(self.terminal_by_path or self.hole_by_path)


def index_catalogs(catalogs: Sequence[object]) -> CatalogIndex:
    """Index every supplied catalog snapshot by prim path.

    A panel is loaded per board placement -- a whole-panel load raises on
    duplicate component identity, because two step-repeats of one board carry
    the same refdes -- so several snapshots are expected and merged here.

    Args:
        catalogs: Catalog snapshots, typically one per attached source.

    Returns:
        The merged index.
    """
    index = CatalogIndex()
    for catalog in catalogs:
        for terminal in getattr(catalog, "terminals", ()):
            index.terminal_by_path[_relative(terminal)] = terminal
        for component in getattr(catalog, "components", ()):
            index.component_by_path[_relative(component)] = component
        for hole in getattr(catalog, "holes", ()):
            index.hole_by_path[_relative(hole)] = hole
            if hole.component_path and hole.terminal_number is not None:
                index.hole_path_by_terminal[
                    (
                        _strip(hole.component_path, hole.board_root),
                        str(hole.terminal_number),
                    )
                ] = _relative(hole)
    return index


def _strip(path: str, board_root: str) -> str:
    """Return a prim path relative to its board root.

    Args:
        path: Absolute prim path.
        board_root: Board root the path sits under.

    Returns:
        The path with its board-root prefix removed.
    """
    return path[len(board_root):] if path.startswith(board_root) else path


def _relative(record: object) -> str:
    """Return one catalog record's board-relative prim path.

    Args:
        record: Any catalog record carrying ``prim_path`` and ``board_root``.

    Returns:
        The board-relative path.
    """
    return _strip(record.prim_path, record.board_root)


def _lookup(table: dict[str, object], path: str) -> object | None:
    """Resolve an absolute prim path against board-relative keys.

    Tries the path's suffixes longest-first, so a prototype path matches the
    instance-proxy path the catalog recorded without either side having to
    know the other's board root.

    Args:
        table: Board-relative keys.
        path: Absolute prim path.

    Returns:
        The record, or None.
    """
    segments = path.split("/")
    for start in range(1, len(segments)):
        found = table.get("/" + "/".join(segments[start:]))
        if found is not None:
            return found
    return None


@dataclass(frozen=True)
class Hole:
    """One drilled and plated barrel.

    Attributes:
        path: Prim path of the barrel mesh.
        center: Bore centre in XY.
        radius_mm: Drilled bore radius.
        z_range: Lower and upper extent of the barrel mesh.
        plated: Whether the hole carries plating.
        component_path: Owning component, when the board states one.
        terminal_number: Terminal this hole belongs to, when stated.
    """

    path: str
    center: tuple[float, float]
    radius_mm: float
    z_range: tuple[float, float]
    plated: bool = True
    component_path: str | None = None
    terminal_number: str | None = None


@dataclass(frozen=True)
class DiscoveredLead:
    """One through-hole lead, its identity and the barrel it passes through.

    Attributes:
        path: Prim path of the lead mesh.
        refdes: Reference designator of the owning component.
        pin: Pin number, when the board states one.
        package_type: Package type of the owning component, when stated.
        process_class: Manufacturing process class, when stated.
        points: World-space lead vertices.
        hole: The barrel this lead runs through.
        world_to_local: Inverse of the lead's local-to-world transform.
        pairing_source: ``"catalog"`` when the board authored the
            terminal-to-hole association, ``"geometry"`` when it was inferred.
    """

    path: str
    refdes: str
    pin: object
    package_type: str | None
    process_class: str | None
    points: np.ndarray = field(repr=False)
    hole: Hole | None = None
    world_to_local: object = field(default=None, repr=False)
    pairing_source: str = "geometry"

    def as_pose_lead(self, board_far_face_z: float) -> Lead:
        """Return the geometry-module view of this lead.

        Args:
            board_far_face_z: Board face the protrusion is measured from.

        Returns:
            A `Lead` carrying only what the pose maths needs.

        Raises:
            ValueError: If the lead was never paired with a barrel.
        """
        if self.hole is None:
            message = f"lead {self.path!r} has no barrel to be posed inside"
            raise ValueError(message)
        return Lead(
            points=self.points,
            hole_center=self.hole.center,
            hole_radius_mm=self.hole.radius_mm,
            barrel_z=self.hole.z_range,
            board_far_face_z=board_far_face_z,
            tip_is_low=True,
        )


def _attribute(prim: object, field: str, default: object = None) -> object:
    """Return one authored value for a field, trying each known name.

    Args:
        prim: Prim to read.
        field: Key in `_KEYS`.
        default: Value to use when no name is authored.

    Returns:
        The value, or the default.
    """
    for name in _KEYS[field]:
        attribute = prim.GetAttribute(name)
        if attribute:
            value = attribute.Get()
            if value is not None:
                return value
    return default


def scan_board(
    stage: object,
    index: CatalogIndex | None = None,
) -> tuple[list[DiscoveredLead], list[Hole]]:
    """Return every through-hole lead and barrel the board declares.

    Uses ``TraverseAll`` rather than ``Traverse``. A panelised export keeps its
    board geometry in an **abstract class prototype** behind instanceable
    step-repeats, and the default predicate skips abstract prims -- a plain
    traversal finds a few dozen meshes out of tens of thousands and reports
    success, which is the worst possible failure mode for a discovery step.

    ``TraverseAll`` also visits the prototype itself, so a lead that is BOTH
    authored in a class and composed into a placement is seen twice. Where a
    concrete placement exists it is the one that gets varied: authoring on the
    prototype would give every placement on a panel the same variation, which
    is a rig rather than a randomisation. When the geometry is reachable only
    through the prototype -- a genuinely ``instanceable`` reference, whose
    proxies this traversal does not descend into -- the prototype is used and
    that sharing is unavoidable.

    Args:
        stage: Open USD stage.
        index: Catalog index consulted before the prim's own attributes.

    Returns:
        The leads found and the barrels found, unpaired.
    """
    from pxr import UsdGeom

    index = index or CatalogIndex()

    cache = UsdGeom.XformCache()
    leads: list[DiscoveredLead] = []
    abstract_leads: list[DiscoveredLead] = []
    holes: list[Hole] = []

    for prim in stage.TraverseAll():
        role = _attribute(prim, "role")
        if role != _LEAD_ROLE and role not in _HOLE_ROLES:
            continue
        raw_points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not raw_points:
            continue
        matrix = cache.GetLocalToWorldTransform(prim)
        points = transform_points(matrix, raw_points)

        if role in _HOLE_ROLES:
            # `pcb_aoi.hole_plane` authors a single flat quad per side that
            # also carries `board_hole`; it is a segmentation aid seen through
            # the drilled voids, not a bore. Taking it as a hole would give a
            # radius of half the board and swallow every lead. A bore is a
            # tube -- the barrels on a real board carry 288 points -- so the
            # quad's four are the discriminator.
            if len(points) < 8:
                continue
            low, high = points.min(0), points.max(0)
            # The barrel MESH is the plating including its annular caps, so its
            # XY extent is the outside of the copper and not the bore. The
            # drilled radius is stated on the prim; take only the centre from
            # the mesh.
            stated = _attribute(prim, "hole_radius")
            record = _lookup(index.hole_by_path, str(prim.GetPath()))
            holes.append(
                Hole(
                    path=str(prim.GetPath()),
                    center=(
                        0.5 * float(low[0] + high[0]),
                        0.5 * float(low[1] + high[1]),
                    ),
                    radius_mm=(
                        float(stated)
                        if stated is not None
                        else 0.5 * float(high[0] - low[0])
                    ),
                    z_range=(float(low[2]), float(high[2])),
                    # The catalog owns plating and the association; the RADIUS
                    # is still read here because HoleRecord carries no radius
                    # field and this board authors no holeDimsMm.
                    plated=bool(
                        record.plated
                        if record is not None and record.plated is not None
                        else _attribute(prim, "plated", True),
                    ),
                    component_path=(
                        record.component_path if record is not None else None
                    ),
                    terminal_number=(
                        str(record.terminal_number)
                        if record is not None and record.terminal_number is not None
                        else None
                    ),
                ),
            )
            continue

        component = prim.GetParent()
        path = str(prim.GetPath())
        terminal = _lookup(index.terminal_by_path, path)
        record = (
            _lookup(index.component_by_path, terminal.component_path)
            if terminal is not None
            else None
        )
        found = (abstract_leads if prim.IsAbstract() else leads)
        found.append(
            DiscoveredLead(
                path=str(prim.GetPath()),
                # Catalog first. The prim fallbacks are what a board without a
                # catalog still resolves by, and what covers the fields the
                # catalog does not carry -- notably the lead type, which is
                # unset on every TerminalRecord of some panelised THT boards.
                #
                # The refdes IS authored (`pcba:component:refdes`); the prim
                # name is only a last resort. The two usually match, which is
                # why reading the wrong key can still produce the right answer
                # and hide the mistake.
                refdes=str(
                    getattr(record, "refdes", None)
                    or _attribute(component, "refdes", component.GetName()),
                ),
                pin=(
                    terminal.number
                    if terminal is not None and terminal.number is not None
                    else _attribute(prim, "pin")
                ),
                package_type=(
                    getattr(record, "package_type", None)
                    or _attribute(component, "package_type")
                ),
                process_class=(
                    getattr(terminal, "process_class", None)
                    or _attribute(component, "process_class")
                ),
                points=points,
                world_to_local=matrix.GetInverse(),
            ),
        )
    return (leads or abstract_leads), holes


def _passes_through(lead: DiscoveredLead, hole: Hole) -> bool:
    """Report whether a lead physically runs through one barrel.

    Args:
        lead: Lead to test.
        hole: Candidate barrel.

    Returns:
        Whether the lead is near enough and spans enough of the barrel.
    """
    centroid = lead.points[:, :2].mean(0)
    distance = float(
        np.hypot(hole.center[0] - centroid[0], hole.center[1] - centroid[1]),
    )
    low, high = float(lead.points[:, 2].min()), float(lead.points[:, 2].max())
    span = max(hole.z_range[1] - hole.z_range[0], 1e-9)
    coverage = (min(high, hole.z_range[1]) - max(low, hole.z_range[0])) / span
    return (
        distance <= hole.radius_mm + MATCH_SLACK_MM
        and coverage >= COVERAGE_FRACTION
    )


def pair_leads(
    leads: Sequence[DiscoveredLead],
    holes: Sequence[Hole],
    index: CatalogIndex | None = None,
) -> tuple[list[DiscoveredLead], list[tuple[DiscoveredLead, float, float]]]:
    """Attach each lead to the barrel it passes through.

    **The board's own answer wins, but not against physics.** Where a hole
    states the component and terminal it belongs to, that association is used
    -- after checking the lead actually passes through it. On a panelised THT
    board, leads can be authored against barrels their own geometry sits
    millimetres away from, which no pose can be solved inside; those fall back
    to geometry and are reported as disagreements rather than believed or
    silently dropped.

    Where no association is authored, the lead is matched to the nearest
    **plated** bore it runs the length of. That span test is what separates a
    through-hole lead from a surface-mount termination sitting next to a via.
    Deleting the fallback in favour of the catalog alone would silently stop
    varying every lead without an association.

    Args:
        leads: Leads to pair.
        holes: Candidate barrels.
        index: Catalog index carrying any authored associations.

    Returns:
        The paired leads, and the rejected ones with their distance and
        coverage so a miss can be diagnosed rather than merely counted.
        Authored associations the geometry contradicts are logged on
        ``pair_leads.disagreements`` for the caller to report.
    """
    index = index or CatalogIndex()
    hole_by_path = {hole.path: hole for hole in holes}
    disagreements: list[tuple[str, str, float]] = []
    plated = [hole for hole in holes if hole.plated]
    if not plated or not leads:
        return [], [(lead, float("inf"), 0.0) for lead in leads]

    centres = np.array([hole.center for hole in plated], float)
    paired: list[DiscoveredLead] = []
    rejected: list[tuple[DiscoveredLead, float, float]] = []

    for lead in leads:
        # Authored association first.
        authored = None
        if lead.pin is not None:
            terminal = _lookup(index.terminal_by_path, lead.path)
            if terminal is not None and terminal.component_path:
                relative = index.hole_path_by_terminal.get(
                    (
                        _strip(terminal.component_path, terminal.board_root),
                        str(lead.pin),
                    ),
                )
                if relative:
                    authored = next(
                        (h for h in holes if h.path.endswith(relative)), None,
                    )
        if authored is not None and not _passes_through(lead, authored):
            # The board named a barrel the lead is nowhere near. Believing it
            # would place the lead in a hole it does not occupy; dropping the
            # lead would lose a pin the board plainly has. Fall back and say so.
            centroid = lead.points[:, :2].mean(0)
            disagreements.append(
                (
                    lead.path,
                    authored.path,
                    float(
                        np.hypot(
                            authored.center[0] - centroid[0],
                            authored.center[1] - centroid[1],
                        ),
                    ),
                ),
            )
            authored = None
        if authored is not None:
            paired.append(
                DiscoveredLead(
                    path=lead.path,
                    refdes=lead.refdes,
                    pin=lead.pin,
                    package_type=lead.package_type,
                    process_class=lead.process_class,
                    points=lead.points,
                    hole=authored,
                    world_to_local=lead.world_to_local,
                    pairing_source="catalog",
                ),
            )
            continue

        centroid = lead.points[:, :2].mean(0)
        distances = np.hypot(
            centres[:, 0] - centroid[0], centres[:, 1] - centroid[1],
        )
        nearest = int(np.argmin(distances))
        hole = plated[nearest]
        distance = float(distances[nearest])
        low, high = float(lead.points[:, 2].min()), float(lead.points[:, 2].max())
        span = max(hole.z_range[1] - hole.z_range[0], 1e-9)
        coverage = (
            min(high, hole.z_range[1]) - max(low, hole.z_range[0])
        ) / span
        if distance > hole.radius_mm + MATCH_SLACK_MM or coverage < COVERAGE_FRACTION:
            rejected.append((lead, distance, float(coverage)))
            continue
        paired.append(
            DiscoveredLead(
                path=lead.path,
                refdes=lead.refdes,
                pin=lead.pin,
                package_type=lead.package_type,
                process_class=lead.process_class,
                points=lead.points,
                hole=hole,
                world_to_local=lead.world_to_local,
                pairing_source="geometry",
            ),
        )
    pair_leads.disagreements = tuple(disagreements)
    return paired, rejected


def discover_through_hole_leads(
    stage: object,
    catalogs: Sequence[object] = (),
) -> tuple[list[DiscoveredLead], list[tuple[DiscoveredLead, float, float]]]:
    """Return the board's through-hole leads, each paired with its barrel.

    Args:
        stage: Open USD stage.
        catalogs: Board metadata catalog snapshots the job has already loaded.
            Supplying none is supported and falls back to the prims entirely.

    Returns:
        The paired leads and the rejected candidates.
    """
    index = index_catalogs(catalogs)
    leads, holes = scan_board(stage, index)
    return pair_leads(leads, holes, index)
