"""
region_manager.py
=================
Responsible for:
  - Partitioning the map into 20x20 regions
  - Assigning agents to regions and roles at startup
  - Tracking per-region worker rosters and availability
  - Managing cross-region reallocation requests (facilitated via global leader)

Roles
-----
  ROLE_GLOBAL_LEADER   – exactly 1, manages cross-region comms & reallocation
  ROLE_REGIONAL_LEADER – 1 per region (static, position-fixed conceptually)
  ROLE_WORKER          – all other agents

Under-resourced fallback
------------------------
  If total_agents < target_size (i.e. fewer agents than needed to fully staff
  all regions at MIN_WORKERS_PER_REGION), active regions are reduced:
    num_active_regions = max(1, total_workers // MIN_WORKERS_PER_REGION)
  Remaining grid area is merged into active regions proportionally.
  The three-tier hierarchy is preserved — only the number of active regions shrinks.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

from crew_algorithms.wildfire_alg.algorithms.HMAS_3.agent import (
    Agent,
    ROLE_WORKER,
    ROLE_REGIONAL_LEADER,
    ROLE_GLOBAL_LEADER,
)

REGION_SIZE = 20          # each region covers REGION_SIZE × REGION_SIZE cells
MIN_WORKERS_PER_REGION = 2  # minimum workers to warrant an active region


# ---------------------------------------------------------------------------
# Region descriptor
# ---------------------------------------------------------------------------

class Region:
    """Represents a 20×20 grid region."""

    def __init__(self, region_id: int, x_min: int, y_min: int, x_max: int, y_max: int):
        self.region_id = region_id
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max  # exclusive upper bound
        self.y_max = y_max  # exclusive upper bound

        self.leader: Optional[Agent] = None          # the regional leader agent
        self.workers: List[Agent] = []               # current workers (may include temps)
        self.pending_request: bool = False           # True if this region has requested help
        self.active: bool = True                     # False if merged into a neighbour

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x_min + self.x_max) // 2, (self.y_min + self.y_max) // 2)

    def contains(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return self.x_min <= x < self.x_max and self.y_min <= y < self.y_max

    def free_worker_count(self) -> int:
        return len(self.workers)

    def __repr__(self) -> str:
        return (
            f"Region(id={self.region_id}, "
            f"x=[{self.x_min},{self.x_max}), y=[{self.y_min},{self.y_max}), "
            f"active={self.active}, workers={len(self.workers)})"
        )


# ---------------------------------------------------------------------------
# Reallocation request
# ---------------------------------------------------------------------------

class ReallocRequest:
    """Sent by a regional leader to the global leader when it has fires but no workers."""

    def __init__(self, from_region_id: int, urgency: str, fire_cells: List[Tuple[int, int]], workers_needed: int = 1):
        self.from_region_id = from_region_id
        self.urgency = urgency           # "HIGH" | "MEDIUM" | "LOW"
        self.fire_cells = fire_cells
        self.workers_needed = workers_needed
        self.fulfilled = False

    def __repr__(self) -> str:
        return (
            f"ReallocRequest(from={self.from_region_id}, urgency={self.urgency}, "
            f"need={self.workers_needed}, fulfilled={self.fulfilled})"
        )


# ---------------------------------------------------------------------------
# Region Manager
# ---------------------------------------------------------------------------

class RegionManager:
    """
    Central data structure used by the global leader to manage regions.

    Instantiated once in __main__.py and stored inside global_data.
    """

    def __init__(self, map_size: int, agents: List[Agent]):
        self.map_size = map_size
        self.regions: Dict[int, Region] = {}
        self.pending_requests: List[ReallocRequest] = []  # priority queue (simple list, sorted on insert)
        self._build_regions()
        self._assign_agents(agents)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_regions(self):
        """Tile the map into REGION_SIZE × REGION_SIZE cells."""
        rid = 0
        y = 0
        while y < self.map_size:
            x = 0
            while x < self.map_size:
                x_max = min(x + REGION_SIZE, self.map_size)
                y_max = min(y + REGION_SIZE, self.map_size)
                self.regions[rid] = Region(rid, x, y, x_max, y_max)
                rid += 1
                x += REGION_SIZE
            y += REGION_SIZE
        print(f"[RegionManager] Created {len(self.regions)} regions for {self.map_size}×{self.map_size} map.")

    def _assign_agents(self, agents: List[Agent]):
        """
        Assign roles and regions.

        Priority:
          1. One global leader (first agent in list, any type — typically a drone for wide view).
          2. One regional leader per active region.
          3. Remaining agents become workers distributed round-robin across active regions.

        Under-resourced fallback: if workers are too few, deactivate low-priority regions
        and merge their area responsibility into adjacent active regions.
        """
        total = len(agents)
        num_regions = len(self.regions)

        # Need 1 global leader + 1 regional leader per region + MIN_WORKERS_PER_REGION workers per region
        target_size = 1 + num_regions * (1 + MIN_WORKERS_PER_REGION)
        workers_available = total - 1 - num_regions  # after reserving global + regional leaders

        if workers_available < 0:
            # Not even enough for global + all regional leaders — collapse regions
            num_active = max(1, (total - 1))  # 1 global + remaining as regional leaders
        elif workers_available < num_regions * MIN_WORKERS_PER_REGION:
            # Under-resourced: shrink active regions
            num_active = max(1, workers_available // MIN_WORKERS_PER_REGION)
            print(
                f"[RegionManager] UNDER-RESOURCED: {total} agents < target {target_size}. "
                f"Activating {num_active}/{num_regions} regions (degraded hierarchy)."
            )
        else:
            num_active = num_regions

        # Deactivate low-priority regions (those with highest IDs — least central)
        sorted_region_ids = sorted(self.regions.keys())
        active_ids = sorted_region_ids[:num_active]
        for rid in sorted_region_ids[num_active:]:
            self.regions[rid].active = False

        agent_iter = iter(agents)

        # --- Global leader ---
        global_agent = next(agent_iter)
        global_agent.role = ROLE_GLOBAL_LEADER
        global_agent.region_id = None
        global_agent.home_region_id = None
        print(f"[RegionManager] AGENT_{global_agent.id} → GLOBAL_LEADER")

        # --- Regional leaders ---
        regional_leaders: List[Agent] = []
        for rid in active_ids:
            try:
                rl = next(agent_iter)
            except StopIteration:
                break
            rl.role = ROLE_REGIONAL_LEADER
            rl.region_id = rid
            rl.home_region_id = rid
            self.regions[rid].leader = rl
            regional_leaders.append(rl)
            print(f"[RegionManager] AGENT_{rl.id} → REGIONAL_LEADER of region {rid} {self.regions[rid]}")

        # --- Workers: round-robin across active regions ---
        active_regions_with_leaders = [r for r in active_ids if self.regions[r].leader is not None]
        rr_idx = 0
        for agent in agent_iter:
            if not active_regions_with_leaders:
                break
            rid = active_regions_with_leaders[rr_idx % len(active_regions_with_leaders)]
            agent.role = ROLE_WORKER
            agent.region_id = rid
            agent.home_region_id = rid
            self.regions[rid].workers.append(agent)
            print(f"[RegionManager] AGENT_{agent.id} → WORKER in region {rid}")
            rr_idx += 1

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------

    def get_region_for_position(self, pos: Tuple[int, int]) -> Optional[Region]:
        """Return the region (active or inactive) that contains pos."""
        for region in self.regions.values():
            if region.contains(pos):
                return region
        return None

    def get_effective_region(self, pos: Tuple[int, int]) -> Optional[Region]:
        """
        Return the *active* region responsible for pos.
        If the containing region is inactive, return the nearest active region.
        """
        containing = self.get_region_for_position(pos)
        if containing and containing.active:
            return containing
        # Find nearest active region by center distance
        best: Optional[Region] = None
        best_dist = float("inf")
        for region in self.regions.values():
            if not region.active:
                continue
            cx, cy = region.center
            dist = abs(cx - pos[0]) + abs(cy - pos[1])
            if dist < best_dist:
                best_dist = dist
                best = region
        return best

    def workers_in_region(self, region_id: int) -> List[Agent]:
        return self.regions[region_id].workers

    def has_fire_context(self, region_id: int, global_data: dict) -> bool:
        """
        Heuristic: check if any worker perception in a region mentions fire.
        Used by regional leader to decide whether to request reallocation.
        """
        for w in self.regions[region_id].workers:
            if w.last_perception and ("fire" in w.last_perception.lower() or "ignited" in w.last_perception.lower()):
                return True
        return False

    # ------------------------------------------------------------------
    # Reallocation
    # ------------------------------------------------------------------

    def submit_reallocation_request(self, request: ReallocRequest):
        """Regional leader submits a help request to the global leader."""
        urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        self.pending_requests.append(request)
        self.pending_requests.sort(key=lambda r: urgency_order.get(r.urgency, 3))
        self.regions[request.from_region_id].pending_request = True
        print(f"[RegionManager] Realloc request queued: {request}")

    def resolve_reallocation_requests(self) -> List[Tuple[Agent, int, int]]:
        """
        Global leader resolves pending requests.
        Returns list of (worker_agent, from_region_id, to_region_id) moves to execute.
        Donor selection: region with most surplus workers AND lowest pending urgency.
        """
        moves: List[Tuple[Agent, int, int]] = []
        still_pending: List[ReallocRequest] = []

        for req in self.pending_requests:
            if req.fulfilled:
                continue
            target_region = self.regions[req.from_region_id]
            donated = 0

            # Find donor regions (active, not themselves requesting, has surplus)
            donors = sorted(
                [
                    r for r in self.regions.values()
                    if r.active
                    and r.region_id != req.from_region_id
                    and not r.pending_request
                    and len(r.workers) > MIN_WORKERS_PER_REGION
                ],
                key=lambda r: -len(r.workers),  # most surplus first
            )

            for donor in donors:
                if donated >= req.workers_needed:
                    break
                # Take the last worker (least recently assigned)
                worker = donor.workers.pop()
                old_region = worker.region_id
                worker.region_id = req.from_region_id  # temporary reassignment
                target_region.workers.append(worker)
                moves.append((worker, old_region, req.from_region_id))
                donated += 1
                print(
                    f"[RegionManager] Global leader reallocates AGENT_{worker.id} "
                    f"from region {old_region} → region {req.from_region_id}"
                )

            if donated >= req.workers_needed:
                req.fulfilled = True
                target_region.pending_request = False
            else:
                still_pending.append(req)

        self.pending_requests = still_pending
        return moves

    def recall_worker(self, worker: Agent):
        """
        Called when a region's fire is suppressed — return worker to home region.
        """
        current_region = self.regions.get(worker.region_id)
        if current_region and worker in current_region.workers:
            current_region.workers.remove(worker)

        worker.region_id = worker.home_region_id
        if worker.home_region_id is not None:
            self.regions[worker.home_region_id].workers.append(worker)
            print(
                f"[RegionManager] AGENT_{worker.id} recalled to home region {worker.home_region_id}"
            )

    def summary(self) -> str:
        lines = ["=== RegionManager Summary ==="]
        for rid, region in self.regions.items():
            leader_id = region.leader.id if region.leader else "None"
            worker_ids = [w.id for w in region.workers]
            lines.append(
                f"  Region {rid} {'[ACTIVE]' if region.active else '[INACTIVE]'}: "
                f"leader=AGENT_{leader_id}, workers={worker_ids}"
            )
        return "\n".join(lines)