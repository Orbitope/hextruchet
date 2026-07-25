"""Batched tensor implementation of hex_truchet.

Written from spec.md ONLY — NOT from reference.py, and NOT reusing _hexcore's
loop-closure/area LOGIC. State lives in int64 tensors with a leading [N] dim;
branching is replaced by torch.where masking; the base class owns auto-reset
and RNG keying (self.keys, self.t).

Loop closure & enclosed area are reimplemented here as tensor ops (batched
connected-components via min-label propagation over the 222-port graph, then
ray-cast crossing parity for area). Only fixed *geometry lookup tables* (cell
adjacency, per-tile arc matchings, the ray structure) are derived at module
load from _hexcore's pure-geometry enumeration functions — never its Board /
union-find / enclosed_cells algorithms. The differential test validates this
independent reimplementation against the reference's union-find path.
"""

from __future__ import annotations

import numpy as np
import torch

from simulacrum import BatchedEnv, invariant, rng

from hex_truchet import (
    ACTION_SPACE_SIZE, DECK_TYPE0_COUNT, DECK_SIZE, HAND_SIZE, N_CELLS,
    N_PORTS_PER_CELL, N_REFILL_STEPS, N_ROTATIONS, ROT_NORM, SCORE_NORM, Slots,
    TILE_NORM,
)
from hex_truchet import _hexcore  # pure-geometry lookup ONLY (see module docstring)

N_PORTS = N_CELLS * N_PORTS_PER_CELL   # 37 * 6 = 222
_ACT_CELLROT = N_CELLS * N_ROTATIONS   # 222


# ---------------------------------------------------------------------------
# Static geometry lookup tables (built once at load from pure geometry).
# ---------------------------------------------------------------------------

def _build_static():
    cells = list(_hexcore.hex_board(3))
    idx = {c: i for i, c in enumerate(cells)}
    assert len(cells) == N_CELLS

    # CELL_NEIGH[c, e] = neighbor cell index across edge e, or -1 if off board.
    cell_neigh = np.full((N_CELLS, 6), -1, dtype=np.int64)
    for c, cell in enumerate(cells):
        for e in range(6):
            nb = _hexcore.neighbor(cell, e)
            if nb in idx:
                cell_neigh[c, e] = idx[nb]

    # PORT_PARTNER[p] = boundary partner port id across the shared edge, or -1.
    # port id p = cell*6 + edge; partner = (neighbor(cell,e), (e+3)%6).
    port_partner = np.full(N_PORTS, -1, dtype=np.int64)
    for c in range(N_CELLS):
        for e in range(6):
            nb = cell_neigh[c, e]
            if nb >= 0:
                port_partner[c * 6 + e] = nb * 6 + ((e + 3) % 6)

    # INTERNAL_MATCH[type_idx, rot, edge] = the edge paired with `edge` in the
    # tile's arc matching. type_idx 0 -> tile type 0, 1 -> tile type 2.
    tiles = _hexcore.canonical_tiles()
    type_values = [0, 2]
    internal_match = np.zeros((2, 6, 6), dtype=np.int64)
    for ti, tv in enumerate(type_values):
        matching = tiles[tv]["matching"]
        for rot in range(6):
            for (ea, eb) in _hexcore.tile_arcs(matching, rot):
                internal_match[ti, rot, ea] = eb
                internal_match[ti, rot, eb] = ea

    # RAY_CELL[c, k] = k-th cell index along the edge-0 ray from c (incl. c),
    # replicating stage0.enclosed_cells' walk; -1 padded. R = max ray length.
    ray_lists = []
    R = 0
    for c, cell in enumerate(cells):
        seq = [c]
        cur = cell
        while True:
            nxt = _hexcore.neighbor(cur, 0)
            if nxt not in idx:
                break
            cur = nxt
            seq.append(idx[cur])
        ray_lists.append(seq)
        R = max(R, len(seq))
    ray_cell = np.full((N_CELLS, R), -1, dtype=np.int64)
    for c, seq in enumerate(ray_lists):
        ray_cell[c, :len(seq)] = seq

    # Action decode tables: action -> (hand_slot, cell_idx, rotation).
    a = np.arange(ACTION_SPACE_SIZE, dtype=np.int64)
    a_hand = a // _ACT_CELLROT
    rem = a % _ACT_CELLROT
    a_cell = rem // N_ROTATIONS
    a_rot = rem % N_ROTATIONS

    return {
        "CELL_NEIGH": torch.from_numpy(cell_neigh),
        "PORT_PARTNER": torch.from_numpy(port_partner),
        "INTERNAL_MATCH": torch.from_numpy(internal_match),
        "RAY_CELL": torch.from_numpy(ray_cell),
        "A_HAND": torch.from_numpy(a_hand),
        "A_CELL": torch.from_numpy(a_cell),
        "A_ROT": torch.from_numpy(a_rot),
        "R": R,
    }


_STATIC = _build_static()
# Safety cap on label-propagation iterations. Real arc components are tiny and
# the loop breaks early on convergence, so this bound is only ever hit by a
# pathological board; the port count is a safe over-estimate of any component
# diameter.
_LABEL_CAP = N_PORTS


class HexTruchetBatched(BatchedEnv):
    def __init__(self, n, **kwargs):
        super().__init__(n, **kwargs)
        dev = self.device
        self.CELL_NEIGH = _STATIC["CELL_NEIGH"].to(dev)
        self.PORT_PARTNER = _STATIC["PORT_PARTNER"].to(dev)
        self.INTERNAL_MATCH = _STATIC["INTERNAL_MATCH"].to(dev)
        self.RAY_CELL = _STATIC["RAY_CELL"].to(dev)
        self.A_HAND = _STATIC["A_HAND"].to(dev)
        self.A_CELL = _STATIC["A_CELL"].to(dev)
        self.A_ROT = _STATIC["A_ROT"].to(dev)
        self.R = _STATIC["R"]
        self._edge0_ports = torch.arange(N_CELLS, device=dev) * 6
        self._has_bnd = self.PORT_PARTNER >= 0
        self._bnd_safe = torch.where(self._has_bnd, self.PORT_PARTNER,
                                     torch.zeros_like(self.PORT_PARTNER))

    # -- reset -------------------------------------------------------------

    def _reset_instances(self, mask: torch.Tensor) -> None:
        dev = self.device
        N = self.n
        m_cell = mask.view(N, 1)

        # spec: Reset — empty board, scores 0, current_player 0; deal 6 tiles
        # via INITIAL_DEAL (step 0, index 0..5), each written before the next
        # draw's counts are computed. Draw for the whole batch (stateless RNG).
        fresh_bt = torch.full((N, N_CELLS), -1, dtype=torch.int64, device=dev)
        fresh_br = torch.zeros((N, N_CELLS), dtype=torch.int64, device=dev)
        fresh_h0 = torch.full((N, HAND_SIZE), -1, dtype=torch.int64, device=dev)
        fresh_h1 = torch.full((N, HAND_SIZE), -1, dtype=torch.int64, device=dev)

        deal_order = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        for index, (which, slot) in enumerate(deal_order):
            tile = self._draw_tile(self.keys, 0, Slots.INITIAL_DEAL, index,
                                   fresh_bt, fresh_h0, fresh_h1)
            if which == 0:
                fresh_h0[:, slot] = tile
            else:
                fresh_h1[:, slot] = tile

        fresh_cp = torch.zeros(N, dtype=torch.int64, device=dev)
        fresh_s0 = torch.zeros(N, dtype=torch.int64, device=dev)
        fresh_s1 = torch.zeros(N, dtype=torch.int64, device=dev)

        def upd(name, fresh, shaped_mask):
            cur = getattr(self, name, None)
            if cur is None:
                cur = torch.zeros_like(fresh)
            setattr(self, name, torch.where(shaped_mask, fresh, cur))

        upd("board_tile", fresh_bt, m_cell)
        upd("board_rotation", fresh_br, m_cell)
        upd("hand_p0", fresh_h0, m_cell)
        upd("hand_p1", fresh_h1, m_cell)
        upd("current_player", fresh_cp, mask)
        upd("score_p0", fresh_s0, mask)
        upd("score_p1", fresh_s1, mask)
        # self.t is zeroed for masked instances by the base class.

    # -- deck draw (spec: RNG slots) --------------------------------------

    def _draw_tile(self, keys, step, slot, index, bt, h0, h1):
        """One sequential Bernoulli-without-replacement draw for the whole
        batch. Returns tile type tensor {0,2} [N]. p = remaining_type0 /
        remaining_total in float64, matching the reference's
        draw_uniform(...) < p bit-for-bit."""
        type0_drawn = ((bt == 0).sum(1) + (h0 == 0).sum(1) + (h1 == 0).sum(1))
        total_drawn = ((bt != -1).sum(1) + (h0 != -1).sum(1) + (h1 != -1).sum(1))
        remaining_type0 = DECK_TYPE0_COUNT - type0_drawn
        remaining_total = DECK_SIZE - total_drawn
        # Sanitize the discarded branch (footgun #1): remaining_total can be 0
        # for instances that would not actually draw (deck exhausted); clamp
        # before dividing, the result is masked out by the caller anyway.
        remaining_total_safe = torch.clamp(remaining_total, min=1)
        p = remaining_type0.to(torch.float64) / remaining_total_safe.to(torch.float64)
        u = rng.draw_uniform_torch(keys, step, slot, index)
        draw_is_type0 = u < p
        return torch.where(draw_is_type0,
                           torch.zeros_like(type0_drawn),
                           torch.full_like(type0_drawn, 2))

    # -- legality (spec: Actions) -----------------------------------------

    def _legal_mask(self, bt, cp, t):
        """[N, 666] bool legal-action mask from public board + acting player's
        own hand. spec: Actions legality."""
        N = bt.shape[0]
        occupied = bt != -1                                  # [N,37]
        is_empty = ~occupied
        neigh = self.CELL_NEIGH                              # [37,6]
        neigh_valid = neigh >= 0
        neigh_safe = torch.clamp(neigh, min=0)
        occ_at_neigh = occupied[:, neigh_safe.reshape(-1)].reshape(N, N_CELLS, 6)
        neigh_occ = (occ_at_neigh & neigh_valid.view(1, N_CELLS, 6)).any(-1)  # [N,37]
        t_is_zero = (t == 0).view(N, 1)
        cell_legal = is_empty & (t_is_zero | neigh_occ)      # [N,37]
        my_hand = torch.where((cp == 0).view(N, 1), self.hand_p0, self.hand_p1)
        hand_occ = my_hand != -1                             # [N,3]
        return hand_occ[:, self.A_HAND] & cell_legal[:, self.A_CELL]  # [N,666]

    # -- loop closure + enclosed area (spec: Rewards / transition step 2) ---

    def _total_loop_area(self, bt, br):
        """Sum over all closed loops of their enclosed-cell counts, batched.
        Reimplements the union-find loop detection + ray-cast area rule as
        tensor ops. Returns [N] int64."""
        N = bt.shape[0]
        dev = bt.device

        placed = bt != -1                                    # [N,37]
        type_idx = torch.where(bt == 0, torch.zeros_like(bt),
                               torch.where(bt == 2, torch.ones_like(bt),
                                           torch.zeros_like(bt)))  # [N,37]

        # internal arc partner edge for every (cell, edge): IM[type_idx, rot, e]
        base = (type_idx.clamp(0, 1) * 6 + br.clamp(0, 5))   # [N,37]
        im_flat = self.INTERNAL_MATCH.reshape(12, 6)         # [12,6]
        partner_edge = im_flat[base]                         # [N,37,6]
        cell_ids = torch.arange(N_CELLS, device=dev).view(1, N_CELLS, 1)
        internal_partner = (cell_ids * 6 + partner_edge).reshape(N, N_PORTS)  # [N,222]

        active = placed.unsqueeze(-1).expand(N, N_CELLS, 6).reshape(N, N_PORTS)

        bnd_safe = self._bnd_safe.view(1, N_PORTS).expand(N, N_PORTS)
        active_at_bnd = torch.gather(active, 1, bnd_safe)
        bnd_active = active & self._has_bnd.view(1, N_PORTS) & active_at_bnd  # [N,222]
        open_port = active & ~bnd_active                      # [N,222]

        # connected components via min-label propagation; max-propagate each
        # component's "has an open port" flag. Loop = component with no open
        # port (matches graph.py: every arc's ports boundary-connected).
        label = torch.arange(N_PORTS, device=dev).view(1, N_PORTS).expand(N, N_PORTS).clone()
        co = open_port.to(torch.int64)
        new_label, new_co = label, co
        for _ in range(_LABEL_CAP):
            lab_ip = torch.gather(label, 1, internal_partner)
            co_ip = torch.gather(co, 1, internal_partner)
            new_label = torch.where(active, torch.minimum(label, lab_ip), label)
            new_co = torch.where(active, torch.maximum(co, co_ip), co)
            lab_bp = torch.gather(new_label, 1, bnd_safe)
            co_bp = torch.gather(new_co, 1, bnd_safe)
            new_label = torch.where(bnd_active, torch.minimum(new_label, lab_bp), new_label)
            new_co = torch.where(bnd_active, torch.maximum(new_co, co_bp), new_co)
            if torch.equal(new_label, label) and torch.equal(new_co, co):
                break
            label, co = new_label, new_co
        label, co = new_label, new_co

        loop_port = active & (co == 0)                        # [N,222]

        # edge-0 loop label per cell (port (d,0)=d*6): label if loop-port else -1
        e0_is_loop = loop_port[:, self._edge0_ports]          # [N,37]
        e0_label = torch.where(e0_is_loop, label[:, self._edge0_ports],
                               torch.full((1,), -1, dtype=torch.int64, device=dev))

        # ray-cast crossing parity: per cell, count loops crossed an ODD number
        # of times along its edge-0 ray = number of loops containing the cell.
        R = self.R
        rc = self.RAY_CELL                                    # [37,R], -1 pad
        valid_slot = (rc >= 0)
        rc_safe = torch.clamp(rc, min=0)
        idx_flat = rc_safe.reshape(-1).view(1, -1).expand(N, N_CELLS * R)
        ray_lab = e0_label.gather(1, idx_flat).reshape(N, N_CELLS, R)  # [N,37,R]
        ray_lab = torch.where(valid_slot.view(1, N_CELLS, R), ray_lab,
                              torch.full_like(ray_lab, -1))
        a = ray_lab.unsqueeze(-1)                             # [N,37,R,1]
        b = ray_lab.unsqueeze(-2)                             # [N,37,1,R]
        eq = (a == b) & (a != -1)                             # [N,37,R,R]
        kk = torch.arange(R, device=dev)
        le = (kk.view(R, 1) >= kk.view(1, R)).view(1, 1, R, R)  # j<=k
        gt = (kk.view(R, 1) < kk.view(1, R)).view(1, 1, R, R)   # j>k
        running = (eq & le).sum(-1)                           # [N,37,R]
        has_later = (eq & gt).any(-1)                         # [N,37,R]
        is_last = ~has_later
        valid_lab = ray_lab != -1
        contributes = valid_lab & is_last & (running % 2 == 1)
        odd_count = contributes.sum(-1)                       # [N,37]
        return odd_count.sum(-1)                              # [N]

    # -- step (spec: transition) ------------------------------------------

    def _step_impl(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N = self.n
        dev = self.device
        p = self.current_player                               # [N]
        k = self.t                                            # [N] pre-move step

        bt, br = self.board_tile, self.board_rotation
        h0, h1 = self.hand_p0, self.hand_p1

        # spec: Actions — clamp illegal action to smallest legal index.
        legal = self._legal_mask(bt, p, k)                    # [N,666]
        chosen_legal = legal.gather(1, actions.view(N, 1)).squeeze(1)
        arange_a = torch.arange(ACTION_SPACE_SIZE, device=dev).view(1, -1)
        first_legal = torch.where(legal, arange_a,
                                  torch.full_like(arange_a, ACTION_SPACE_SIZE)).min(1).values
        eff = torch.where(chosen_legal, actions, first_legal)  # [N]
        hs = eff // _ACT_CELLROT
        rem = eff % _ACT_CELLROT
        ci = rem // N_ROTATIONS
        rot = rem % N_ROTATIONS

        my_hand = torch.where((p == 0).view(N, 1), h0, h1)     # [N,3]
        tile_type = my_hand.gather(1, hs.view(N, 1)).squeeze(1)  # [N]

        # spec: transition step 2 — place, score = area(after) - area(before).
        area_before = self._total_loop_area(bt, br)
        bt_new = bt.scatter(1, ci.view(N, 1), tile_type.view(N, 1))
        br_new = br.scatter(1, ci.view(N, 1), rot.view(N, 1))
        area_after = self._total_loop_area(bt_new, br_new)
        gained = area_after - area_before                      # [N]
        s0 = torch.where(p == 0, self.score_p0 + gained, self.score_p0)
        s1 = torch.where(p == 1, self.score_p1 + gained, self.score_p1)

        # spec: transition step 3 — remove played tile, keep hand left-packed.
        old_hand = my_hand
        shifted = torch.cat([old_hand[:, 1:],
                             torch.full((N, 1), -1, dtype=torch.int64, device=dev)], dim=1)
        slot_idx = torch.arange(HAND_SIZE, device=dev).view(1, HAND_SIZE)
        removed_hand = torch.where(slot_idx < hs.view(N, 1), old_hand, shifted)
        sz = (old_hand != -1).sum(1)                           # [N]
        refill_slot = sz - 1                                   # [N]

        # spec: transition step 4 — refill from deck for k < 31 (write into the
        # opened slot); counts computed on the post-place, post-removal state.
        h0_mid = torch.where((p == 0).view(N, 1), removed_hand, h0)
        h1_mid = torch.where((p == 1).view(N, 1), removed_hand, h1)
        drawn = self._draw_tile(self.keys, k, Slots.DECK_DRAW, 0, bt_new, h0_mid, h1_mid)
        do_draw = (k < N_REFILL_STEPS)
        refill_value = torch.where(do_draw, drawn, torch.full_like(drawn, -1))
        refilled_hand = removed_hand.scatter(1, refill_slot.view(N, 1),
                                             refill_value.view(N, 1))
        h0_new = torch.where((p == 0).view(N, 1), refilled_hand, h0)
        h1_new = torch.where((p == 1).view(N, 1), refilled_hand, h1)

        # spec: transition steps 5-8 — flip player, reward, terminate.
        cp_new = 1 - p
        t_next = k + 1
        terminated = (t_next == N_CELLS)
        margin = torch.where(p == 0, s0 - s1, s1 - s0)          # [N]
        rewards = torch.where(terminated, margin.to(torch.float64),
                              torch.zeros(N, dtype=torch.float64, device=dev))

        self.board_tile, self.board_rotation = bt_new, br_new
        self.hand_p0, self.hand_p1 = h0_new, h1_new
        self.score_p0, self.score_p1 = s0, s1
        self.current_player = cp_new
        return rewards, terminated

    # -- observation (spec: Observations) ---------------------------------

    def observe(self) -> torch.Tensor:
        N = self.n
        cp = self.current_player
        cp0 = (cp == 0).view(N, 1)
        my_hand = torch.where(cp0, self.hand_p0, self.hand_p1)
        opp_hand = torch.where(cp0, self.hand_p1, self.hand_p0)
        tn = float(TILE_NORM)

        board_tile_norm = self.board_tile.to(torch.float32) / tn                  # [N,37]
        board_rot_norm = self.board_rotation.to(torch.float32) / float(ROT_NORM)  # [N,37]
        my_hand_norm = my_hand.to(torch.float32) / tn                             # [N,3]
        # opponent private masking: empty -> -1/4, occupied (hidden) -> -2/4.
        opp_raw = torch.where(opp_hand == -1,
                              torch.full_like(opp_hand, -1),
                              torch.full_like(opp_hand, -2))
        opp_hand_norm = opp_raw.to(torch.float32) / tn                            # [N,3]
        my_score = torch.where(cp == 0, self.score_p0, self.score_p1)
        opp_score = torch.where(cp == 0, self.score_p1, self.score_p0)
        my_score_norm = (my_score.to(torch.float32) / float(SCORE_NORM)).view(N, 1)
        opp_score_norm = (opp_score.to(torch.float32) / float(SCORE_NORM)).view(N, 1)
        t_norm = (self.t.to(torch.float32) / float(N_CELLS)).view(N, 1)
        legal = self._legal_mask(self.board_tile, cp, self.t).to(torch.float32)   # [N,666]

        return torch.cat([board_tile_norm, board_rot_norm, my_hand_norm,
                          opp_hand_norm, my_score_norm, opp_score_norm,
                          t_norm, legal], dim=1)

    # -- serialization -----------------------------------------------------

    def state_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "t": self.t,
            "current_player": self.current_player,
            "board_tile": self.board_tile,
            "board_rotation": self.board_rotation,
            "hand_p0": self.hand_p0,
            "hand_p1": self.hand_p1,
            "score_p0": self.score_p0,
            "score_p1": self.score_p1,
        }

    # -- invariants (spec: Invariants 1-12) -------------------------------

    @invariant("t_range")
    def _inv_t_range(self):
        return (self.t >= 0) & (self.t <= N_CELLS)

    @invariant("current_player_range")
    def _inv_cp_range(self):
        return (self.current_player >= 0) & (self.current_player <= 1)

    @invariant("turn_alternation")
    def _inv_turn(self):
        return self.current_player == (self.t % 2)

    @invariant("occupied_equals_t")
    def _inv_occ_t(self):
        return (self.board_tile != -1).sum(1) == self.t

    @invariant("board_tile_types")
    def _inv_tile_types(self):
        bt = self.board_tile
        return ((bt == -1) | (bt == 0) | (bt == 2)).all(1)

    @invariant("rotation_zero_when_empty")
    def _inv_rot_empty(self):
        return ((self.board_tile != -1) | (self.board_rotation == 0)).all(1)

    @invariant("tile_conservation_total")
    def _inv_cons_total(self):
        occ = (self.board_tile != -1).sum(1)
        h0 = (self.hand_p0 != -1).sum(1)
        h1 = (self.hand_p1 != -1).sum(1)
        deck_rem = N_CELLS - 6 - torch.clamp(self.t, max=N_REFILL_STEPS)
        return (occ + h0 + h1 + deck_rem) == N_CELLS

    @invariant("tile_conservation_by_type")
    def _inv_cons_type(self):
        bt, h0, h1 = self.board_tile, self.hand_p0, self.hand_p1
        t0 = (bt == 0).sum(1) + (h0 == 0).sum(1) + (h1 == 0).sum(1)
        t2 = (bt == 2).sum(1) + (h0 == 2).sum(1) + (h1 == 2).sum(1)
        present = (bt != -1).sum(1) + (h0 != -1).sum(1) + (h1 != -1).sum(1)
        return (t0 <= DECK_TYPE0_COUNT) & (t2 <= DECK_SIZE - DECK_TYPE0_COUNT) \
            & (t0 + t2 == present)

    @invariant("hands_left_packed")
    def _inv_left_packed(self):
        def lp(h):
            return ((h[:, 0] != -1) | (h[:, 1] == -1)) \
                & ((h[:, 1] != -1) | (h[:, 2] == -1))
        return lp(self.hand_p0) & lp(self.hand_p1)

    @invariant("scores_nonneg")
    def _inv_scores(self):
        return (self.score_p0 >= 0) & (self.score_p1 >= 0)

    @invariant("reset_shape")
    def _inv_reset(self):
        at0 = self.t == 0
        cond = (self.board_tile == -1).all(1) \
            & ((self.hand_p0 != -1).sum(1) == HAND_SIZE) \
            & ((self.hand_p1 != -1).sum(1) == HAND_SIZE) \
            & (self.score_p0 == 0) & (self.score_p1 == 0) \
            & (self.current_player == 0)
        return (~at0) | cond

    @invariant("terminal_shape")
    def _inv_terminal(self):
        atT = self.t == N_CELLS
        cond = (self.board_tile != -1).all(1) \
            & (self.hand_p0 == -1).all(1) & (self.hand_p1 == -1).all(1)
        return (~atT) | cond
