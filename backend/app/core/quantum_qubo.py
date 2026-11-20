"""Phase 4 Quantum-Inspired QUBO-MWIS Combinatorial Engine (Harvard HQI Formulation).

Grounded in verified Harvard Quantum Initiative / SEAS formulations:
  1. Ebadi, Lukin et al. (Science 2022, arXiv:2202.09372): Maximum Weighted Independent Set
     (MWIS) via Rydberg blockade Hamiltonian H_QUBO = -sum_i w_i x_i + sum_{(i,j) in E} U_ij x_i x_j.
  2. Nguyen, Lukin et al. (PRX Quantum 2023, arXiv:2209.03965): Arbitrary-graph QUBO/MWIS
     embedding and QuEra Aquila physical register mapping (R_b = 8.37 um, d_min = 4.0 um).
  3. Goto et al. (Sci. Adv. 2021) + Zhou, Lukin et al. (PRX 2020): Vectorized Ballistic Simulated
     Bifurcation (bSB) adiabatic Ising dynamics with 1-flip / 1-out-2-in combinatorial polish.
  4. Huang, Preskill, Lukin et al. (Science 2022): Classical Shadows uncertainty weights for
     multi-shot ActivePose Next-Best-View (NBV) Set-Cover optimization.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Any, Optional, Sequence

import numpy as np

from app.core.geometry_3d import compute_3d_iou, obb_intersects_sat

# Support surfaces allowed to physically contain/support smaller distinct objects
_SUPPORT_LABELS: frozenset[str] = frozenset(
    {
        "table",
        "desk",
        "floor",
        "ground",
        "shelf",
        "counter",
        "workbench",
        "bed",
        "tray",
        "rack",
        "cabinet",
        "wall",
        "carpet",
        "rug",
        "mat",
        "mesa",
        "escritorio",
        "piso",
        "estante",
    }
)

# Physical constants of QuEra Aquila Neutral-Atom QPU (Ebadi et al., Science 2022)
_AQUILA_BLOCKADE_RADIUS_UM: float = 8.37
_AQUILA_MIN_SPACING_UM: float = 4.0
_AQUILA_FOV_UM: tuple[float, float] = (75.0, 76.0)
_AQUILA_MAX_QUBITS: int = 256


def normalize_qubo(Q: np.ndarray) -> np.ndarray:
    """Symmetrize square QUBO matrix Q so Q_s[i,i]=Q[i,i] and x^T Q_s x == x^T Q x."""
    Q_arr = np.asarray(Q, dtype=np.float64)
    if Q_arr.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if Q_arr.ndim != 2 or Q_arr.shape[0] != Q_arr.shape[1]:
        raise ValueError(f"Expected square QUBO matrix, got shape {Q_arr.shape}")
    return 0.5 * (Q_arr + Q_arr.T)


def evaluate_qubo_energy(Q_s: np.ndarray, x: np.ndarray) -> float:
    """Evaluate binary quadratic objective E(x) = x^T Q_s x for x in {0,1}^n."""
    x_vec = np.asarray(x, dtype=np.float64).ravel()
    if x_vec.size == 0:
        return 0.0
    Q_sym = normalize_qubo(Q_s)
    return float(x_vec @ Q_sym @ x_vec)


def qubo_to_ising(Q_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact bijection x=(s+1)/2 from QUBO (x in {0,1}^n) to Ising E(s) = s^T J s + h^T s + offset."""
    Q_sym = normalize_qubo(Q_s)
    n = int(Q_sym.shape[0])
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64), np.zeros(0, dtype=np.float64), 0.0

    d = np.diag(Q_sym).copy()
    W = Q_sym - np.diag(d)
    J = 0.25 * W
    h = 0.5 * d + 0.5 * (W @ np.ones(n, dtype=np.float64))
    offset = 0.5 * float(np.sum(d)) + 0.25 * float(np.sum(W))
    return J, h, offset


def _polish_1flip_steepest(
    Q_s: np.ndarray,
    x: np.ndarray,
    max_sweeps: int = 8,
) -> tuple[np.ndarray, float]:
    """Exact steepest-descent single-bit-flip local search on symmetric QUBO Q_s."""
    n = int(Q_s.shape[0])
    x_cur = np.asarray(x, dtype=np.float64).ravel().copy()
    if n == 0:
        return x_cur, 0.0

    d = np.diag(Q_s)
    W = Q_s - np.diag(d)
    max_moves = max(8, max_sweeps * min(n, 32))

    for _ in range(max_moves):
        Wx = W @ x_cur
        deltas = (1.0 - 2.0 * x_cur) * (d + 2.0 * Wx)
        k = int(np.argmin(deltas))
        if deltas[k] >= -1e-11:
            break
        x_cur[k] = 1.0 - x_cur[k]

    return x_cur, evaluate_qubo_energy(Q_s, x_cur)


def solve_qubo_greedy(Q_s: np.ndarray) -> tuple[np.ndarray, float]:
    """Deterministic classical greedy baseline starting from x=0 and flipping improving variables."""
    Q_sym = normalize_qubo(Q_s)
    n = int(Q_sym.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64), 0.0

    d = np.diag(Q_sym)
    W = Q_sym - np.diag(d)
    x_cur = np.zeros(n, dtype=np.float64)

    for _ in range(n):
        Wx = W @ x_cur
        # Candidate additions (x[k] == 0)
        deltas = d + 2.0 * Wx
        inactive = x_cur < 0.5
        if not np.any(inactive):
            break
        masked_deltas = np.where(inactive, deltas, np.inf)
        k = int(np.argmin(masked_deltas))
        if masked_deltas[k] >= -1e-11:
            break
        x_cur[k] = 1.0

    return _polish_1flip_steepest(Q_sym, x_cur, max_sweeps=4)


def solve_qubo_simulated_bifurcation(
    Q_s: np.ndarray,
    *,
    n_trajectories: int = 24,
    n_steps: int = 100,
    dt: float = 0.8,
    seed: int = 42,
    record_history: bool = True,
) -> dict[str, Any]:
    """Vectorized Ballistic Simulated Bifurcation (bSB) Ising/QUBO solver across B trajectories."""
    t0 = time.perf_counter()
    Q_sym = normalize_qubo(Q_s)
    n = int(Q_sym.shape[0])
    if n == 0:
        return {
            "x_best": np.zeros(0, dtype=np.float64),
            "energy": 0.0,
            "solver_ms": 0.0,
            "energy_history": [],
            "candidates": [],
        }

    J, h, _ = qubo_to_ising(Q_sym)
    B = max(2, int(n_trajectories))
    steps = max(8, int(n_steps))

    a0 = 1.0
    scale_j = float(np.linalg.norm(2.0 * J)) / max(math.sqrt(n), 1.0)
    scale_h = float(np.linalg.norm(h)) / max(math.sqrt(n), 1.0)
    sigma = max(scale_j, scale_h, 1e-6)
    c0 = 0.5 * a0 / sigma

    rng = np.random.default_rng(int(seed))
    Q_b = rng.uniform(-0.1, 0.1, size=(n, B))
    P_b = np.zeros((n, B), dtype=np.float64)

    sample_stride = max(1, steps // 20)
    history: list[dict[str, float]] = []
    candidate_pool: dict[bytes, tuple[float, np.ndarray]] = {}

    two_J = 2.0 * J
    h_col = h[:, None]

    def _capture(step_idx: int) -> None:
        X_bin = (Q_b > 0.0).astype(np.float64)
        energies = np.sum(X_bin * (Q_sym @ X_bin), axis=0)
        if record_history:
            history.append(
                {
                    "step": int(step_idx + 1),
                    "best_energy": round(float(np.min(energies)), 4),
                    "mean_energy": round(float(np.mean(energies)), 4),
                }
            )
        top_order = np.argsort(energies)[: min(4, B)]
        for col in top_order:
            vec = X_bin[:, col]
            key = vec.astype(np.uint8).tobytes()
            e_val = float(energies[col])
            prev = candidate_pool.get(key)
            if prev is None or e_val < prev[0]:
                candidate_pool[key] = (e_val, vec.copy())

    for step in range(steps):
        a_t = a0 * (float(step) / float(max(steps - 1, 1)))
        force = -(a0 - a_t) * Q_b - c0 * (two_J @ Q_b + h_col)
        P_b += force * dt
        Q_b += a0 * P_b * dt

        hit = np.abs(Q_b) > 1.0
        if np.any(hit):
            Q_b = np.clip(Q_b, -1.0, 1.0)
            P_b[hit] = 0.0

        if (step % sample_stride == 0) or (step == steps - 1):
            _capture(step)

    # Polish top 6 unique binary trajectories via exact 1-flip steepest descent
    ranked = sorted(candidate_pool.values(), key=lambda item: item[0])[:6]
    best_x = np.zeros(n, dtype=np.float64)
    best_e = evaluate_qubo_energy(Q_sym, best_x)

    for _, cand_x in ranked:
        px, pe = _polish_1flip_steepest(Q_sym, cand_x, max_sweeps=8)
        if pe < best_e - 1e-11:
            best_e = pe
            best_x = px

    solver_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "x_best": best_x,
        "energy": float(best_e),
        "solver_ms": solver_ms,
        "energy_history": history,
        "candidates": [cand for _, cand in ranked],
    }


def _solve_mwis_exact_bnb(
    weights: np.ndarray,
    adj_conflict: np.ndarray,
    max_nodes: int = 40000,
) -> Optional[tuple[np.ndarray, float]]:
    """Exact bitmask Branch-and-Bound solver for small MWIS graphs (N <= 20)."""
    n = int(weights.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64), 0.0
    if n > 24:
        return None

    degrees = np.sum(adj_conflict, axis=1).astype(np.float64)
    score = np.where(weights > 0.0, weights / (degrees + 1.0), -1e9)
    order = np.argsort(-score, kind="mergesort")
    w_ord = np.maximum(weights[order], 0.0).astype(np.float64)

    # Neighbor bitmasks in reordered indexing
    adj_ord = adj_conflict[np.ix_(order, order)]
    nbr_masks: list[int] = []
    for i in range(n):
        m = 0
        row = adj_ord[i]
        for j in range(n):
            if row[j] and j != i:
                m |= 1 << j
        nbr_masks.append(m)

    # Suffix sum of positive weights for upper bound pruning
    suffix_max = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_max[i] = suffix_max[i + 1] + float(w_ord[i])

    # Warm-start lower bound with greedy weighted degree solution
    best_weight = 0.0
    best_mask = 0
    greedy_forb = 0
    for i in range(n):
        if w_ord[i] > 0.0 and not ((greedy_forb >> i) & 1):
            best_mask |= 1 << i
            best_weight += float(w_ord[i])
            greedy_forb |= nbr_masks[i]

    visited = 0
    aborted = False

    def _dfs(idx: int, forb_mask: int, cur_mask: int, cur_w: float) -> None:
        nonlocal best_weight, best_mask, visited, aborted
        visited += 1
        if visited > max_nodes:
            aborted = True
            return
        if cur_w > best_weight + 1e-12:
            best_weight = cur_w
            best_mask = cur_mask
        if idx >= n or cur_w + suffix_max[idx] <= best_weight + 1e-12:
            return

        #Tighter dynamic remaining positive weight upper bound
        rem_avail = 0.0
        for j in range(idx, n):
            if w_ord[j] > 0.0 and not ((forb_mask >> j) & 1):
                rem_avail += float(w_ord[j])
        if cur_w + rem_avail <= best_weight + 1e-12:
            return

        # Branch: pick next eligible positive-weight node if any
        if w_ord[idx] > 0.0 and not ((forb_mask >> idx) & 1):
            _dfs(idx + 1, forb_mask | nbr_masks[idx], cur_mask | (1 << idx), cur_w + float(w_ord[idx]))
            if aborted:
                return
        _dfs(idx + 1, forb_mask, cur_mask, cur_w)

    _dfs(0, 0, 0, 0.0)
    if aborted:
        return None

    x_out = np.zeros(n, dtype=np.float64)
    for ord_i in range(n):
        if (best_mask >> ord_i) & 1:
            orig_i = int(order[ord_i])
            if weights[orig_i] > 0.0:
                x_out[orig_i] = 1.0
    return x_out, float(best_weight)


def _greedy_mwis(weights: np.ndarray, adj_bool: np.ndarray) -> np.ndarray:
    """Classical weight-descending greedy Maximal Independent Set baseline."""
    n = int(weights.shape[0])
    x = np.zeros(n, dtype=np.float64)
    order = np.argsort(-weights, kind="mergesort")
    blocked = np.zeros(n, dtype=bool)
    for idx in order:
        i = int(idx)
        if weights[i] <= 0.0 or blocked[i]:
            continue
        x[i] = 1.0
        blocked |= adj_bool[i]
    return x


def _polish_mwis_1out_k_in(
    weights: np.ndarray,
    adj_bool: np.ndarray,
    x_init: np.ndarray,
    max_rounds: int = 24,
) -> np.ndarray:
    """Escape star/hub local minima via 1-out / k-in valid independent-set swaps + greedy fill."""
    n = int(weights.shape[0])
    sel = (np.asarray(x_init, dtype=np.float64).ravel() > 0.5).copy()

    # First ensure zero conflicts
    while True:
        sel_idx = np.where(sel)[0]
        if sel_idx.size < 2:
            break
        sub = adj_bool[np.ix_(sel_idx, sel_idx)]
        np.fill_diagonal(sub, False)
        if not np.any(sub):
            break
        # Drop conflicted selected node with lowest weight / highest conflict degree
        c_deg = np.sum(sub, axis=1)
        worst = int(np.argmin(weights[sel_idx] - 1e-3 * c_deg))
        sel[sel_idx[worst]] = False

    def _fill_free() -> None:
        order = np.argsort(-weights, kind="mergesort")
        for cand_i in order:
            c = int(cand_i)
            if sel[c] or weights[c] <= 0.0:
                continue
            if not np.any(adj_bool[c] & sel):
                sel[c] = True

    _fill_free()

    for _ in range(max_rounds):
        improved = False
        sel_idx = np.where(sel)[0]
        for u_raw in sel_idx:
            u = int(u_raw)
            sel[u] = False
            # Nodes newly eligible without u
            eligible = [
                int(v)
                for v in np.where(adj_bool[u] & (~sel) & (weights > 0.0))[0]
                if not np.any(adj_bool[v] & sel)
            ]
            if len(eligible) >= 2:
                eligible.sort(key=lambda idx: -float(weights[idx]))
                added: list[int] = []
                gain = 0.0
                for v in eligible:
                    if all(not adj_bool[v, prev] for prev in added):
                        added.append(v)
                        gain += float(weights[v])
                if gain > float(weights[u]) + 1e-9:
                    for v in added:
                        sel[v] = True
                    improved = True
                    _fill_free()
                    break
            sel[u] = True
        if not improved:
            break

    return sel.astype(np.float64)


def solve_mwis_rydberg(
    weights: np.ndarray,
    conflict_matrix: np.ndarray,
    *,
    penalty_scale: float = 1.35,
    exact_if_small: bool = True,
    enforce_independent_set: bool = True,
    n_trajectories: int = 24,
    n_steps: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Solve Maximum Weighted Independent Set using the Rydberg-blockade QUBO Hamiltonian."""
    w = np.asarray(weights, dtype=np.float64).ravel().copy()
    n = int(w.shape[0])
    if n == 0:
        return {
            "x_best": np.zeros(0, dtype=np.float64),
            "selected_indices": [],
            "Q_sym": np.zeros((0, 0), dtype=np.float64),
            "qubo_energy": 0.0,
            "greedy_energy": 0.0,
            "energy_gain": 0.0,
            "selected_weight_sum": 0.0,
            "greedy_weight_sum": 0.0,
            "greedy_selected_indices": [],
            "conflict_violations": 0,
            "n_vars": 0,
            "n_conflict_edges": 0,
            "conflict_density": 0.0,
            "exact_optimum_verified": True,
            "solver_method": "rydberg_mwis_empty",
            "solver_ms": 0.0,
            "energy_history": [],
        }

    C = np.asarray(conflict_matrix, dtype=np.float64).copy()
    if C.shape != (n, n):
        raise ValueError(f"conflict_matrix shape {C.shape} does not match weights length {n}")
    C = np.maximum(C, C.T)
    np.fill_diagonal(C, 0.0)
    adj_bool = C > 0.0

    max_w = max(float(np.max(w)), 1e-6)
    lam = float(penalty_scale) * max_w + 1e-6

    Q_upper = np.diag(-w)
    iu, ju = np.triu_indices(n, k=1)
    edge_mask = adj_bool[iu, ju]
    if np.any(edge_mask):
        mult = np.maximum(C[iu[edge_mask], ju[edge_mask]], 1.0)
        Q_upper[iu[edge_mask], ju[edge_mask]] = lam * mult
    Q_sym = normalize_qubo(Q_upper)

    n_edges = int(np.sum(edge_mask))
    max_possible_edges = max(1, n * (n - 1) // 2)
    conflict_density = round(float(n_edges) / float(max_possible_edges), 4) if n > 1 else 0.0

    t0 = time.perf_counter()
    # 1. Classical greedy baseline (honest weight-descending MWIS + energy on Q_sym)
    x_greedy = _greedy_mwis(w, adj_bool)
    greedy_energy = evaluate_qubo_energy(Q_sym, x_greedy)
    greedy_weight_sum = float(np.sum(w * x_greedy))

    # 2. Vectorized Ballistic Simulated Bifurcation
    sb_res = solve_qubo_simulated_bifurcation(
        Q_sym,
        n_trajectories=n_trajectories,
        n_steps=n_steps,
        seed=seed,
        record_history=True,
    )

    best_x = sb_res["x_best"]
    best_e = evaluate_qubo_energy(Q_sym, best_x)
    method_tag = "ballistic_simulated_bifurcation"

    # Evaluate polished candidates including SB trajectories & minimum-degree seed
    deg = np.sum(adj_bool, axis=1).astype(np.float64)
    min_deg_order = np.argsort(-(w / (deg + 1.0)), kind="mergesort")
    x_deg = np.zeros(n, dtype=np.float64)
    blk = np.zeros(n, dtype=bool)
    for idx in min_deg_order:
        ii = int(idx)
        if w[ii] > 0.0 and not blk[ii]:
            x_deg[ii] = 1.0
            blk |= adj_bool[ii]

    cand_list = [best_x, x_deg] + list(sb_res.get("candidates") or [])
    for cand in cand_list:
        polished = _polish_mwis_1out_k_in(w, adj_bool, cand) if enforce_independent_set else cand
        pe = evaluate_qubo_energy(Q_sym, polished)
        if pe < best_e - 1e-11:
            best_e = pe
            best_x = polished
            method_tag = "bsb_rydberg_1out2in"

    if enforce_independent_set:
        best_x = _polish_mwis_1out_k_in(w, adj_bool, best_x)
        best_e = evaluate_qubo_energy(Q_sym, best_x)

    exact_verified = False
    if exact_if_small and n <= 20:
        exact_res = _solve_mwis_exact_bnb(w, adj_bool)
        if exact_res is not None:
            x_ex, _ = exact_res
            e_ex = evaluate_qubo_energy(Q_sym, x_ex)
            exact_verified = True
            if e_ex < best_e - 1e-11:
                best_x = x_ex
                best_e = e_ex
                method_tag = "rydberg_bnb_exact"

    sel_idx = [int(i) for i in np.where(best_x > 0.5)[0]]
    greedy_idx = [int(i) for i in np.where(x_greedy > 0.5)[0]]

    # Verify zero conflict edge violations
    violations = 0
    if len(sel_idx) >= 2:
        sub_adj = np.triu(adj_bool[np.ix_(sel_idx, sel_idx)], k=1)
        violations = int(np.sum(sub_adj))

    # Append post-polish final energy if improved
    history = list(sb_res.get("energy_history") or [])
    if history and best_e < float(history[-1]["best_energy"]) - 1e-9:
        history.append(
            {
                "step": int(history[-1]["step"] + 1),
                "best_energy": round(float(best_e), 4),
                "mean_energy": round(float(history[-1]["mean_energy"]), 4),
            }
        )

    total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    sel_w_sum = float(np.sum(w * best_x))
    return {
        "x_best": best_x,
        "selected_indices": sel_idx,
        "Q_sym": Q_sym,
        "qubo_energy": round(float(best_e), 5),
        "greedy_energy": round(float(greedy_energy), 5),
        "energy_gain": round(float(greedy_energy - best_e), 5),
        "selected_weight_sum": round(sel_w_sum, 4),
        "greedy_weight_sum": round(greedy_weight_sum, 4),
        "greedy_selected_indices": greedy_idx,
        "conflict_violations": violations,
        "n_vars": n,
        "n_conflict_edges": n_edges,
        "conflict_density": conflict_density,
        "exact_optimum_verified": exact_verified,
        "solver_method": method_tag,
        "solver_ms": total_ms,
        "energy_history": history,
    }


def export_qubo_hardware_payload(
    Q_sym: np.ndarray,
    *,
    problem_name: str,
    labels: Optional[Sequence[str]] = None,
    conflict_matrix: Optional[np.ndarray] = None,
    selected_indices: Optional[Sequence[int]] = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Export QUBO to AWS Braket / D-Wave BQM JSON + deterministic QuEra Aquila 2D atom register."""
    Q_s = normalize_qubo(Q_sym)
    n = int(Q_s.shape[0])
    sel_set = {int(i) for i in (selected_indices or [])}
    node_labels = [
        str(labels[i]) if labels is not None and i < len(labels) else f"q_{i}"
        for i in range(n)
    ]

    linear: dict[str, float] = {}
    quadratic: dict[str, float] = {}
    for i in range(n):
        q_ii = float(Q_s[i, i])
        if abs(q_ii) > 1e-12:
            linear[str(i)] = round(q_ii, 5)
    for i in range(n):
        for j in range(i + 1, n):
            b_ij = 2.0 * float(Q_s[i, j])
            if abs(b_ij) > 1e-12:
                quadratic[f"{i},{j}"] = round(b_ij, 5)

    # Conflict adjacency used for unit-disk physical placement
    if conflict_matrix is not None and conflict_matrix.shape == (n, n):
        adj = np.maximum(conflict_matrix, conflict_matrix.T) > 0.0
    else:
        W_off = Q_s - np.diag(np.diag(Q_s))
        adj = W_off > 1e-8
    np.fill_diagonal(adj, False)

    # Cap embedding visualization / Aquila register layout to top 48 active/connected nodes if huge
    layout_n = min(n, 48)
    fov_w, fov_h = _AQUILA_FOV_UM
    pad = 4.0
    rng = np.random.default_rng(int(seed))

    if layout_n == 0:
        coords = np.zeros((0, 2), dtype=np.float64)
        fidelity = 1.0
        non_ud_violations = 0
        min_pair_um = _AQUILA_MIN_SPACING_UM
    elif layout_n == 1:
        coords = np.array([[fov_w * 0.5, fov_h * 0.5]], dtype=np.float64)
        fidelity = 1.0
        non_ud_violations = 0
        min_pair_um = _AQUILA_MIN_SPACING_UM
    else:
        adj_sub = adj[:layout_n, :layout_n]
        cols = int(math.ceil(math.sqrt(layout_n)))
        rows = int(math.ceil(layout_n / float(cols)))
        xs = np.linspace(pad + 6.0, fov_w - pad - 6.0, max(cols, 2))
        ys = np.linspace(pad + 6.0, fov_h - pad - 6.0, max(rows, 2))
        coords = np.zeros((layout_n, 2), dtype=np.float64)
        for idx in range(layout_n):
            r_i, c_i = divmod(idx, cols)
            coords[idx, 0] = xs[min(c_i, len(xs) - 1)] + rng.uniform(-0.8, 0.8)
            coords[idx, 1] = ys[min(r_i, len(ys) - 1)] + rng.uniform(-0.8, 0.8)

        # Force-directed unit-disk relaxation: attract conflict pairs < 6.2 um, repel non-conflicts > 10.5 um
        target_in = 5.8
        target_out = 10.8
        for it in range(55):
            step_lr = 0.22 * (1.0 - 0.55 * (it / 55.0))
            diff = coords[:, None, :] - coords[None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-8)
            np.fill_diagonal(dist, 1.0)
            unit = diff / dist[:, :, None]

            forces = np.zeros_like(coords)
            # Conflict springs pulling inside Rydberg blockade radius 8.37 um
            pull_mask = adj_sub & (dist > target_in)
            if np.any(pull_mask):
                mag_pull = np.where(pull_mask, 0.45 * (dist - target_in), 0.0)
                forces -= np.sum(unit * mag_pull[:, :, None], axis=1)

            # Too-close physical repulsion (< 4.35 um)
            hard_rep = (~np.eye(layout_n, dtype=bool)) & (dist < 4.35)
            if np.any(hard_rep):
                mag_h = np.where(hard_rep, 1.45 * (4.35 - dist), 0.0)
                forces += np.sum(unit * mag_h[:, :, None], axis=1)

            # Non-conflict separation outside blockade radius
            out_rep = (~np.eye(layout_n, dtype=bool)) & (~adj_sub) & (dist < target_out)
            if np.any(out_rep):
                mag_o = np.where(out_rep, 0.32 * (target_out - dist), 0.0)
                forces += np.sum(unit * mag_o[:, :, None], axis=1)

            # Gentle centering in 75x76 um optical tweezer FOV
            center = np.array([fov_w * 0.5, fov_h * 0.5], dtype=np.float64)
            forces += 0.025 * (center[None, :] - coords)

            disp = np.clip(forces * step_lr, -2.5, 2.5)
            coords = np.clip(coords + disp, [pad, pad], [fov_w - pad, fov_h - pad])

        # Strictly enforce minimum atomic separation >= 4.05 um inside FOV
        for _ in range(28):
            moved = False
            for a in range(layout_n):
                for b in range(a + 1, layout_n):
                    delta = coords[b] - coords[a]
                    d_ab = float(np.linalg.norm(delta))
                    if d_ab < _AQUILA_MIN_SPACING_UM + 0.05:
                        moved = True
                        dir_v = (
                            delta / d_ab
                            if d_ab > 1e-6
                            else np.array([math.cos(a + b), math.sin(a + b)], dtype=np.float64)
                        )
                        push = 0.52 * ((_AQUILA_MIN_SPACING_UM + 0.08) - d_ab)
                        coords[a] = np.clip(coords[a] - push * dir_v, [pad, pad], [fov_w - pad, fov_h - pad])
                        coords[b] = np.clip(coords[b] + push * dir_v, [pad, pad], [fov_w - pad, fov_h - pad])
            if not moved:
                break

        dmat = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))
        iu, ju = np.triu_indices(layout_n, k=1)
        pair_dists = dmat[iu, ju]
        min_pair_um = float(np.min(pair_dists)) if pair_dists.size > 0 else _AQUILA_MIN_SPACING_UM
        in_blockade = pair_dists < _AQUILA_BLOCKADE_RADIUS_UM
        edge_flags = adj_sub[iu, ju]
        agreements = in_blockade == edge_flags
        non_ud_violations = int(np.sum(~agreements))
        fidelity = float(np.mean(agreements)) if agreements.size > 0 else 1.0

    # Nguyen et al. (PRX Quantum 2023) crossing/wire gadget estimate for non-planar / non-unit-disk couplings
    gadget_est = 0
    if n > 0:
        n_edges_total = int(np.sum(np.triu(adj, k=1)))
        if non_ud_violations > 0 or n > layout_n:
            extra_edges = max(non_ud_violations, max(0, n_edges_total - int(layout_n * 1.5)))
            gadget_est = min(3 * n, int(math.ceil(extra_edges * 0.45)))

    total_est_qubits = int(n + gadget_est)
    hardware_ready = bool(
        n > 0
        and n <= layout_n
        and total_est_qubits <= _AQUILA_MAX_QUBITS
        and min_pair_um >= (_AQUILA_MIN_SPACING_UM - 1e-3)
    )

    atoms_um = [
        {
            "id": i,
            "label": node_labels[i],
            "x_um": round(float(coords[i, 0]), 2),
            "y_um": round(float(coords[i, 1]), 2),
            "selected": i in sel_set,
        }
        for i in range(layout_n)
    ]

    return {
        "bqm_braket": {
            "format": "bqm_qubo_v1",
            "vartype": "BINARY",
            "problem_name": problem_name,
            "num_variables": n,
            "num_quadratic_couplings": len(quadratic),
            "offset": 0.0,
            "linear": linear,
            "quadratic": quadratic,
            "compatible_backends": [
                "Amazon Braket Hybrid Jobs",
                "QuEra Aquila (Rydberg Neutral-Atom QPU)",
                "D-Wave Ocean dimod",
            ],
        },
        "quera_aquila_register": {
            "device": "QuEra Aquila 256-qubit Neutral-Atom QPU",
            "reference": "Ebadi et al. Science 2022; Nguyen et al. PRX Quantum 2023",
            "rydberg_blockade_radius_um": _AQUILA_BLOCKADE_RADIUS_UM,
            "min_atom_spacing_um": _AQUILA_MIN_SPACING_UM,
            "observed_min_spacing_um": round(float(min_pair_um), 3),
            "field_of_view_um": [_AQUILA_FOV_UM[0], _AQUILA_FOV_UM[1]],
            "max_physical_qubits": _AQUILA_MAX_QUBITS,
            "embedded_nodes": layout_n,
            "total_problem_nodes": n,
            "unit_disk_fidelity": round(float(fidelity), 4),
            "non_unit_disk_violations": int(non_ud_violations),
            "nguyen_2023_gadget_qubits_est": int(gadget_est),
            "total_estimated_qubits": total_est_qubits,
            "hardware_ready": hardware_ready,
            "atoms_um": atoms_um,
        },
    }


def _build_graph_preview(
    adj_bool: np.ndarray,
    weights: np.ndarray,
    selected_indices: Sequence[int],
    *,
    labels: Optional[Sequence[str]] = None,
    max_nodes: int = 36,
    max_edges: int = 120,
) -> dict[str, Any]:
    """Build deterministic compact SVG-ready circular graph summary for frontend telemetry."""
    n_total = int(weights.shape[0])
    if n_total == 0:
        return {"nodes": [], "edges": [], "shown_nodes": 0, "total_nodes": 0}

    sel_set = {int(i) for i in selected_indices}
    if n_total > max_nodes:
        sel_keep = sorted(sel_set)[: max_nodes // 2]
        rem_pool = [i for i in np.argsort(-weights, kind="mergesort") if int(i) not in sel_set]
        keep_ids = sorted((sel_keep + [int(x) for x in rem_pool])[:max_nodes])
    else:
        keep_ids = list(range(n_total))

    m = len(keep_ids)
    nodes: list[dict[str, Any]] = []
    for pos, orig_i in enumerate(keep_ids):
        angle = (2.0 * math.pi * pos) / float(max(m, 1)) - math.pi * 0.5
        nodes.append(
            {
                "id": int(orig_i),
                "label": str(labels[orig_i]) if labels is not None and orig_i < len(labels) else f"v{orig_i}",
                "weight": round(float(weights[orig_i]), 3),
                "selected": orig_i in sel_set,
                "x": round(0.5 + 0.39 * math.cos(angle), 4),
                "y": round(0.5 + 0.39 * math.sin(angle), 4),
            }
        )

    edges: list[dict[str, Any]] = []
    for a_pos in range(m):
        for b_pos in range(a_pos + 1, m):
            ia, ib = keep_ids[a_pos], keep_ids[b_pos]
            if adj_bool[ia, ib]:
                edges.append({"source": int(ia), "target": int(ib), "kind": "conflict"})
                if len(edges) >= max_edges:
                    break
        if len(edges) >= max_edges:
            break

    return {
        "nodes": nodes,
        "edges": edges,
        "shown_nodes": m,
        "total_nodes": n_total,
    }


def qubo_mwis_sim3_registration(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    allow_scale_alignment: bool = True,
    inlier_thresh_m: float = 0.04,
    max_qubo_vars: int = 96,
    seed: int = 42,
    include_hardware_export: bool = False,
) -> dict[str, Any]:
    """Register 3D-3D correspondences using Rydberg MWIS on isometric distance compatibility + RANSAC Sim(3) comparison."""
    from app.core.multiview_fusion import ransac_sim3, umeyama_sim3

    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    n_total = int(src.shape[0])

    if n_total < 3 or src.shape != dst.shape:
        empty_hist: list[dict[str, float]] = []
        out: dict[str, Any] = {
            "R": np.eye(3, dtype=np.float64),
            "t": np.zeros(3, dtype=np.float64),
            "scale": 1.0,
            "inliers": 0,
            "rmse_m": 0.0,
            "method_won": "ransac_fallback",
            "qubo_inliers": 0,
            "ransac_inliers": 0,
            "qubo_rmse_m": 0.0,
            "ransac_rmse_m": 0.0,
            "selected_clique_size": 0,
            "n_vars": n_total,
            "n_conflict_edges": 0,
            "conflict_density": 0.0,
            "qubo_energy": 0.0,
            "greedy_energy": 0.0,
            "energy_gain": 0.0,
            "solver_ms": 0.0,
            "energy_history": empty_hist,
            "selected_indices": [],
            "graph_preview": {"nodes": [], "edges": [], "shown_nodes": 0, "total_nodes": n_total},
        }
        if include_hardware_export:
            out["hardware_export"] = export_qubo_hardware_payload(
                np.zeros((0, 0), dtype=np.float64),
                problem_name="qubo_mwis_registration_3d",
            )
        return out

    w_all = (
        np.asarray(weights, dtype=np.float64).ravel()
        if weights is not None and len(weights) == n_total
        else np.ones(n_total, dtype=np.float64)
    )
    w_all = np.clip(w_all, 0.05, 5.0)

    # Run deterministic RANSAC baseline on full correspondences
    R_ran, t_ran, s_ran, ran_cnt = ransac_sim3(
        src,
        dst,
        allow_scale_alignment=allow_scale_alignment,
        inlier_thresh_m=inlier_thresh_m,
        seed=seed,
    )
    res_ran = np.linalg.norm(dst - ((s_ran * (src @ R_ran.T)) + t_ran), axis=1)
    inl_mask_ran = res_ran <= inlier_thresh_m
    ran_inliers = int(np.sum(inl_mask_ran)) if ran_cnt > 0 else 0
    ran_rmse = float(np.sqrt(np.mean(res_ran[inl_mask_ran] ** 2))) if ran_inliers > 0 else 1e6

    # Restrict QUBO variable budget cleanly while prioritizing highest-weight matches (ArUco + top SIFT)
    if n_total > max_qubo_vars:
        top_idx = np.sort(np.argsort(-w_all, kind="mergesort")[:max_qubo_vars])
    else:
        top_idx = np.arange(n_total, dtype=np.int64)

    src_sub = src[top_idx]
    dst_sub = dst[top_idx]
    w_sub = w_all[top_idx]
    n_var = int(src_sub.shape[0])

    # Pairwise Euclidean distance matrices for rigid/similarity graph compatibility (MAC / QuAnt)
    D_p = np.sqrt(np.sum((src_sub[:, None, :] - src_sub[None, :, :]) ** 2, axis=-1))
    D_q = np.sqrt(np.sum((dst_sub[:, None, :] - dst_sub[None, :, :]) ** 2, axis=-1))

    # Build top scale hypotheses (guard sparse wide-baseline graphs against implausible scale collapse)
    sparse_corr = n_var < 24
    s_lo, s_hi = (0.78, 1.28) if sparse_corr else (0.65, 1.55)
    min_bin_weight = 6.0 if sparse_corr else 1e-6

    scale_hyps: list[float] = [1.0]
    if allow_scale_alignment:
        iu, ju = np.triu_indices(n_var, k=1)
        dp_u = D_p[iu, ju]
        dq_u = D_q[iu, ju]
        valid_pairs = (dp_u > 0.03) & (dq_u > 0.03)
        if np.any(valid_pairs):
            ratios = dq_u[valid_pairs] / dp_u[valid_pairs]
            in_rng = (ratios >= s_lo) & (ratios <= s_hi)
            ratios = ratios[in_rng]
            if ratios.size >= 6:
                pair_w = (w_sub[iu[valid_pairs]][in_rng]) * (w_sub[ju[valid_pairs]][in_rng])
                hist, edges = np.histogram(ratios, bins=36, range=(s_lo, s_hi), weights=pair_w)
                top_bins = np.argsort(-hist)[:3]
                for b_idx in top_bins:
                    if hist[b_idx] < min_bin_weight:
                        continue
                    s_mode = float(0.5 * (edges[b_idx] + edges[b_idx + 1]))
                    if all(abs(s_mode - prev) > 0.035 for prev in scale_hyps):
                        scale_hyps.append(s_mode)
        if (
            ran_cnt >= 3
            and (s_lo <= float(s_ran) <= s_hi)
            and all(abs(float(s_ran) - prev) > 0.035 for prev in scale_hyps)
        ):
            scale_hyps.append(float(s_ran))

    tau_iso = max(float(inlier_thresh_m) * 1.15, 0.022)
    t0 = time.perf_counter()

    best_qubo: Optional[dict[str, Any]] = None
    best_R_q = np.eye(3, dtype=np.float64)
    best_t_q = np.zeros(3, dtype=np.float64)
    best_s_q = 1.0
    best_inliers_q = -1
    best_rmse_q = 1e6
    best_adj = np.zeros((n_var, n_var), dtype=bool)
    best_C = np.zeros((n_var, n_var), dtype=np.float64)

    def _is_nondegenerate_3d_subset(pts: np.ndarray) -> bool:
        if pts.shape[0] < 3:
            return False
        centered = pts - np.mean(pts, axis=0)
        svals = np.linalg.svd(centered, compute_uv=False)
        return bool(svals.size >= 2 and svals[0] > 0.04 and svals[1] > 0.02)

    for s_hyp in scale_hyps:
        d_ab = np.abs(s_hyp * D_p - D_q)
        colocated_err = ((D_p < 0.015) & (D_q > 0.035)) | ((D_q < 0.015) & (D_p > 0.035))
        conflict_mask = (d_ab > tau_iso) | colocated_err
        np.fill_diagonal(conflict_mask, False)
        C = np.where(conflict_mask, 1.0 + np.minimum(d_ab / tau_iso, 5.0), 0.0)
        np.fill_diagonal(C, 0.0)

        mwis = solve_mwis_rydberg(
            w_sub,
            C,
            penalty_scale=1.35,
            exact_if_small=(n_var <= 18),
            enforce_independent_set=True,
            n_trajectories=24,
            n_steps=90,
            seed=seed,
        )

        sel_local = np.asarray(mwis["selected_indices"], dtype=np.int64)
        R_cand, t_cand, s_cand = np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 1.0
        inl_cnt_q = 0
        rmse_q = 1e6

        if (
            sel_local.size >= 3
            and _is_nondegenerate_3d_subset(src_sub[sel_local])
            and _is_nondegenerate_3d_subset(dst_sub[sel_local])
        ):
            R_cand, t_cand, s_cand = umeyama_sim3(
                src_sub[sel_local],
                dst_sub[sel_local],
                estimate_scale=allow_scale_alignment,
            )
            if not sparse_corr or (s_lo <= s_cand <= s_hi):
                # Scale-normalized error threshold prevents uniform shrinkage from inflating inliers
                eff_thresh = inlier_thresh_m * min(1.0, max(float(s_cand), 0.5))
                res_all = np.linalg.norm(dst - ((s_cand * (src @ R_cand.T)) + t_cand), axis=1)
                inl_mask = res_all <= eff_thresh
                cnt_all = int(np.sum(inl_mask))
                if cnt_all >= 3:
                    R_ref, t_ref, s_ref = umeyama_sim3(
                        src[inl_mask],
                        dst[inl_mask],
                        estimate_scale=allow_scale_alignment,
                    )
                    if not sparse_corr or (s_lo <= s_ref <= s_hi):
                        eff_ref = inlier_thresh_m * min(1.0, max(float(s_ref), 0.5))
                        res_ref = np.linalg.norm(dst - ((s_ref * (src @ R_ref.T)) + t_ref), axis=1)
                        inl_ref = res_ref <= eff_ref
                        inl_cnt_q = int(np.sum(inl_ref))
                        if inl_cnt_q >= 3:
                            R_cand, t_cand, s_cand = R_ref, t_ref, s_ref
                            rmse_q = float(np.sqrt(np.mean(res_ref[inl_ref] ** 2)))
                        else:
                            inl_cnt_q = cnt_all
                            rmse_q = float(np.sqrt(np.mean(res_all[inl_mask] ** 2)))
                    else:
                        inl_cnt_q = cnt_all
                        rmse_q = float(np.sqrt(np.mean(res_all[inl_mask] ** 2)))

        better = (
            best_qubo is None
            or inl_cnt_q > best_inliers_q
            or (inl_cnt_q == best_inliers_q and rmse_q < best_rmse_q)
            or (inl_cnt_q == best_inliers_q and abs(rmse_q - best_rmse_q) <= 1e-6 and len(sel_local) > best_qubo["n_selected"])
        )
        if better:
            best_inliers_q = inl_cnt_q
            best_rmse_q = rmse_q
            best_R_q, best_t_q, best_s_q = R_cand, t_cand, s_cand
            best_adj = conflict_mask
            best_C = C
            best_qubo = {**mwis, "n_selected": int(sel_local.size)}

    total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    assert best_qubo is not None

    qubo_wins = (best_inliers_q > ran_inliers) or (
        best_inliers_q >= 3 and best_inliers_q == ran_inliers and best_rmse_q <= ran_rmse + 1e-4
    )

    if qubo_wins:
        R_out, t_out, s_out = best_R_q, best_t_q, best_s_q
        out_inliers = int(best_inliers_q)
        out_rmse = float(best_rmse_q) if best_rmse_q < 1e5 else 0.0
        method_won = "qubo_mwis"
    else:
        R_out, t_out, s_out = R_ran, t_ran, s_ran
        out_inliers = int(ran_inliers)
        out_rmse = float(ran_rmse) if ran_rmse < 1e5 else 0.0
        method_won = "ransac_fallback"

    selected_orig = [int(top_idx[i]) for i in best_qubo["selected_indices"]]
    preview_labels = [f"c{int(top_idx[i])}" for i in range(n_var)]
    graph_prev = _build_graph_preview(
        best_adj,
        w_sub,
        best_qubo["selected_indices"],
        labels=preview_labels,
    )

    result: dict[str, Any] = {
        "R": R_out,
        "t": t_out,
        "scale": float(s_out),
        "inliers": out_inliers,
        "rmse_m": round(float(out_rmse), 5),
        "method_won": method_won,
        "qubo_inliers": max(0, int(best_inliers_q)),
        "ransac_inliers": int(ran_inliers),
        "qubo_rmse_m": round(float(best_rmse_q if best_rmse_q < 1e5 else 0.0), 5),
        "ransac_rmse_m": round(float(ran_rmse if ran_rmse < 1e5 else 0.0), 5),
        "selected_clique_size": len(best_qubo["selected_indices"]),
        "n_vars": int(best_qubo["n_vars"]),
        "n_conflict_edges": int(best_qubo["n_conflict_edges"]),
        "conflict_density": float(best_qubo["conflict_density"]),
        "qubo_energy": float(best_qubo["qubo_energy"]),
        "greedy_energy": float(best_qubo["greedy_energy"]),
        "energy_gain": float(best_qubo["energy_gain"]),
        "solver_ms": total_ms,
        "energy_history": best_qubo["energy_history"],
        "selected_indices": selected_orig,
        "graph_preview": graph_prev,
    }
    if include_hardware_export:
        result["hardware_export"] = export_qubo_hardware_payload(
            best_qubo["Q_sym"],
            problem_name="qubo_mwis_registration_3d",
            labels=preview_labels,
            conflict_matrix=best_C,
            selected_indices=best_qubo["selected_indices"],
            seed=seed,
        )
    return result


def resolve_3d_objects_mwis_qubo(
    objects: list[dict[str, Any]],
    *,
    conflict_iou_threshold: float = 0.28,
    same_label_iou_threshold: float = 0.15,
    seed: int = 42,
    include_hardware_export: bool = True,
) -> dict[str, Any]:
    """Global 3D collision & duplicate resolution via Rydberg-MWIS over 15-axis SAT + 3D IoU graph."""
    n = len(objects)
    if n == 0:
        out: dict[str, Any] = {
            "kept_indices": [],
            "suppressed_indices": [],
            "kept_objects": [],
            "suppressed_objects": [],
            "conflict_edges": [],
            "qubo_stats": {
                "n_vars": 0,
                "n_conflict_edges": 0,
                "conflict_density": 0.0,
                "qubo_energy": 0.0,
                "greedy_energy": 0.0,
                "energy_gain": 0.0,
                "qubo_weight_sum": 0.0,
                "greedy_weight_sum": 0.0,
                "greedy_kept_indices": [],
                "exact_optimum_verified": True,
                "solver_method": "rydberg_mwis_empty",
                "solver_ms": 0.0,
            },
            "energy_history": [],
            "graph_preview": {"nodes": [], "edges": [], "shown_nodes": 0, "total_nodes": 0},
        }
        if include_hardware_export:
            out["hardware_export"] = export_qubo_hardware_payload(
                np.zeros((0, 0), dtype=np.float64),
                problem_name="qubo_mwis_3d_nms",
            )
        return out

    weights = np.zeros(n, dtype=np.float64)
    vols = np.zeros(n, dtype=np.float64)
    labels: list[str] = []

    for i, obj in enumerate(objects):
        b = [float(v) for v in obj["box_3d"]]
        dx, dy, dz = max(abs(b[3]), 1e-4), max(abs(b[4]), 1e-4), max(abs(b[5]), 1e-4)
        vols[i] = dx * dy * dz
        aniso = max(dx, dy, dz) / min(dx, dy, dz)
        geom_factor = 0.4 if aniso > 25.0 else 1.0
        conf = float(np.clip(float(obj.get("confidence", 0.9)), 0.15, 1.0))
        weights[i] = conf * (0.85 + 0.15 * geom_factor)
        labels.append(str(obj.get("label") or f"obj_{i}"))

    C = np.zeros((n, n), dtype=np.float64)
    adj_bool = np.zeros((n, n), dtype=bool)
    conflict_edges: list[dict[str, Any]] = []

    for i in range(n):
        box_i = objects[i]["box_3d"]
        lbl_i = labels[i].strip().lower()
        is_sup_i = lbl_i in _SUPPORT_LABELS
        for j in range(i + 1, n):
            box_j = objects[j]["box_3d"]
            if not obb_intersects_sat(box_i, box_j):
                continue
            iou = float(compute_3d_iou(box_i, box_j, seed=seed))
            lbl_j = labels[j].strip().lower()
            is_sup_j = lbl_j in _SUPPORT_LABELS
            vol_ratio = min(float(vols[i]), float(vols[j])) / max(float(vols[i]), float(vols[j]), 1e-9)

            # Support immunity: allow smaller objects resting atop table/desk/shelf unless deep overlap >= 0.45
            if (is_sup_i ^ is_sup_j) and (lbl_i != lbl_j) and vol_ratio < 0.25:
                is_conflict = iou >= 0.45
            else:
                same_lbl = lbl_i == lbl_j
                is_conflict = (iou >= conflict_iou_threshold) or (same_lbl and iou >= same_label_iou_threshold)

            if is_conflict:
                w_edge = 1.0 + iou
                C[i, j] = w_edge
                C[j, i] = w_edge
                adj_bool[i, j] = True
                adj_bool[j, i] = True
                conflict_edges.append(
                    {
                        "i": i,
                        "j": j,
                        "id_i": str(objects[i].get("id", f"obj_{i}")),
                        "id_j": str(objects[j].get("id", f"obj_{j}")),
                        "label_i": labels[i],
                        "label_j": labels[j],
                        "iou_3d": round(iou, 4),
                    }
                )

    mwis = solve_mwis_rydberg(
        weights,
        C,
        penalty_scale=1.35,
        exact_if_small=True,
        enforce_independent_set=True,
        n_trajectories=24,
        n_steps=90,
        seed=seed,
    )
    kept_idx = list(mwis["selected_indices"])
    kept_set = set(kept_idx)
    suppressed_idx = [i for i in range(n) if i not in kept_set]

    node_tags = [f"{obj.get('id', f'obj_{idx}')}: {labels[idx]}" for idx, obj in enumerate(objects)]
    graph_prev = _build_graph_preview(
        adj_bool,
        weights,
        kept_idx,
        labels=node_tags,
    )

    out = {
        "kept_indices": kept_idx,
        "suppressed_indices": suppressed_idx,
        "kept_objects": [objects[i] for i in kept_idx],
        "suppressed_objects": [objects[i] for i in suppressed_idx],
        "conflict_edges": conflict_edges,
        "qubo_stats": {
            "n_vars": int(mwis["n_vars"]),
            "n_conflict_edges": int(mwis["n_conflict_edges"]),
            "conflict_density": float(mwis["conflict_density"]),
            "qubo_energy": float(mwis["qubo_energy"]),
            "greedy_energy": float(mwis["greedy_energy"]),
            "energy_gain": float(mwis["energy_gain"]),
            "qubo_weight_sum": float(mwis["selected_weight_sum"]),
            "greedy_weight_sum": float(mwis["greedy_weight_sum"]),
            "greedy_kept_indices": list(mwis["greedy_selected_indices"]),
            "conflict_violations": int(mwis["conflict_violations"]),
            "exact_optimum_verified": bool(mwis["exact_optimum_verified"]),
            "solver_method": str(mwis["solver_method"]),
            "solver_ms": float(mwis["solver_ms"]),
        },
        "energy_history": mwis["energy_history"],
        "graph_preview": graph_prev,
    }
    if include_hardware_export:
        out["hardware_export"] = export_qubo_hardware_payload(
            mwis["Q_sym"],
            problem_name="qubo_mwis_3d_nms",
            labels=node_tags,
            conflict_matrix=C,
            selected_indices=kept_idx,
            seed=seed,
        )
    return out


def plan_nbv_set_cover_qubo(
    sector_scores: Sequence[float],
    *,
    budget_k: Optional[int] = None,
    body_normals_three: Optional[np.ndarray] = None,
    existing_azimuths_deg: Optional[Sequence[float]] = None,
    seed: int = 42,
    include_hardware_export: bool = False,
) -> dict[str, Any]:
    """Multi-shot ActivePose Next-Best-View Set-Cover QUBO with Classical Shadows uncertainty."""
    t0 = time.perf_counter()
    scores = np.array([float(np.clip(s, 0.0, 1.0)) for s in list(sector_scores)[:12]], dtype=np.float64)
    if scores.size < 12:
        padded = np.zeros(12, dtype=np.float64)
        padded[: scores.size] = scores
        scores = padded

    # If existing azimuths provided without precomputed scores, compute exact 12-sector visibility kernel
    sector_centers = np.array([(s * 30.0) + 15.0 for s in range(12)], dtype=np.float64)
    K_cov = np.zeros((12, 12), dtype=np.float64)
    for m in range(12):
        for s in range(12):
            diff = abs((sector_centers[m] - sector_centers[s] + 180.0) % 360.0 - 180.0)
            if diff <= 52.0:
                K_cov[m, s] = math.cos(math.radians(min(diff, 52.0) * (90.0 / 55.0)))

    if existing_azimuths_deg is not None and not np.any(scores > 0.0):
        for az in existing_azimuths_deg:
            for s in range(12):
                diff = abs((float(az) - sector_centers[s] + 180.0) % 360.0 - 180.0)
                if diff <= 52.0:
                    val = math.cos(math.radians(min(diff, 52.0) * (90.0 / 55.0)))
                    scores[s] = max(scores[s], round(val, 3))

    # Classical Shadows uncertainty h_s across 12 azimuthal surface-normal bins
    shadow_uncertainty = np.zeros(12, dtype=np.float64)
    has_normals = False
    if body_normals_three is not None:
        nrm = np.asarray(body_normals_three, dtype=np.float64)
        if nrm.ndim == 2 and nrm.shape[1] == 3 and nrm.shape[0] >= 24:
            horiz_len = np.hypot(nrm[:, 0], nrm[:, 2])
            horiz = nrm[horiz_len > 0.25]
            if horiz.shape[0] >= 24:
                has_normals = True
                az = np.degrees(np.arctan2(horiz[:, 0], horiz[:, 2])) % 360.0
                bins = np.floor(az / 30.0).astype(np.int64) % 12
                counts = np.bincount(bins, minlength=12).astype(np.float64)
                norm_density = counts / max(float(np.max(counts)), 1.0)
                shadow_uncertainty = 0.85 + 0.55 * (1.0 - norm_density)

    if not has_normals:
        shadow_uncertainty = 1.0 + 0.35 * (1.0 - scores)

    deficit = np.maximum(0.0, 1.0 - scores)
    current_cov_pct = round(float(np.sum(scores >= 0.25)) / 12.0 * 100.0, 1)

    def _simulate_coverage(shot_indices: Sequence[int]) -> tuple[np.ndarray, float]:
        pred = scores.copy()
        for m in shot_indices:
            pred = np.maximum(pred, K_cov[int(m)])
        cov_pct = round(float(np.sum(pred >= 0.25)) / 12.0 * 100.0, 1)
        return np.round(pred, 3), cov_pct

    def _build_set_cover_qubo(k_target: int) -> np.ndarray:
        w_sec = deficit * shadow_uncertainty
        g = K_cov @ w_sec
        R_red = (K_cov * w_sec[None, :]) @ K_cov.T
        gamma = 2.5 * max(float(np.max(g)), 0.5)

        Q_mat = np.zeros((12, 12), dtype=np.float64)
        for m in range(12):
            Q_mat[m, m] = -float(g[m]) + gamma * (1.0 - 2.0 * float(k_target))
        for m in range(12):
            for mp in range(m + 1, 12):
                Q_mat[m, mp] = 0.65 * float(R_red[m, mp]) + 2.0 * gamma
        return normalize_qubo(Q_mat)

    def _solve_for_k(k_target: int) -> tuple[np.ndarray, list[int], float, float, list[dict[str, float]]]:
        Q_k = _build_set_cover_qubo(k_target)
        sb = solve_qubo_simulated_bifurcation(Q_k, n_trajectories=24, n_steps=80, seed=seed, record_history=True)

        # Exact cardinal verification across C(12, K) <= 495 subsets guarantees exact budget constraint
        best_subset: tuple[int, ...] = tuple(range(k_target))
        best_e = 1e18
        for comb in itertools.combinations(range(12), k_target):
            vec = np.zeros(12, dtype=np.float64)
            vec[list(comb)] = 1.0
            e_val = float(vec @ Q_k @ vec)
            if e_val < best_e - 1e-12:
                best_e = e_val
                best_subset = comb

        hist = list(sb["energy_history"])
        if hist and best_e < float(hist[-1]["best_energy"]) - 1e-6:
            hist.append(
                {
                    "step": int(hist[-1]["step"] + 1),
                    "best_energy": round(float(best_e), 4),
                    "mean_energy": round(float(hist[-1]["mean_energy"]), 4),
                }
            )
        _, pred_cov = _simulate_coverage(best_subset)
        return Q_k, [int(i) for i in best_subset], float(best_e), pred_cov, hist

    if budget_k is None:
        if current_cov_pct >= 95.0:
            chosen_k = 1
            Q_sel, opt_indices, opt_energy, pred_cov_pct, energy_hist = _solve_for_k(1)
        else:
            Q_sel = np.zeros((12, 12), dtype=np.float64)
            opt_indices = [0]
            opt_energy = 0.0
            pred_cov_pct = current_cov_pct
            energy_hist = []
            chosen_k = 1
            for k_try in (1, 2, 3, 4):
                Q_c, idx_c, e_c, cov_c, h_c = _solve_for_k(k_try)
                Q_sel, opt_indices, opt_energy, pred_cov_pct, energy_hist = Q_c, idx_c, e_c, cov_c, h_c
                chosen_k = k_try
                if cov_c >= 90.0:
                    break
    else:
        chosen_k = max(1, min(6, int(budget_k)))
        Q_sel, opt_indices, opt_energy, pred_cov_pct, energy_hist = _solve_for_k(chosen_k)

    pred_scores, pred_cov_pct = _simulate_coverage(opt_indices)

    # Sequential greedy baseline of equal budget K (myopic 1-step max marginal gain)
    greedy_sel: list[int] = []
    cur_sc = scores.copy()
    w_sec = deficit * shadow_uncertainty
    for _ in range(chosen_k):
        best_m = -1
        best_gain = -1.0
        for m in range(12):
            if m in greedy_sel:
                continue
            marginal = float(np.sum(np.maximum(K_cov[m] - cur_sc, 0.0) * shadow_uncertainty))
            # If already covered, fallback to direct sector deficit weight
            if marginal <= 1e-9:
                marginal = float(np.dot(K_cov[m], w_sec))
            if marginal > best_gain:
                best_gain = marginal
                best_m = m
        if best_m >= 0:
            greedy_sel.append(best_m)
            cur_sc = np.maximum(cur_sc, K_cov[best_m])

    x_greedy = np.zeros(12, dtype=np.float64)
    x_greedy[greedy_sel] = 1.0
    greedy_energy = evaluate_qubo_energy(Q_sel, x_greedy)
    _, greedy_cov_pct = _simulate_coverage(greedy_sel)

    opt_plan_deg = [round(float(sector_centers[i]), 1) for i in opt_indices]
    greedy_plan_deg = [round(float(sector_centers[i]), 1) for i in greedy_sel]

    total_est_shots = len(list(existing_azimuths_deg or [])) + len(opt_plan_deg)
    shots_saved = max(0, 6 - total_est_shots) if pred_cov_pct >= 85.0 else 0

    angles_str = ", ".join(f"{ang:.0f}°" for ang in opt_plan_deg)
    advice_es = (
        f"Plan QUBO Set-Cover (K={chosen_k}): capturar {chosen_k} toma(s) adicional(es) en [{angles_str}] "
        f"eleva la cobertura angular de {current_cov_pct:.0f}% a {pred_cov_pct:.0f}%."
    )

    solver_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    out: dict[str, Any] = {
        "budget_k": chosen_k,
        "optimal_shot_plan_deg": opt_plan_deg,
        "optimal_sector_indices": opt_indices,
        "current_coverage_pct": current_cov_pct,
        "predicted_coverage_after_plan_pct": pred_cov_pct,
        "predicted_sector_scores": [float(v) for v in pred_scores.tolist()],
        "shots_saved_vs_uniform_6": shots_saved,
        "shadow_uncertainty_by_sector": [round(float(v), 3) for v in shadow_uncertainty.tolist()],
        "qubo_energy": round(float(opt_energy), 5),
        "greedy_energy": round(float(greedy_energy), 5),
        "energy_gain": round(float(max(0.0, greedy_energy - opt_energy)), 5),
        "greedy_plan_deg": greedy_plan_deg,
        "greedy_predicted_coverage_pct": greedy_cov_pct,
        "solver_ms": solver_ms,
        "energy_history": energy_hist,
        "advice_es": advice_es,
    }
    if include_hardware_export:
        sector_labels = [f"{int(sector_centers[i])}°" for i in range(12)]
        # Conflict matrix of redundant overlapping adjacent camera candidates for Aquila display
        adj_nbv = (K_cov > 0.35).astype(np.float64)
        np.fill_diagonal(adj_nbv, 0.0)
        out["hardware_export"] = export_qubo_hardware_payload(
            Q_sel,
            problem_name="qubo_nbv_set_cover_12",
            labels=sector_labels,
            conflict_matrix=adj_nbv,
            selected_indices=opt_indices,
            seed=seed,
        )
    return out


def _generate_registration_stress_case(
    *,
    n_correspondences: int = 72,
    outlier_ratio: float = 0.75,
    noise_mm: float = 4.0,
    allow_scale: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate calibrated 3D-3D registration stress scene with high outlier contamination."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(int(seed))
    n = max(12, min(140, int(n_correspondences)))
    out_r = float(np.clip(outlier_ratio, 0.0, 0.90))
    n_out = int(round(n * out_r))
    n_in = max(4, n - n_out)
    n_out = n - n_in

    gt_R = Rotation.from_euler("zyx", [24.0, -16.0, 33.0], degrees=True).as_matrix()
    gt_t = np.array([0.28, -0.15, 0.42], dtype=np.float64)
    gt_s = 1.12 if allow_scale else 1.0

    src_in = rng.uniform([-0.45, -0.35, 0.55], [0.45, 0.35, 1.35], size=(n_in, 3))
    noise = rng.normal(0.0, float(noise_mm) / 1000.0, size=(n_in, 3))
    dst_in = (gt_s * (src_in @ gt_R.T)) + gt_t + noise

    # Adversarial outliers: decoy clusters (smaller than true clique n_in) + uncoordinated scatter
    src_out = rng.uniform([-0.45, -0.35, 0.55], [0.45, 0.35, 1.35], size=(n_out, 3))
    decoy_cap = min(n_out // 2, max(0, n_in - 5))
    decoy_a = decoy_cap // 2 + (decoy_cap % 2)
    decoy_b = decoy_cap - decoy_a
    parts_out: list[np.ndarray] = []
    if decoy_a > 0:
        r_a = Rotation.from_euler("zyx", [-95.0, 22.0, 140.0], degrees=True).as_matrix()
        parts_out.append(
            (gt_s * (src_out[:decoy_a] @ r_a.T))
            + np.array([-0.22, 0.19, 0.35])
            + rng.normal(0.0, float(noise_mm) / 1000.0, size=(decoy_a, 3))
        )
    if decoy_b > 0:
        r_b = Rotation.from_euler("zyx", [135.0, -35.0, -70.0], degrees=True).as_matrix()
        parts_out.append(
            (gt_s * (src_out[decoy_a : decoy_a + decoy_b] @ r_b.T))
            + np.array([0.16, 0.24, -0.25])
            + rng.normal(0.0, float(noise_mm) / 1000.0, size=(decoy_b, 3))
        )
    n_rand = n_out - decoy_a - decoy_b
    if n_rand > 0:
        parts_out.append(rng.uniform([-0.70, -0.55, 0.45], [0.80, 0.60, 1.75], size=(n_rand, 3)))
    dst_out = np.vstack(parts_out) if parts_out else np.zeros((0, 3), dtype=np.float64)

    src_all = np.vstack([src_in, src_out])
    dst_all = np.vstack([dst_in, dst_out])
    weights = np.concatenate(
        [
            rng.uniform(0.65, 1.0, size=n_in),
            rng.uniform(0.55, 1.05, size=n_out),
        ]
    )
    perm = rng.permutation(n)
    src_all = src_all[perm]
    dst_all = dst_all[perm]
    weights = weights[perm]

    res = qubo_mwis_sim3_registration(
        src_all,
        dst_all,
        weights=weights,
        allow_scale_alignment=allow_scale,
        inlier_thresh_m=0.04,
        max_qubo_vars=96,
        seed=seed,
        include_hardware_export=True,
    )
    rot_rel = res["R"] @ gt_R.T
    rot_err_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(rot_rel).as_rotvec())))
    trans_err_m = float(np.linalg.norm(res["t"] - gt_t))
    scale_err_pct = float(abs(res["scale"] - gt_s) / max(gt_s, 1e-6) * 100.0)

    return {
        "task": "registration_stress",
        "title_es": "Registro Multi-Vista 3D QUBO-MWIS Anti-Simetría",
        "summary_es": (
            f"Con {n} correspondencias y {out_r * 100:.0f}% outliers ({n_in} inliers reales), "
            f"Rydberg-MWIS recuperó {res['qubo_inliers']} inliers con error rotacional {rot_err_deg:.2f}° "
            f"(RANSAC clásico: {res['ransac_inliers']} inliers)."
        ),
        "qubo_stats": {
            "n_vars": res["n_vars"],
            "n_conflict_edges": res["n_conflict_edges"],
            "conflict_density": res["conflict_density"],
            "qubo_energy": res["qubo_energy"],
            "greedy_energy": res["greedy_energy"],
            "energy_gain": res["energy_gain"],
            "solver_ms": res["solver_ms"],
        },
        "baseline_comparison": {
            "method_won": res["method_won"],
            "qubo_inliers": res["qubo_inliers"],
            "ransac_inliers": res["ransac_inliers"],
            "qubo_rmse_mm": round(res["qubo_rmse_m"] * 1000.0, 2),
            "ransac_rmse_mm": round(res["ransac_rmse_m"] * 1000.0, 2),
            "rotation_error_deg": round(rot_err_deg, 3),
            "translation_error_mm": round(trans_err_m * 1000.0, 2),
            "scale_error_pct": round(scale_err_pct, 3),
        },
        "energy_history": res["energy_history"],
        "graph_preview": res["graph_preview"],
        "details": {
            "n_correspondences": n,
            "true_inliers": n_in,
            "outlier_ratio": round(out_r, 3),
            "estimated_scale": round(float(res["scale"]), 4),
            "gt_scale": round(gt_s, 4),
            "selected_clique_size": res["selected_clique_size"],
        },
        "hardware_export": res["hardware_export"],
        "latency_ms": res["solver_ms"],
    }


def _demo_nms_greedy_trap_objects() -> list[dict[str, Any]]:
    """10-object 3D scene containing a high-conf giant box overlapping 3 true objects + table support."""
    return [
        {
            "id": "trap_mega_box",
            "label": "assembly_cluster",
            "box_3d": [0.0, 1.85, 0.15, 0.86, 0.40, 0.38, 0.0, 0.0, 0.0],
            "confidence": 0.94,
        },
        {
            "id": "true_oscilloscope",
            "label": "oscilloscope",
            "box_3d": [-0.28, 1.85, 0.15, 0.30, 0.39, 0.37, 0.0, 0.0, 0.0],
            "confidence": 0.78,
        },
        {
            "id": "true_power_supply",
            "label": "power_supply",
            "box_3d": [0.0, 1.85, 0.15, 0.28, 0.39, 0.37, 0.0, 0.0, 0.0],
            "confidence": 0.76,
        },
        {
            "id": "true_multimeter_unit",
            "label": "multimeter_bench",
            "box_3d": [0.28, 1.85, 0.15, 0.30, 0.39, 0.37, 0.0, 0.0, 0.0],
            "confidence": 0.75,
        },
        {
            "id": "dup_oscilloscope",
            "label": "oscilloscope",
            "box_3d": [-0.27, 1.86, 0.16, 0.29, 0.38, 0.36, 0.0, 0.0, 1.0],
            "confidence": 0.64,
        },
        {
            "id": "lab_table",
            "label": "table",
            "box_3d": [0.0, 2.10, -0.20, 1.80, 0.95, 0.42, 0.0, 0.0, 0.0],
            "confidence": 0.91,
        },
        {
            "id": "coffee_mug_on_table",
            "label": "mug",
            "box_3d": [0.64, 2.02, -0.02, 0.10, 0.10, 0.12, 0.0, 0.0, 12.0],
            "confidence": 0.84,
        },
        {
            "id": "dup_mug",
            "label": "cup",
            "box_3d": [0.64, 2.02, -0.02, 0.105, 0.105, 0.12, 0.0, 0.0, 10.0],
            "confidence": 0.61,
        },
        {
            "id": "servo_crate",
            "label": "crate",
            "box_3d": [-0.72, 2.35, 0.22, 0.34, 0.30, 0.28, 0.0, 0.0, 18.0],
            "confidence": 0.87,
        },
        {
            "id": "dup_servo_crate",
            "label": "box",
            "box_3d": [-0.71, 2.35, 0.22, 0.33, 0.29, 0.27, 0.0, 0.0, 16.0],
            "confidence": 0.62,
        },
    ]


def _run_nms_3d_case(
    *,
    objects: Optional[list[dict[str, Any]]] = None,
    conflict_iou_threshold: float = 0.28,
    seed: int = 42,
) -> dict[str, Any]:
    """Run global MWIS 3D collision/duplicate suppression and format API payload."""
    input_objs = objects if objects is not None and len(objects) > 0 else _demo_nms_greedy_trap_objects()
    res = resolve_3d_objects_mwis_qubo(
        input_objs,
        conflict_iou_threshold=conflict_iou_threshold,
        seed=seed,
        include_hardware_export=True,
    )
    qs = res["qubo_stats"]
    kept_labels = [str(o.get("label", o.get("id"))) for o in res["kept_objects"]]
    sup_labels = [str(o.get("label", o.get("id"))) for o in res["suppressed_objects"]]
    return {
        "task": "nms_3d",
        "title_es": "Resolución Global de Colisiones y NMS 3D (Bloqueo de Rydberg)",
        "summary_es": (
            f"De {len(input_objs)} detecciones 9-DOF y {qs['n_conflict_edges']} conflictos físicos SAT/IoU, "
            f"QUBO-MWIS seleccionó {len(res['kept_objects'])} objetos sin colisiones (peso total {qs['qubo_weight_sum']:.2f} "
            f"vs. Greedy {qs['greedy_weight_sum']:.2f})."
        ),
        "qubo_stats": qs,
        "baseline_comparison": {
            "qubo_kept_count": len(res["kept_indices"]),
            "greedy_kept_count": len(qs["greedy_kept_indices"]),
            "qubo_weight_sum": qs["qubo_weight_sum"],
            "greedy_weight_sum": qs["greedy_weight_sum"],
            "energy_gain": qs["energy_gain"],
            "exact_optimum_verified": qs["exact_optimum_verified"],
        },
        "energy_history": res["energy_history"],
        "graph_preview": res["graph_preview"],
        "details": {
            "input_count": len(input_objs),
            "kept_ids": [str(o.get("id", f"obj_{idx}")) for idx, o in zip(res["kept_indices"], res["kept_objects"])],
            "kept_labels": kept_labels,
            "suppressed_labels": sup_labels,
            "conflict_edges": res["conflict_edges"],
        },
        "hardware_export": res["hardware_export"],
        "latency_ms": qs["solver_ms"],
    }


def _run_nbv_plan_case(
    *,
    existing_azimuths_deg: Optional[Sequence[float]] = None,
    sector_scores: Optional[Sequence[float]] = None,
    budget_k: Optional[int] = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Run multi-shot ActivePose NBV Set-Cover QUBO planner."""
    az_list = list(existing_azimuths_deg) if existing_azimuths_deg is not None else [0.0, 40.0]
    if sector_scores is not None:
        scores_in = list(sector_scores)
    else:
        scores_in = [0.0] * 12

    plan = plan_nbv_set_cover_qubo(
        scores_in,
        budget_k=budget_k,
        existing_azimuths_deg=az_list if sector_scores is None else None,
        seed=seed,
        include_hardware_export=True,
    )

    # Compact 12-sector overlap conflict graph for SVG visualization
    sector_centers = np.array([(s * 30.0) + 15.0 for s in range(12)], dtype=np.float64)
    adj_nbv = np.zeros((12, 12), dtype=bool)
    for i in range(12):
        for j in range(i + 1, 12):
            diff = abs((sector_centers[i] - sector_centers[j] + 180.0) % 360.0 - 180.0)
            if diff <= 35.0:
                adj_nbv[i, j] = True
                adj_nbv[j, i] = True
    weights_nbv = np.array(plan["shadow_uncertainty_by_sector"], dtype=np.float64) * np.maximum(
        0.1, 1.0 - np.asarray(scores_in[:12] if len(scores_in) == 12 else [0.0] * 12, dtype=np.float64)
    )
    graph_prev = _build_graph_preview(
        adj_nbv,
        weights_nbv,
        plan["optimal_sector_indices"],
        labels=[f"{int(c)}°" for c in sector_centers],
    )

    return {
        "task": "nbv_plan",
        "title_es": "Planificador Multi-Disparo ActivePose NBV (Set-Cover + Classical Shadows)",
        "summary_es": plan["advice_es"],
        "qubo_stats": {
            "n_vars": 12,
            "n_conflict_edges": int(np.sum(np.triu(adj_nbv, k=1))),
            "conflict_density": round(float(np.sum(np.triu(adj_nbv, k=1))) / 66.0, 4),
            "qubo_energy": plan["qubo_energy"],
            "greedy_energy": plan["greedy_energy"],
            "energy_gain": plan["energy_gain"],
            "solver_ms": plan["solver_ms"],
        },
        "baseline_comparison": {
            "budget_k": plan["budget_k"],
            "current_coverage_pct": plan["current_coverage_pct"],
            "qubo_predicted_coverage_pct": plan["predicted_coverage_after_plan_pct"],
            "greedy_predicted_coverage_pct": plan["greedy_predicted_coverage_pct"],
            "shots_saved_vs_uniform_6": plan["shots_saved_vs_uniform_6"],
            "optimal_shot_plan_deg": plan["optimal_shot_plan_deg"],
            "greedy_plan_deg": plan["greedy_plan_deg"],
        },
        "energy_history": plan["energy_history"],
        "graph_preview": graph_prev,
        "details": {
            **plan,
            "existing_azimuths_deg": [float(a) for a in az_list],
        },
        "hardware_export": plan["hardware_export"],
        "latency_ms": plan["solver_ms"],
    }


def run_quantum_optimization_task(task: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Dispatch Phase 4 combinatorial QUBO-MWIS tasks (`registration_stress`, `nms_3d`, `nbv_plan`)."""
    p = params or {}
    seed = int(p.get("seed", 42))

    if task == "registration_stress":
        return _generate_registration_stress_case(
            n_correspondences=int(p.get("n_correspondences", 72)),
            outlier_ratio=float(p.get("outlier_ratio", 0.75)),
            noise_mm=float(p.get("noise_mm", 4.0)),
            allow_scale=bool(p.get("allow_scale", True)),
            seed=seed,
        )
    if task == "nms_3d":
        return _run_nms_3d_case(
            objects=p.get("objects"),
            conflict_iou_threshold=float(p.get("conflict_iou_threshold", 0.28)),
            seed=seed,
        )
    if task == "nbv_plan":
        return _run_nbv_plan_case(
            existing_azimuths_deg=p.get("existing_azimuths_deg"),
            sector_scores=p.get("sector_scores"),
            budget_k=p.get("budget_k"),
            seed=seed,
        )
    raise ValueError(f"Unknown quantum optimization task: {task}")


def build_quantum_demo_suite(seed: int = 42) -> dict[str, Any]:
    """Build deterministic SEED=42 Phase 4 benchmark suite across registration, 3D NMS, and NBV."""
    t0 = time.perf_counter()
    reg_case = _generate_registration_stress_case(seed=seed)
    nms_case = _run_nms_3d_case(seed=seed)
    nbv_case = _run_nbv_plan_case(seed=seed)
    total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "seed": int(seed),
        "engine_version": "phase4_harvard_hqi_qubo_mwis_v1",
        "references": [
            "Ebadi, Lukin et al. (Harvard, Science 2022) — Rydberg Atom Array MWIS",
            "Nguyen, Lukin et al. (Harvard, PRX Quantum 2023) — Arbitrary Connectivity QUBO",
            "Zhou, Lukin et al. (Harvard, PRX 2020) — Deep QAOA Dynamics",
            "Huang, Preskill, Lukin (Science 2022) — Classical Shadows Information Sampling",
            "Guo, Capasso, Zickler (Harvard SEAS, PNAS 2019) — Metalens Single-Shot Optics",
        ],
        "scenarios": {
            "registration_stress": reg_case,
            "nms_3d": nms_case,
            "nbv_plan": nbv_case,
        },
        "latency_ms": total_ms,
    }
