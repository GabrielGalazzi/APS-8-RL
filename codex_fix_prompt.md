# Codex Task: CPP Project — Fix All Listed Issues

You are working on a Coverage Path Planning (CPP) reinforcement learning project
using Stable-Baselines3 RecurrentPPO. The codebase consists of the following files:

- `grid_world_cpp.py` — Gymnasium environment
- `train_grid_world_cpp.py` — Training script and all hyperparameters
- `continue_latest_stage_training.py` — Resume/continuation script
- `goal_conditioned_policy.py` — Custom actor-critic MLP extractor and LSTM policy
- `CNN.py` — Custom CNN feature extractor
- `cpp_env_factory.py` — Environment factory helpers
- `cpp_subproc_vec_env.py` — Custom subprocess vectorized environment
- `cpp_subproc_worker.py` — Subprocess worker

Apply every fix described below. Do not change anything not listed. Do not
alter hyperparameters, the curriculum config, ADAM_BETAS, weight_decay,
DETERMINISTIC_PROMOTION_FLOOR_FRACTION, or evaluation frequency logic.

---

## Fix 1 — GoalConditionedMlpExtractor: critic path contradicts its own docstring

**File:** `goal_conditioned_policy.py`

**Problem:** The class docstring states "the critic receives the unchanged features."
However, `forward_critic` currently runs the `goal_head`, concatenates the goal
vector, and passes the result into `value_net`. `last_layer_dim_vf` is also
initialized to `feature_dim + goal_dim`, which only makes sense if the concatenation
is intentional. The docstring and the implementation disagree.

**Fix:** Make the implementation match the docstring. The critic should receive only
the raw features, unchanged. Specifically:

1. In `__init__`, change `last_layer_dim_vf` to be initialized from `feature_dim`
   alone (not `feature_dim + goal_dim`). The value network layers should build from
   `feature_dim` as their starting width.

2. In `forward_critic`, remove the `goal_head` call and concatenation entirely.
   Pass `features` directly into `self.value_net` without modification.

3. The `forward` method and `forward_actor` are correct and should not change.

---

## Fix 2 — Remove dead constant

**File:** `train_grid_world_cpp.py`

**Problem:** The following commented-out line exists near the top of the
hyperparameters section:

```python
# FINAL_ENTROPY_COEF = 0.0005 NOT USED
```

**Fix:** Delete this line entirely.

---

## Fix 3 — Remove dead parameter from `set_neighbors`

**File:** `grid_world_cpp.py`

**Problem:** `set_neighbors` accepts an `obstacles_locations=None` parameter that
is never read inside the method body. The method only calls
`self._compute_neighbors()`, which reads from `self.obstacles_locations` directly.

**Fix:** Remove the `obstacles_locations` parameter from the method signature.
Update all call sites in the same file to match (they already call it without
arguments, so only the signature needs to change).

---

## Fix 4 — Obstacle placement uses slow linear scan in `reset()`

**File:** `grid_world_cpp.py`

**Problem:** Inside the `while True` loop in `reset()`, existing obstacle locations
are checked using `any(np.array_equal(...) for loc in self.obstacles_locations)`,
which is an O(n) scan per placement attempt.

**Fix:** Before the inner obstacle-placement loop, build a set of occupied positions:

```python
occupied = {tuple(self._agent_location)}
```

After each successfully placed obstacle, add its tuple to `occupied`. Replace all
`np.array_equal`-based collision checks with `tuple(obstacle_location) in occupied`.
Do not change the BFS solvability check that follows — only the placement loop.

---

## Fix 5 — Two BFS passes per `step()` call

**File:** `grid_world_cpp.py`

**Problem:** `step()` calls `_distance_to_nearest_frontier` twice per step:
once before the move (to get `prev_frontier_dist`) and once after (to get
`current_frontier_dist`). Each call is a full BFS over the grid. This doubles
the BFS cost on every environment step.

**Fix:** Restructure `step()` so that:

1. `prev_frontier_dist` is computed once before the move, as it currently is.
2. After the move and after `self.visited` is updated (if `is_new_cell`), compute
   `current_frontier_dist` only when it will actually be used — i.e., only when
   `not is_new_cell and not terminated`. In that case compute it once and use it
   for both the routing reward and backtrack penalty.
3. If `is_new_cell` or `terminated` is True, skip the second BFS entirely by
   setting `current_frontier_dist = None` (routing shaping is already skipped
   in those branches).

The routing reward/penalty block already guards on `prev_frontier_dist is not None
and current_frontier_dist is not None`, so setting `current_frontier_dist = None`
in the skipped branches is safe and requires no other changes to that block.

---

## Fix 6 — Extract shared training loop to eliminate duplication

**Files:** `train_grid_world_cpp.py`, `continue_latest_stage_training.py`

**Problem:** The core curriculum training loop — the `for stage_idx, stage in
enumerate(CURRICULUM)` block including warmup, the `while True` training chunk,
the entropy/LR schedules, the eval gate, consecutive-hit logic, and stage
checkpointing — is substantially duplicated between `train_grid_world_cpp.py`
(inside `main()` under `mode == "train"`) and `continue_latest_stage_training.py`.
Any change to the loop must currently be applied in two places.

**Fix:** Extract the shared loop into a standalone function in
`train_grid_world_cpp.py` with a signature along the lines of:

```python
def run_curriculum(
    model,
    env_holder,
    run_name: str,
    log_dir: str,
    checkpoint_dir: Path,
    cbp,
    train_callback,
    start_stage_idx: int = 0,
    start_stage_timesteps: int = 0,
) -> None:
```

The function should contain the full `for stage_idx, stage in enumerate(CURRICULUM)`
loop body exactly as it currently exists in `main()`, parameterised on `start_stage_idx`
and `start_stage_timesteps` so the resume script can pass non-zero values.

In `main()` under `mode == "train"`, replace the loop body with a call to
`run_curriculum(model, env_holder, run_name, log_dir, checkpoint_dir, cbp,
train_callback)`.

In `continue_latest_stage_training.py`, import `run_curriculum` from
`train_grid_world_cpp` and replace the duplicated loop body with a call to it,
passing the resume point's `stage_idx` and `stage_timesteps`.

Do not change the logic of the loop itself — only move and parameterise it.

---

## Fix 7 — OPTIMAL_STEPS baseline is wrong

**File:** `grid_world_cpp.py`

**Problem:** Inside `step()`, the efficiency baseline is computed as:

```python
OPTIMAL_STEPS = self.size ** 2
```

`size ** 2` includes obstacle cells, which the agent can never visit. The
correct minimum step count for full coverage is `total_free_cells - 1`
(a Hamiltonian path over free cells). Using `size ** 2` makes efficiency
ratios systematically optimistic and misaligns the efficiency bonus and
penalty thresholds.

**Fix:** Replace:

```python
OPTIMAL_STEPS = self.size ** 2
```

with:

```python
OPTIMAL_STEPS = max(self.total_free_cells - 1, 1)
```

This is the only change required. `RATIO` and all downstream uses of it
are correct as-is once the baseline is fixed.

---

## Fix 8 — ILLEGAL_PENALTY is too large and discourages perimeter exploration

**File:** `grid_world_cpp.py`

**Problem:** `ILLEGAL_PENALTY = 2.0 / n_free` is twice `NEW_CELL_REWARD`. In
CPP, boundary and corner cells can only be reached by moving adjacent to walls,
meaning legitimate exploration near perimeter cells frequently triggers this
penalty. The double-penalty teaches the agent that walls are dangerous, which
directly suppresses coverage of boundary cells.

**Fix:** Reduce the illegal/bump penalty to match `NEW_CELL_REWARD`:

```python
ILLEGAL_PENALTY = 1.0 / n_free
```

Do not change any other constant or penalty.

---

## Fix 9 — STAGNATION_PENALTY accumulates indefinitely

**File:** `grid_world_cpp.py`

**Problem:** The stagnation penalty fires every step once `steps_since_new >=
STAGNATION_STEPS`. It never stops accumulating. In late-game CPP on large grids,
the agent legitimately needs to traverse long corridors of already-visited cells
to reach isolated unvisited pockets. The unlimited accumulation punishes this
necessary navigation.

**Fix — two changes:**

1. Raise the threshold from `max(15, int(0.25 * n_free))` to
   `max(20, int(0.40 * n_free))`. This gives more room for late-game corridor
   traversal before the penalty fires.

2. Cap the total stagnation debt per episode. Add a new constant:
   ```python
   STAGNATION_CAP = 3.0 / n_free
   ```
   Add an instance variable `self.stagnation_debt = 0.0` initialized in both
   `__init__` and `reset()`. In the stagnation block:
   ```python
   if self.steps_since_new >= STAGNATION_STEPS and not terminated:
       if self.stagnation_debt < STAGNATION_CAP:
           reward -= STAGNATION_PENALTY
           self.stagnation_debt += STAGNATION_PENALTY
   ```
   Reset `self.stagnation_debt = 0.0` at the top of `reset()`.

---

## Fix 10 — Truncation partial credit formula is too weak at low coverage

**File:** `grid_world_cpp.py`

**Problem:** The truncation partial credit is:

```python
reward += 2.0 * (self.coverage_ratio ** 2)
```

Squaring the coverage ratio makes the gradient near 0–70% coverage nearly zero,
meaning truncated episodes during early training provide almost no useful terminal
signal. The agent relies entirely on shaping rewards to learn, while the truncation
terminal signal sits dormant for the first half of training.

**Fix:** Replace the exponent `2` with `1.5`:

```python
reward += 2.0 * (self.coverage_ratio ** 1.5)
```

This preserves the progressive structure (higher coverage = more credit) and still
rewards near-complete episodes significantly more than partial ones, but provides
a meaningful gradient signal even at 40–70% coverage.

---

## Fix 11 — Oscillation between two adjacent cells is weakly rewarded

**File:** `grid_world_cpp.py`

**Problem:** The routing reward/backtrack structure doesn't distinguish purposeful
corridor traversal from oscillation (agent bouncing A → B → A → B). When oscillating
near a frontier, the agent alternately earns `ROUTING_PROGRESS_REWARD` and
`ROUTING_BACKTRACK_PENALTY`, but since the progress reward coefficient is 10× the
backtrack penalty, net oscillation is weakly rewarded.

**Fix:**

1. Add an instance variable `self._prev_location = None` initialized in `__init__`
   and reset to `None` in `reset()`.

2. At the top of `step()`, after recording `old_location`, save it:
   ```python
   prev_prev_location = self._prev_location
   self._prev_location = tuple(old_location)
   ```

3. After computing `current_pos`, add:
   ```python
   OSCILLATION_PENALTY = 0.3 / n_free
   is_oscillating = (
       prev_prev_location is not None
       and not is_new_cell
       and current_pos == prev_prev_location
   )
   ```

4. In the reward computation, after the revisit penalty block, add:
   ```python
   if is_oscillating:
       reward -= OSCILLATION_PENALTY
   ```

The penalty only fires when the agent returns to the cell it was at two steps ago
without having discovered any new cell. This does not affect agents traversing
corridors directionally, only true back-and-forth oscillation.

---

## Summary of files modified

| File | Fixes applied |
|---|---|
| `goal_conditioned_policy.py` | Fix 1 |
| `train_grid_world_cpp.py` | Fix 2, Fix 6 |
| `continue_latest_stage_training.py` | Fix 6 |
| `grid_world_cpp.py` | Fix 3, Fix 4, Fix 5, Fix 7, Fix 8, Fix 9, Fix 10, Fix 11 |

`CNN.py`, `cpp_env_factory.py`, `cpp_subproc_vec_env.py`, and
`cpp_subproc_worker.py` are not modified.
