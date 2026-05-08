from collections import deque
from typing import Optional
import numpy as np
import gymnasium as gym

import pygame

#
# Coverage Path Planning (CPP) environment based on GridWorld with obstacles.
#
# The agent must visit as many free cells as possible while avoiding obstacles.
# The reward function is designed to encourage exploration of new cells and
# discourage revisiting already-visited cells.
#
# Reward function:
#   - Scale-normalized reward for visiting a new cell.
#   - Mild penalties for revisits, illegal/stuck moves, and time.
#   - Frontier-distance shaping for purposeful routing across visited cells.
#   - A strong fixed bonus for full coverage.
#   - A truncation penalty with smaller partial-coverage credit, so "almost
#     covered" does not look too similar to a completed episode.
#   VecNormalize handles reward normalization during training, making the same
#   reward function usable across different grid sizes.
#
# The observation space includes:
#   - Agent's (x, y) location (normalized)
#   - Coverage ratio (proportion of free cells visited)
#   - A 3-channel 3x3 matrix of neighboring cells centered on the agent:
#       channel 0 = obstacle or wall (including out-of-bounds)
#       channel 1 = visited free cell
#       channel 2 = unvisited free cell
#
# The episode ends when all free cells are visited or max steps is reached.
#

class GridWorldCPPEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size: int = 5, obs_quantity: int = 3, max_steps: int = 200):
        self.size = size
        self.window_size = 512
        self.obs_quantity = obs_quantity
        self.obstacles_locations = []
        self.count_steps = 0
        self.max_steps = max_steps
        self.steps_since_new = 0

        # Track visited cells
        self.visited = set()

        self._agent_location = np.array([-1, -1], dtype=int)
        # Local map around the agent, channel-first for PyTorch Conv2d:
        # obstacle/wall, visited free cell, unvisited free cell.
        self._neighbors = np.zeros((3, 3, 3), dtype=np.float32)

        # Observation: Dict with agent info (x, y, coverage) and 3-channel neighbors.
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(
                low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            ),
            "neighbors": gym.spaces.Box(
                # Binary local occupancy tensor. Values are one-hot per nearby
                # cell, so low/high are 0/1 instead of encoded class ids.
                low=np.zeros((3, 3, 3), dtype=np.float32),
                high=np.ones((3, 3, 3), dtype=np.float32),
                dtype=np.float32
            ),
        })

        # 4 actions: right, up, left, down
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, -1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, 1]),   # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    @property
    def total_free_cells(self):
        return self.size * self.size - len(self.obstacles_locations)

    @property
    def coverage_ratio(self):
        return len(self.visited) / self.total_free_cells if self.total_free_cells > 0 else 1.0

    def _get_obs(self):
        return {
            "agent": np.array([
                self._agent_location[0] / self.size,
                self._agent_location[1] / self.size,
                self.coverage_ratio,
            ], dtype=np.float32),
            "neighbors": self._neighbors.astype(np.float32),
        }

    def _get_info(self):
        return {
            "coverage": self.coverage_ratio,
            "visited_cells": len(self.visited),
            "total_free_cells": self.total_free_cells,
            "steps": self.count_steps,
            "size": self.size,
        }

    def _compute_neighbors(self):
        # Build a 3-channel 3x3 local view centered on the agent. Because only
        # relative cells are encoded, this observation is independent of grid size.
        obs = np.zeros((3, 3, 3), dtype=np.float32)
        obstacle_positions = {tuple(loc) for loc in self.obstacles_locations}

        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r = self._agent_location[0] + dr
                c = self._agent_location[1] + dc
                # Convert offsets (-1, 0, 1) to tensor indices (0, 1, 2).
                nr, nc = dr + 1, dc + 1

                if not (0 <= r < self.size and 0 <= c < self.size):
                    obs[0, nr, nc] = 1.0
                elif (r, c) in obstacle_positions:
                    obs[0, nr, nc] = 1.0
                elif (r, c) in self.visited:
                    obs[1, nr, nc] = 1.0
                else:
                    obs[2, nr, nc] = 1.0

        return obs

    def set_neighbors(self, obstacles_locations=None):
        # Kept as a small wrapper so older call sites can update neighbors
        # without knowing the new 3-channel representation.
        self._neighbors = self._compute_neighbors()

    def _is_solvable(self, agent_location: np.ndarray, obstacles_locations: list) -> bool:
        """
        BFS flood-fill from the agent's starting cell.
        Returns True if every free cell (non-obstacle, in-bounds) is reachable
        from the agent's starting position — i.e. the episode is completable.
        """
        obstacle_set = {tuple(loc) for loc in obstacles_locations}
        start = tuple(agent_location)

        total_free = self.size * self.size - len(obstacle_set)

        visited = {start}
        queue = deque([start])

        while queue:
            cx, cy = queue.popleft()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < self.size and 0 <= ny < self.size
                        and (nx, ny) not in obstacle_set
                        and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return len(visited) == total_free

    def _distance_to_nearest_frontier(self, location) -> Optional[int]:
        """Shortest free-space distance to a visited cell bordering unexplored space."""
        obstacle_set = {tuple(loc) for loc in self.obstacles_locations}
        start = tuple(location)

        if start in obstacle_set:
            return None

        def is_unvisited_free(pos):
            return (
                0 <= pos[0] < self.size
                and 0 <= pos[1] < self.size
                and pos not in obstacle_set
                and pos not in self.visited
            )

        def is_frontier(pos):
            if pos not in self.visited:
                return False
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if is_unvisited_free((pos[0] + dx, pos[1] + dy)):
                    return True
            return False

        seen = {start}
        queue = deque([(start, 0)])

        while queue:
            current, dist = queue.popleft()
            if is_frontier(current):
                return dist

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nxt = (current[0] + dx, current[1] + dy)
                if (
                    0 <= nxt[0] < self.size
                    and 0 <= nxt[1] < self.size
                    and nxt not in obstacle_set
                    and nxt not in seen
                ):
                    seen.add(nxt)
                    queue.append((nxt, dist + 1))

        return None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.count_steps = 0
        self.steps_since_new = 0
        self.obstacles_locations = []
        self.visited = set()

        # Place agent and obstacles, retrying until the layout is solvable.
        # A layout is solvable iff every free cell is reachable from the
        # agent's starting position (checked via BFS flood-fill).
        while True:
            self.obstacles_locations = []

            # Place agent randomly
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

            # Place obstacles (never on the agent's starting cell)
            for _ in range(self.obs_quantity):
                obstacle_location = self._agent_location.copy()
                while (np.array_equal(obstacle_location, self._agent_location) or
                       any(np.array_equal(obstacle_location, loc) for loc in self.obstacles_locations)):
                    obstacle_location = self.np_random.integers(0, self.size, size=2, dtype=int)
                self.obstacles_locations.append(obstacle_location)

            if self._is_solvable(self._agent_location, self.obstacles_locations):
                break  # Valid layout found — proceed with this configuration

        # Mark starting position as visited.
        self.visited.add(tuple(self._agent_location))

        self.set_neighbors()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        old_location = self._agent_location.copy()
        prev_frontier_dist = self._distance_to_nearest_frontier(old_location)

        # Move agent (clip to grid bounds)
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # If the agent hits an obstacle, stay in place
        if any(np.array_equal(self._agent_location, loc) for loc in self.obstacles_locations):
            self._agent_location = old_location

        self.count_steps += 1

        current_pos     = tuple(self._agent_location)
        is_new_cell     = current_pos not in self.visited
        stayed_in_place = np.array_equal(self._agent_location, old_location)
        current_frontier_dist = self._distance_to_nearest_frontier(current_pos)

        # Snapshot coverage BEFORE this step's cell is added so milestone
        # detection can tell exactly when a threshold is crossed.
        n_free        = self.total_free_cells
        prev_coverage = len(self.visited) / n_free

        # ── Reward constants (all scale-invariant) ────────────────────────────
        #
        # NEW_CELL_REWARD   — sum of all discoveries = +1.0 for any grid size.
        # ILLEGAL_PENALTY   — wall / obstacle bump: 2× a new-cell reward.
        # REVISIT_PENALTY   — kept very mild so corridors stay usable.
        # TIME_PENALTY      — fully consuming max_steps costs −0.25 total.
        #
        # STAGNATION_STEPS  — raised from max(5, 8% of free) to
        #                      max(15, 25% of free).  Late-game CPP requires
        #                      long traversals through visited corridors to
        #                      reach isolated cells; 5 steps was far too
        #                      aggressive and was actively punishing necessary
        #                      navigation.
        # STAGNATION_PENALTY— halved from 2.0/n to 0.5/n for the same reason.
        #
        # MILESTONE_THRESHOLDS / MILESTONE_BONUS — explicit gradient signal at
        #                      the 80 → 90 → 95 → 98% plateau zones.  Between
        #                      those thresholds the only signal is tiny
        #                      NEW_CELL_REWARD increments which, discounted
        #                      over 300+ steps at γ=0.995, are nearly zero to
        #                      the optimizer.  The milestones fire exactly once
        #                      per episode per threshold.
        #
        # COMPLETION_BONUS  — unchanged; must remain clearly above any partial
        #                      milestone accumulation so 100% is always the
        #                      dominant terminal signal.
        # TRUNCATION_PENALTY— unchanged.
        # ─────────────────────────────────────────────────────────────────────

        NEW_CELL_REWARD      =  1.0  / n_free
        ILLEGAL_PENALTY      =  2.0  / n_free
        REVISIT_PENALTY      =  0.1  / n_free
        TIME_PENALTY         =  0.25 / self.max_steps
        STAGNATION_STEPS     =  max(15, int(0.25 * n_free))
        STAGNATION_PENALTY   =  0.5  / n_free
        ROUTING_PROGRESS_REWARD = 0.5 / n_free
        ROUTING_BACKTRACK_PENALTY = 0.05 / n_free
        MILESTONE_THRESHOLDS =  [0.80, 0.90, 0.95, 0.98]
        MILESTONE_BONUS      =  0.2
        COMPLETION_BONUS     =  6.0
        TRUNCATION_PENALTY   =  0.7
        OPTIMAL_STEPS = self.size ** 2
        RATIO = self.count_steps / OPTIMAL_STEPS

        # ── Base step reward ──────────────────────────────────────────────────
        if stayed_in_place:
            reward = -ILLEGAL_PENALTY
            self.steps_since_new += 1
        elif is_new_cell:
            reward = NEW_CELL_REWARD
            self.visited.add(current_pos)
            self.steps_since_new = 0
        else:
            reward = -REVISIT_PENALTY
            self.steps_since_new += 1

        # ── Terminal check (must happen after visited is updated) ─────────────
        full_coverage = len(self.visited) >= n_free
        terminated    = full_coverage

        # ── Per-step time cost ────────────────────────────────────────────────
        reward -= TIME_PENALTY

        # Reward purposeful travel through already-visited cells when it moves
        # the agent closer to the nearest frontier of unexplored space.
        if (
            not is_new_cell
            and not terminated
            and prev_frontier_dist is not None
            and current_frontier_dist is not None
        ):
            frontier_delta = prev_frontier_dist - current_frontier_dist
            if frontier_delta > 0:
                reward += ROUTING_PROGRESS_REWARD * frontier_delta
            elif frontier_delta < 0:
                reward += ROUTING_BACKTRACK_PENALTY * frontier_delta

        # ── Stagnation penalty ────────────────────────────────────────────────
        # Only fires when the agent has genuinely stopped making progress for
        # an extended period, not during routine corridor traversal.
        if self.steps_since_new >= STAGNATION_STEPS and not terminated:
            reward -= STAGNATION_PENALTY

        # ── Inefficient Use of steps penalty ──────────────────────────────────

        if RATIO > 1.2:
            inefficiency_penalty = (RATIO - 1.2) * (0.05 / n_free)
            reward -= inefficiency_penalty 

        # ── Milestone bonuses ─────────────────────────────────────────────────
        # Fire exactly once per episode per threshold, at the step that crosses
        # it. prev_coverage was captured before self.visited was updated so
        # the boundary detection is accurate.
        if is_new_cell and not terminated:
            current_coverage = len(self.visited) / n_free
            for threshold in MILESTONE_THRESHOLDS:
                if prev_coverage < threshold <= current_coverage:
                    reward += MILESTONE_BONUS
                    break   # at most one milestone fires per step

        # ── Completion bonus ──────────────────────────────────────────────────
        if terminated:
            # Reward efficiency
            efficiency_bonus = max(0.0, 1.3 - RATIO)

            # Penalize inefficiency
            efficiency_penalty = max(0.0, RATIO - 1.3)

            reward += COMPLETION_BONUS
            reward += 2.0 * efficiency_bonus
            reward -= 3.0 * efficiency_penalty

        # ── Truncation ────────────────────────────────────────────────────────
        if self.count_steps >= self.max_steps and not terminated:
            truncated  = True
            reward    -= TRUNCATION_PENALTY
            reward    += 2.0 * (self.coverage_ratio ** 2) # partial credit
        else:
            truncated = False

        # ── Update local view and build observation ───────────────────────────
        self.set_neighbors()

        observation = self._get_obs()
        info        = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        # Draw visited cells in light green
        for cell in self.visited:
            cell_arr = np.array(cell)
            pygame.draw.rect(
                canvas,
                (144, 238, 144),  # light green
                pygame.Rect(
                    pix_square_size * cell_arr,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw obstacles in black
        for obs in self.obstacles_locations:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * obs,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw agent as blue circle
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw coverage info text
        font = pygame.font.SysFont(None, 24)
        coverage_text = font.render(
            f"Coverage: {self.coverage_ratio:.1%} | Steps: {self.count_steps}",
            True, (0, 0, 0)
        )
        canvas.blit(coverage_text, (5, 5))

        # Draw gridlines
        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, pix_square_size * x),
                             (self.window_size, pix_square_size * x), width=3)
            pygame.draw.line(canvas, 0, (pix_square_size * x, 0),
                             (pix_square_size * x, self.window_size), width=3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
