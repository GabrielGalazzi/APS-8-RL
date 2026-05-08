# python train_grid_world_cpp.py <train|test|run|random>
#
# Coverage Path Planning (CPP) training script.

import sys
import atexit
import gc
import json
import multiprocessing as mp
import os
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from cpp_env_factory import make_cpp_env, make_cpp_env_factory, register_cpp_env


def print_action(action: int) -> str:
    return {0: "right", 1: "up", 2: "left", 3: "down"}.get(action, "unknown")


def resolve_model_paths(model_name: str):
    primary_model_path = f"data/{model_name}"
    best_model_path = Path("log") / model_name / "best_model" / "best_model"
    if best_model_path.with_suffix(".zip").exists():
        return primary_model_path, str(best_model_path)
    return primary_model_path, None


def close_managed_env(env, label: str) -> None:
    """Close an env stack and promptly release subprocess handles if any exist."""
    if env is None:
        return

    try:
        env.close()
    except Exception as exc:
        print(f"  Warning: failed to close {label}: {exc}")
    finally:
        # active_children() also reaps any completed multiprocessing children
        # still tracked by this process.
        mp.active_children()
        gc.collect()


def save_emergency_checkpoint(model, env, run_name: str, progress: dict, reason) -> None:
    if model is None:
        print("  Emergency checkpoint skipped: model was not created yet.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_number = progress.get("stage_idx", -1) + 1
    global_timesteps = getattr(model, "num_timesteps", 0)
    stem = f"{run_name}_interrupt_stage{stage_number}_{global_timesteps}ts_{timestamp}"

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    model_path = data_dir / f"{stem}.zip"
    latest_model_path = data_dir / f"{run_name}_interrupt_latest.zip"
    vecnorm_path = data_dir / f"{stem}_vecnorm.pkl"
    latest_vecnorm_path = data_dir / f"{run_name}_interrupt_latest_vecnorm.pkl"
    metadata_path = data_dir / f"{stem}_metadata.json"
    latest_metadata_path = data_dir / f"{run_name}_interrupt_latest_metadata.json"

    reason_name = reason.__class__.__name__ if reason is not None else "manual"
    metadata = {
        **progress,
        "reason": reason_name,
        "reason_message": str(reason) if reason is not None else "",
        "saved_at": timestamp,
        "run_name": run_name,
        "model_path": str(model_path),
        "latest_model_path": str(latest_model_path),
        "vecnorm_path": str(vecnorm_path),
        "latest_vecnorm_path": str(latest_vecnorm_path),
        "global_timesteps": global_timesteps,
        "traceback": "" if isinstance(reason, KeyboardInterrupt) else traceback.format_exc(),
    }

    print("\n  Emergency checkpoint requested.")
    print(f"  Reason: {reason_name}")

    model.save(str(model_path))
    model.save(str(latest_model_path))
    print(f"  Saved model   -> {model_path}")
    print(f"  Latest model  -> {latest_model_path}")

    if env is not None and hasattr(env, "save"):
        env.save(str(vecnorm_path))
        env.save(str(latest_vecnorm_path))
        print(f"  Saved vecnorm -> {vecnorm_path}")

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    latest_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  Saved state   -> {metadata_path}")


def get_vec_env_backend(purpose: str) -> str:
    env_var = f"CPP_{purpose.upper()}_VEC_ENV"
    default_backend = "subproc"
    backend = os.getenv(env_var, default_backend).strip().lower()
    if backend not in {"dummy", "subproc"}:
        raise ValueError(f"{env_var} must be 'dummy' or 'subproc', got {backend!r}")
    return backend


def make_parallel_cpp_env(
    size: int,
    obs_quantity: int,
    max_steps: int,
    n_envs: int | None = None,
    purpose: str = "train",
):
    n_envs = n_envs or N_WORKERS
    env_fns = [
        make_cpp_env_factory(size, obs_quantity, max_steps)
        for _ in range(n_envs)
    ]

    backend = get_vec_env_backend(purpose)
    if backend == "dummy":
        from stable_baselines3.common.vec_env import DummyVecEnv

        return DummyVecEnv(env_fns)

    start_method = "spawn" if sys.platform == "win32" else "fork"
    from cpp_subproc_vec_env import CppSubprocVecEnv

    return CppSubprocVecEnv(
        env_fns,
        start_method=start_method,
    )

# Hyperparameters
INITAL_ENTROPY_COEF = 0.015
# FINAL_ENTROPY_COEF = 0.0005 NOT USED
LEARNING_RATE = 7e-5
ADAPTATION_LEARNING_RATE = 3e-5
ADAM_BETAS = (0.99, 0.99)
GAMMA = 0.995
N_STEPS = 512  # Effective rollout = N_STEPS * N_WORKERS = 512 * 10 = 5120
BATCH_SIZE = 128
N_EPOCHS = 8
GAE_LAMBDA = 0.97
CLIP_RANGE = 0.08   
TARGET_KL = 0.02
CBP_ENABLED = os.getenv("CPP_CBP_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
CBP_REPLACEMENT_RATE = float(os.getenv("CPP_CBP_REPLACEMENT_RATE", "1e-4"))
CBP_MATURITY_THRESHOLD = int(os.getenv("CPP_CBP_MATURITY_THRESHOLD", "10000"))
CBP_UTILITY_DECAY = float(os.getenv("CPP_CBP_UTILITY_DECAY", "0.99"))

# Larger MLP to process the richer dict observation.
POLICY_KWARGS = dict(net_arch=[256, 256, 256])

# Curriculum config
# Each stage trains until `threshold` full-coverage rate is reached for
# `CONSECUTIVE_REQUIRED` consecutive back-to-back evaluations after the
# `min_timesteps` floor is reached. Failed validation returns to training for
# another EVAL_FREQ_STEPS interval. The final stage has threshold=None and
# trains to budget only.
CONSECUTIVE_REQUIRED = 3
EVAL_FREQ_STEPS = 25_000
EVAL_EPISODES = 100
EVAL_DETERMINISTIC = False
EVAL_MODE_GAP_MARGIN = 0.03
DETERMINISTIC_PROMOTION_FLOOR_FRACTION = 0.00
PROMOTION_MIN_MEAN_REWARD = 6.0
PROMOTION_MAX_MEAN_LENGTH_SIZE_FACTOR = 1.8

CURRICULUM = [
    dict(size=5,  obs=3,  max_steps=200, threshold=0.95, min_timesteps=100_000,
         ent_start=0.015, ent_end=0.001, ent_decay_timesteps=100_000, ent_failure_bump=0.0,
         lr_start=LEARNING_RATE, lr_end=LEARNING_RATE, lr_decay_timesteps=100_000),

    dict(size=7,  obs=6,  max_steps=280, threshold=0.95, min_timesteps=100_000,
         ent_start=0.012, ent_end=0.004, ent_decay_timesteps=200_000, ent_failure_bump=0.0001,
         lr_start=ADAPTATION_LEARNING_RATE, lr_end=LEARNING_RATE, lr_decay_timesteps=150_000),

    dict(size=9,  obs=10, max_steps=360, threshold=0.94, min_timesteps=120_000,
         ent_start=0.011, ent_end=0.004, ent_decay_timesteps=250_000, ent_failure_bump=0.0001,
         lr_start=ADAPTATION_LEARNING_RATE, lr_end=LEARNING_RATE, lr_decay_timesteps=175_000),

    dict(size=11, obs=15, max_steps=440, threshold=0.93, min_timesteps=150_000,
         ent_start=0.011, ent_end=0.003, ent_decay_timesteps=300_000, ent_failure_bump=0.0002,
         lr_start=ADAPTATION_LEARNING_RATE, lr_end=LEARNING_RATE, lr_decay_timesteps=225_000),

    dict(size=13, obs=20, max_steps=520, threshold=0.92, min_timesteps=200_000,
         ent_start=0.010, ent_end=0.003, ent_decay_timesteps=350_000, ent_failure_bump=0.0002,
         lr_start=ADAPTATION_LEARNING_RATE, lr_end=LEARNING_RATE, lr_decay_timesteps=275_000),

    dict(size=15, obs=27, max_steps=600, threshold=0.91, min_timesteps=250_000,
         ent_start=0.010, ent_end=0.002, ent_decay_timesteps=400_000, ent_failure_bump=0.0002,
         lr_start=ADAPTATION_LEARNING_RATE, lr_end=LEARNING_RATE, lr_decay_timesteps=350_000),

    dict(size=17, obs=35, max_steps=680, threshold=0.90, min_timesteps=350_000,
         ent_start=0.010, ent_end=0.002, ent_decay_timesteps=500_000, ent_failure_bump=0.0002,
         lr_start=ADAPTATION_LEARNING_RATE, lr_end=LEARNING_RATE, lr_decay_timesteps=450_000),

    dict(size=19, obs=43, max_steps=760, threshold=0.90, min_timesteps=500_000,
         ent_start=0.008, ent_end=0.001, ent_decay_timesteps=600_000, ent_failure_bump=0.0001,
         lr_start=ADAPTATION_LEARNING_RATE, lr_end=LEARNING_RATE, lr_decay_timesteps=550_000),

    dict(size=20, obs=48, max_steps=800, threshold=0.90, min_timesteps=1_600_000,
         ent_start=0.0075, ent_end=0.0005, ent_decay_timesteps=1_600_000, ent_failure_bump=0.00003,
         lr_start=ADAPTATION_LEARNING_RATE, lr_end=LEARNING_RATE, lr_decay_timesteps=600_000),
]

N_WORKERS = int(os.getenv("CPP_N_WORKERS", "10"))
EVAL_WORKERS = int(os.getenv("CPP_EVAL_WORKERS", str(N_WORKERS)))
DEVICE = os.getenv("CPP_DEVICE", "cuda")
TOTAL_MIN_TIMESTEPS = sum(s["min_timesteps"] for s in CURRICULUM)

# Test config (used in `test`, `run`, and `random` modes).
TEST_DIM = 20
TEST_OBSTACLES = 48
TEST_MAX_STEPS = 1000
TEST_DETERMINISTIC = False


def linear_stage_value(stage: dict, stage_timesteps: int, start_key: str,
                       end_key: str, decay_key: str) -> float:
    decay_timesteps = max(int(stage.get(decay_key, stage["min_timesteps"])), 1)
    frac = min(stage_timesteps / decay_timesteps, 1.0)
    return stage[start_key] + frac * (stage[end_key] - stage[start_key])


def constant_schedule(value: float):
    def schedule(_progress_remaining: float) -> float:
        return value

    return schedule


def set_model_learning_rate(model, learning_rate: float) -> None:
    model.learning_rate = learning_rate
    model.lr_schedule = constant_schedule(learning_rate)
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = learning_rate


def record_plasticity_metrics(model) -> None:
    """Log lightweight correlates of plasticity loss without touching env state."""
    import torch as th
    import torch.nn as nn

    weight_abs_sum = 0.0
    weight_square_sum = 0.0
    weight_max_abs = 0.0
    weight_count = 0

    low_norm_units = 0
    total_units = 0
    effective_rank_sum = 0.0
    normalized_effective_rank_sum = 0.0
    ranked_layers = 0

    with th.no_grad():
        for parameter in model.policy.parameters():
            if not parameter.requires_grad or parameter.ndim < 2:
                continue

            weight = parameter.detach().float()
            weight_abs_sum += float(weight.abs().sum().item())
            weight_square_sum += float(weight.square().sum().item())
            weight_max_abs = max(weight_max_abs, float(weight.abs().max().item()))
            weight_count += weight.numel()

        for module in model.policy.modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue

            weight = module.weight.detach().float()
            matrix = weight.reshape(weight.shape[0], -1)
            row_norms = matrix.norm(dim=1)

            if row_norms.numel() > 0:
                threshold = 0.01 * row_norms.mean().clamp_min(1e-12)
                low_norm_units += int((row_norms <= threshold).sum().item())
                total_units += int(row_norms.numel())

            if min(matrix.shape) < 2:
                continue

            singular_values = th.linalg.svdvals(matrix)
            singular_sum = singular_values.sum()
            if singular_sum <= 0:
                effective_rank = 0
            else:
                cumulative = th.cumsum(singular_values, dim=0) / singular_sum
                effective_rank = int((cumulative >= 0.99).nonzero()[0].item() + 1)

            effective_rank_sum += effective_rank
            normalized_effective_rank_sum += effective_rank / min(matrix.shape)
            ranked_layers += 1

    if weight_count > 0:
        model.logger.record("plasticity/weight_abs_mean", weight_abs_sum / weight_count)
        model.logger.record(
            "plasticity/weight_rms",
            (weight_square_sum / weight_count) ** 0.5,
        )
        model.logger.record("plasticity/weight_max_abs", weight_max_abs)

    if total_units > 0:
        model.logger.record(
            "plasticity/low_weight_unit_ratio",
            low_norm_units / total_units,
        )
        model.logger.record("plasticity/low_weight_units", low_norm_units)
        model.logger.record("plasticity/total_weight_units", total_units)

    if ranked_layers > 0:
        model.logger.record(
            "plasticity/effective_rank_99_mean",
            effective_rank_sum / ranked_layers,
        )
        model.logger.record(
            "plasticity/effective_rank_99_normalized_mean",
            normalized_effective_rank_sum / ranked_layers,
        )


class ContinualBackpropTarget:
    def __init__(self, name, layer, outgoing_layer, replacement_rate,
                 maturity_threshold, utility_decay) -> None:
        import torch as th

        self.name = name
        self.layer = layer
        self.outgoing_layer = outgoing_layer
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.utility_decay = utility_decay
        self.activation_utility = th.zeros(layer.out_features, device=layer.weight.device)
        self.age = th.zeros(layer.out_features, device=layer.weight.device)
        self.replacement_budget = 0.0
        self.last_utility_mean = 0.0
        self.last_replacements = 0
        self.handle = layer.register_forward_hook(self._activation_hook)

    def _activation_hook(self, _module, _inputs, output) -> None:
        import torch as th

        with th.no_grad():
            activations = output.detach().float()
            if activations.ndim == 1:
                activations = activations.reshape(1, -1)
            elif activations.ndim > 2:
                activations = activations.reshape(-1, activations.shape[-1])

            if activations.shape[-1] != self.activation_utility.numel():
                return

            batch_utility = activations.abs().mean(dim=0).to(self.activation_utility.device)
            self.activation_utility.mul_(self.utility_decay).add_(
                batch_utility,
                alpha=1.0 - self.utility_decay,
            )

    def _utility(self):
        import torch as th

        incoming = self.layer.weight.detach().float().abs().mean(dim=1).clamp_min(1e-12)
        outgoing = self.outgoing_layer.weight.detach().float().abs().mean(dim=0)
        utility = self.activation_utility * outgoing.to(self.activation_utility.device)
        utility = utility / incoming.to(self.activation_utility.device)
        self.last_utility_mean = float(utility.mean().item()) if utility.numel() > 0 else 0.0
        return utility

    def _zero_optimizer_rows(self, optimizer, parameter, units) -> None:
        state = optimizer.state.get(parameter)
        if not state:
            return

        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(key)
            if value is None or value.ndim == 0:
                continue

            if value.ndim == 1:
                value[units] = 0
            else:
                value[units, ...] = 0

    def _zero_optimizer_columns(self, optimizer, parameter, units) -> None:
        state = optimizer.state.get(parameter)
        if not state:
            return

        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(key)
            if value is None or value.ndim < 2:
                continue

            value[:, units] = 0

    def recycle(self, optimizer, elapsed_timesteps: int) -> int:
        import math
        import torch as th

        if elapsed_timesteps <= 0:
            elapsed_timesteps = 1

        with th.no_grad():
            self.age.add_(elapsed_timesteps)
            mature = self.age >= self.maturity_threshold
            mature_count = int(mature.sum().item())
            if mature_count == 0:
                self.last_replacements = 0
                return 0

            # The paper applies replacement continually. Rollout callbacks are
            # coarser, so scale by vector-env rollout count while keeping the
            # total reset rate very small.
            rollout_scale = max(elapsed_timesteps / max(N_STEPS, 1), 1.0)
            self.replacement_budget += self.replacement_rate * mature_count * rollout_scale
            n_replace = int(self.replacement_budget)
            if n_replace <= 0:
                self.last_replacements = 0
                return 0

            n_replace = min(n_replace, mature_count)
            self.replacement_budget -= n_replace

            utility = self._utility().clone()
            utility[~mature] = math.inf
            units = th.topk(-utility, k=n_replace).indices

            fan_in = self.layer.weight.shape[1]
            bound = 1.0 / math.sqrt(fan_in)
            self.layer.weight[units].uniform_(-bound, bound)
            if self.layer.bias is not None:
                self.layer.bias[units].uniform_(-bound, bound)

            self.outgoing_layer.weight[:, units] = 0.0

            self.activation_utility[units] = 0.0
            self.age[units] = 0.0

            self._zero_optimizer_rows(optimizer, self.layer.weight, units)
            if self.layer.bias is not None:
                self._zero_optimizer_rows(optimizer, self.layer.bias, units)
            self._zero_optimizer_columns(optimizer, self.outgoing_layer.weight, units)

            self.last_replacements = n_replace
            return n_replace


class ContinualBackprop:
    def __init__(self, policy, replacement_rate=CBP_REPLACEMENT_RATE,
                 maturity_threshold=CBP_MATURITY_THRESHOLD,
                 utility_decay=CBP_UTILITY_DECAY) -> None:
        import torch.nn as nn

        self.targets = []
        self.total_replacements = 0

        def add_sequential_targets(prefix, sequential, final_outgoing):
            linear_layers = [
                (idx, module)
                for idx, module in enumerate(sequential)
                if isinstance(module, nn.Linear)
            ]
            for position, (idx, layer) in enumerate(linear_layers):
                if position + 1 < len(linear_layers):
                    outgoing = linear_layers[position + 1][1]
                else:
                    outgoing = final_outgoing

                self.targets.append(
                    ContinualBackpropTarget(
                        f"{prefix}.{idx}",
                        layer,
                        outgoing,
                        replacement_rate,
                        maturity_threshold,
                        utility_decay,
                    )
                )

        add_sequential_targets(
            "mlp_extractor.policy_net",
            policy.mlp_extractor.policy_net,
            policy.action_net,
        )
        add_sequential_targets(
            "mlp_extractor.value_net",
            policy.mlp_extractor.value_net,
            policy.value_net,
        )

    def step(self, model, elapsed_timesteps: int) -> int:
        replacements = 0
        optimizer = model.policy.optimizer
        for target in self.targets:
            replacements += target.recycle(optimizer, elapsed_timesteps)
        self.total_replacements += replacements
        return replacements

    def record(self, logger) -> None:
        logger.record("cbp/total_replacements", self.total_replacements)
        logger.record("cbp/target_count", len(self.targets))
        if not self.targets:
            return

        logger.record(
            "cbp/last_replacements",
            sum(target.last_replacements for target in self.targets),
        )
        logger.record(
            "cbp/utility_mean",
            sum(target.last_utility_mean for target in self.targets) / len(self.targets),
        )


class ContinualBackpropCallback:
    def __init__(self, cbp) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, continual_backprop):
                super().__init__()
                self.continual_backprop = continual_backprop
                self.last_timesteps = 0

            def _on_training_start(self) -> None:
                self.last_timesteps = self.num_timesteps

            def _on_rollout_end(self) -> None:
                elapsed = self.num_timesteps - self.last_timesteps
                replacements = self.continual_backprop.step(self.model, elapsed)
                self.last_timesteps = self.num_timesteps
                self.logger.record("cbp/rollout_replacements", replacements)
                self.continual_backprop.record(self.logger)

            def _on_step(self) -> bool:
                return True

        self.callback = _Callback(cbp)


def eval_mode_code(mode: str) -> int:
    return {"deterministic": -1, "balanced": 0, "stochastic": 1}.get(mode, 0)


def run_coverage_eval(model, size: int, obs: int, max_steps: int,
                      n_episodes: int = EVAL_EPISODES,
                      stage_timesteps: int = 0,
                      stage_idx: int = 0,
                      deterministic: bool = EVAL_DETERMINISTIC,
                      n_eval_envs: int | None = None,
                      progress_callback=None) -> dict:
    """Run parallel evaluation episodes and return coverage/reward metrics."""

    n_eval_envs = min(n_eval_envs or EVAL_WORKERS, n_episodes)
    env = make_parallel_cpp_env(
        size,
        obs,
        max_steps,
        n_envs=n_eval_envs,
        purpose="eval",
    )

    try:
        obs_dict = env.reset()

        full_coverage = 0
        rewards = []
        lengths = []

        current_rewards = np.zeros(n_eval_envs, dtype=np.float32)
        current_lengths = np.zeros(n_eval_envs, dtype=np.int32)

        completed = 0
        lstm_states = None
        episode_starts = np.ones(n_eval_envs, dtype=bool)

        while completed < n_episodes:
            actions, lstm_states = model.predict(
                obs_dict,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )

            obs_dict, step_rewards, dones, infos = env.step(actions)
            episode_starts = dones

            current_rewards += step_rewards
            current_lengths += 1

            for env_idx, done in enumerate(dones):
                if not done:
                    continue

                if completed >= n_episodes:
                    continue

                info = infos[env_idx]

                rewards.append(float(current_rewards[env_idx]))
                lengths.append(int(current_lengths[env_idx]))

                if info.get("coverage", 0.0) >= 1.0:
                    full_coverage += 1

                completed += 1
                if progress_callback is not None:
                    progress_callback(completed)

                current_rewards[env_idx] = 0.0
                current_lengths[env_idx] = 0
    finally:
        close_managed_env(env, "eval env")

    rewards = np.array(rewards)
    lengths = np.array(lengths)
    rate = full_coverage / n_episodes
    mean_reward = float(rewards.mean())
    std_reward = float(rewards.std())
    mean_length = float(lengths.mean())
    std_length = float(lengths.std())

    print(
        f"  [stage {stage_idx + 1} | {stage_timesteps:,} ts] "
        f"coverage: {rate * 100:.1f}% "
        f"({'deterministic' if deterministic else 'stochastic'})"
    )
    print(f"    Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"    Mean Steps : {mean_length:.2f} +/- {std_length:.2f}")

    model.logger.record("eval/stage", stage_idx + 1)
    model.logger.record("eval/stage_timesteps", stage_timesteps)
    model.logger.record("eval/full_coverage_rate", rate)
    model.logger.record("eval/mean_reward", mean_reward)
    model.logger.record("eval/std_reward", std_reward)
    model.logger.record("eval/mean_ep_length", mean_length)
    model.logger.record("eval/std_ep_length", std_length)
    model.logger.record("eval/deterministic", int(deterministic))
    model.logger.dump(model.num_timesteps)

    return {
        "rate": rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "std_length": std_length,
    }


def run_dual_coverage_eval(model, size: int, obs: int, max_steps: int,
                           n_episodes: int = EVAL_EPISODES,
                           stage_timesteps: int = 0,
                           stage_idx: int = 0,
                           promotion_threshold: float | None = None,
                           progress_callback=None) -> dict:
    stochastic_eval = run_coverage_eval(
        model,
        size,
        obs,
        max_steps,
        n_episodes=n_episodes,
        stage_timesteps=stage_timesteps,
        stage_idx=stage_idx,
        deterministic=False,
        progress_callback=progress_callback,
    )
    deterministic_eval = run_coverage_eval(
        model,
        size,
        obs,
        max_steps,
        n_episodes=n_episodes,
        stage_timesteps=stage_timesteps,
        stage_idx=stage_idx,
        deterministic=True,
        progress_callback=progress_callback,
    )
    stochastic_rate = stochastic_eval["rate"]
    deterministic_rate = deterministic_eval["rate"]

    deterministic_floor = None
    max_mean_length = PROMOTION_MAX_MEAN_LENGTH_SIZE_FACTOR * (size ** 2)
    stochastic_quality_passed = (
        stochastic_eval["mean_reward"] > PROMOTION_MIN_MEAN_REWARD
        and stochastic_eval["mean_length"] < max_mean_length
    )
    deterministic_quality_passed = (
        deterministic_eval["mean_reward"] > PROMOTION_MIN_MEAN_REWARD
        and deterministic_eval["mean_length"] < max_mean_length
    )
    quality_passed = stochastic_quality_passed or deterministic_quality_passed
    promotion_passed = None
    if promotion_threshold is not None:
        deterministic_floor = promotion_threshold * DETERMINISTIC_PROMOTION_FLOOR_FRACTION
        coverage_passed = (
            stochastic_rate >= promotion_threshold
            and deterministic_rate >= deterministic_floor
        )
        efficiency_ok = stochastic_eval["mean_length"] < 2.0 * (size ** 2)

        promotion_passed = (
            coverage_passed
            and efficiency_ok
            and (
                quality_passed
                or deterministic_rate > 0.7
            )
        )
    else:
        coverage_passed = None

    promotion_rate = stochastic_rate
    diagnostic_rate = max(stochastic_rate, deterministic_rate)
    gap = stochastic_rate - deterministic_rate
    if gap > EVAL_MODE_GAP_MARGIN:
        preferred_mode = "stochastic"
    elif gap < -EVAL_MODE_GAP_MARGIN:
        preferred_mode = "deterministic"
    else:
        preferred_mode = "balanced"

    model.logger.record("eval/stage", stage_idx + 1)
    model.logger.record("eval/stage_timesteps", stage_timesteps)
    model.logger.record("eval/stochastic_full_coverage_rate", stochastic_rate)
    model.logger.record("eval/deterministic_full_coverage_rate", deterministic_rate)
    model.logger.record("eval/promotion_full_coverage_rate", promotion_rate)
    model.logger.record("eval/diagnostic_full_coverage_rate", diagnostic_rate)
    model.logger.record("eval/mode_gap", gap)
    model.logger.record("eval/promotion_min_mean_reward", PROMOTION_MIN_MEAN_REWARD)
    model.logger.record("eval/promotion_max_mean_length", max_mean_length)
    model.logger.record("eval/stochastic_quality_passed", int(stochastic_quality_passed))
    model.logger.record("eval/deterministic_quality_passed", int(deterministic_quality_passed))
    model.logger.record("eval/quality_promotion_passed", int(quality_passed))
    if deterministic_floor is not None:
        model.logger.record("eval/deterministic_promotion_floor", deterministic_floor)
        model.logger.record("eval/coverage_promotion_passed", int(coverage_passed))
        model.logger.record("eval/promotion_passed", int(promotion_passed))
    model.logger.dump(model.num_timesteps)

    return {
        "stochastic_rate": stochastic_rate,
        "deterministic_rate": deterministic_rate,
        "promotion_rate": promotion_rate,
        "deterministic_floor": deterministic_floor,
        "coverage_passed": coverage_passed,
        "quality_passed": quality_passed,
        "stochastic_quality_passed": stochastic_quality_passed,
        "deterministic_quality_passed": deterministic_quality_passed,
        "promotion_min_mean_reward": PROMOTION_MIN_MEAN_REWARD,
        "promotion_max_mean_length": max_mean_length,
        "stochastic_mean_reward": stochastic_eval["mean_reward"],
        "deterministic_mean_reward": deterministic_eval["mean_reward"],
        "stochastic_mean_length": stochastic_eval["mean_length"],
        "deterministic_mean_length": deterministic_eval["mean_length"],
        "promotion_passed": promotion_passed,
        "diagnostic_rate": diagnostic_rate,
        "gap": gap,
        "preferred_mode": preferred_mode,
    }


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ["train", "test", "run", "random"]:
        print("Usage: python train_grid_world_cpp.py <train|test|run|random>")
        sys.exit(1)

    mode = sys.argv[1]
    register_cpp_env()

    if mode == "train":
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

        from CNN import CPPFeatureExtractor
        from goal_conditioned_policy import GoalConditionedMultiInputLstmPolicy

        print("=" * 60)
        print("  CPP Training - RECURRENT PPO CURRICULUM MODE")
        print("=" * 60)
        for i, stage in enumerate(CURRICULUM):
            label = f"{stage['threshold'] * 100:.0f}%" if stage["threshold"] is not None else "budget"
            print(
                f"  Stage {i + 1}: {stage['size']}x{stage['size']} | "
                f"obs={stage['obs']} | max_steps={stage['max_steps']} | "
                f"target={label} | min_ts={stage['min_timesteps']:,}"
            )
        print(f"  Consecutive evals required : {CONSECUTIVE_REQUIRED} back-to-back")
        print(
            f"  Eval gate checked          : every {EVAL_FREQ_STEPS:,} train steps "
            f"({EVAL_EPISODES} deterministic episodes)"
        )
        print("=" * 60)
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"recurrent_ppo_cpp_curriculum_{timestamp}"
        log_dir = f"log/{run_name}"
        model_path = f"data/{run_name}.zip"
        checkpoint_dir = Path(log_dir) / "stage_checkpoints"
        Path("data").mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
        first = CURRICULUM[0]
        raw_check_env = make_cpp_env(first["size"], first["obs"], first["max_steps"])
        try:
            check_env(raw_check_env)
        finally:
            close_managed_env(raw_check_env, "check env")
    
        env_holder = {"env": None}

        def close_current_train_env() -> None:
            close_managed_env(env_holder["env"], "train env")
            env_holder["env"] = None

        atexit.register(close_current_train_env)

        env = VecMonitor(
            make_parallel_cpp_env(
                first["size"],
                first["obs"],
                first["max_steps"],
                purpose="train",
            )
        )
        env_holder["env"] = env
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        env_holder["env"] = env
    
        model = RecurrentPPO(
            GoalConditionedMultiInputLstmPolicy,
            env,
            verbose=1,
            ent_coef=INITAL_ENTROPY_COEF,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            clip_range=CLIP_RANGE,
            gae_lambda=GAE_LAMBDA,
            target_kl=TARGET_KL,
            policy_kwargs=dict(
                features_extractor_class=CPPFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=256),
                goal_dim=16,
                net_arch=POLICY_KWARGS["net_arch"],
                optimizer_kwargs=dict(
                    eps=1e-5,
                    weight_decay=1e-4,
                    betas=ADAM_BETAS,
                ),
            ),
            device=DEVICE,
        )
    
        new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

        cbp = ContinualBackprop(model.policy) if CBP_ENABLED else None
        train_callback = (
            ContinualBackpropCallback(cbp).callback
            if cbp is not None
            else None
        )
    
        print(f"\n  Logging to  : {log_dir}")
        print(f"  Final model : {model_path}\n")
        print(
            f"  Train env backend: {get_vec_env_backend('train')} | "
            f"envs={N_WORKERS} | rollout={N_STEPS * N_WORKERS} steps"
        )
        print(
            f"  Eval env backend : {get_vec_env_backend('eval')} | "
            f"envs={EVAL_WORKERS} | episodes={EVAL_EPISODES}"
        )
        print(f"  Torch device     : {DEVICE}\n")
        if cbp is not None:
            print(
                "  Continual BP     : enabled | "
                f"targets={len(cbp.targets)} | "
                f"rr={CBP_REPLACEMENT_RATE:g} | "
                f"maturity={CBP_MATURITY_THRESHOLD:,} ts"
            )
        else:
            print("  Continual BP     : disabled")

        progress_state = {
            "phase": "training_setup_complete",
            "stage_idx": 0,
            "stage_number": 1,
            "stage_timesteps": 0,
            "global_timesteps": model.num_timesteps,
            "eval_attempt": 0,
            "consecutive_hits": 0,
            "train_vec_env": get_vec_env_backend("train"),
            "eval_vec_env": get_vec_env_backend("eval"),
            "n_workers": N_WORKERS,
            "eval_workers": EVAL_WORKERS,
            "device": DEVICE,
        }

        def remember_progress(phase: str, **kwargs) -> None:
            progress_state.update(kwargs)
            progress_state["phase"] = phase
            progress_state["global_timesteps"] = model.num_timesteps

        def handle_training_exception(exc) -> bool:
            save_emergency_checkpoint(
                model,
                env_holder["env"],
                run_name,
                progress_state,
                exc,
            )
            close_current_train_env()
            if isinstance(exc, KeyboardInterrupt):
                print("  Training interrupted by Ctrl+C after checkpoint save.")
                return True
            return False
    
        for stage_idx, stage in enumerate(CURRICULUM):
            size = stage["size"]
            n_obs = stage["obs"]
            max_steps = stage["max_steps"]
            threshold = stage["threshold"]
            min_ts = stage["min_timesteps"]
            is_final = threshold is None
            remember_progress(
                "stage_start",
                stage_idx=stage_idx,
                stage_number=stage_idx + 1,
                stage=dict(stage),
                stage_timesteps=0,
                eval_attempt=0,
                consecutive_hits=0,
            )
    
            print(f"\n{'=' * 60}")
            print(f"  STAGE {stage_idx + 1}/{len(CURRICULUM)} - {size}x{size} grid")
            print(f"{'=' * 60}")
    
            # Swap envs per stage with fresh reward-normalization statistics.
            # Stage 1 already owns a freshly-created env from model setup, so
            # avoid spawning and immediately deleting an identical worker set.
            if stage_idx > 0:
                remember_progress("stage_env_swap")
                try:
                    close_current_train_env()

                    new_inner = VecMonitor(
                        make_parallel_cpp_env(
                            size,
                            n_obs,
                            max_steps,
                            purpose="train",
                        )
                    )
                    env_holder["env"] = new_inner
                    env = VecNormalize(new_inner, norm_obs=False, norm_reward=True, clip_reward=10.0)
                    env_holder["env"] = env
                    model.set_env(env)
                except KeyboardInterrupt as exc:
                    if handle_training_exception(exc):
                        return
                except Exception as exc:
                    handle_training_exception(exc)
                    raise
    
            stage_timesteps = 0
            entropy_failure_bump = 0.0
            lr_eval_boost = 0.0
            preferred_eval_mode = "balanced"
            first_learn_call = stage_idx == 0

            # Reset Adam state so stale moment estimates from previous stages do
            # not suppress adaptation on the new grid size.
            model.policy.optimizer = model.policy.optimizer.__class__(
                model.policy.parameters(),
                **model.policy.optimizer.defaults,
            )

            # Re-inject exploration at stage entry before the first rollout.
            model.ent_coef = stage["ent_start"]
            set_model_learning_rate(model, stage.get("lr_start", LEARNING_RATE))

            # Seed fresh VecNormalize reward statistics before counting the
            # stage budget. Use gentler PPO updates while the value function
            # recalibrates to the new stage distribution.
            WARMUP_STEPS = 20_000
            original_lr = stage.get("lr_start", LEARNING_RATE)
            original_clip = CLIP_RANGE
            warmup_lr = LEARNING_RATE * 0.5
            warmup_clip_range = CLIP_RANGE * 0.7
            set_model_learning_rate(model, warmup_lr)
            model.clip_range = constant_schedule(warmup_clip_range)
            print(
                f"  Warming up VecNormalize/value function for {WARMUP_STEPS:,} steps "
                f"(lr={warmup_lr:.1e}, clip={warmup_clip_range:.3f})..."
            )
            remember_progress("warmup", stage_timesteps=stage_timesteps)
            try:
                model.learn(
                    total_timesteps=WARMUP_STEPS,
                    reset_num_timesteps=first_learn_call,
                    callback=train_callback,
                )
            except KeyboardInterrupt as exc:
                if handle_training_exception(exc):
                    return
            except Exception as exc:
                handle_training_exception(exc)
                raise
            finally:
                set_model_learning_rate(model, original_lr)
                model.clip_range = constant_schedule(original_clip)
            first_learn_call = False
    
            while True:

                base_ent_coef = linear_stage_value(
                    stage,
                    stage_timesteps,
                    "ent_start",
                    "ent_end",
                    "ent_decay_timesteps",
                )
                model.ent_coef = min(
                    stage["ent_start"],
                    base_ent_coef + entropy_failure_bump,
                )
                current_lr = linear_stage_value(
                    stage,
                    stage_timesteps,
                    "lr_start",
                    "lr_end",
                    "lr_decay_timesteps",
                )
                current_lr = min(
                    stage.get("lr_start", LEARNING_RATE),
                    current_lr + lr_eval_boost,
                )
                set_model_learning_rate(model, current_lr)
                model.logger.record("curriculum/stage", stage_idx + 1)
                model.logger.record("curriculum/grid_size", size)
                model.logger.record("curriculum/obs_quantity", n_obs)
                model.logger.record("curriculum/max_steps", max_steps)
                model.logger.record("curriculum/stage_timesteps", stage_timesteps)
                model.logger.record("curriculum/preferred_eval_mode", eval_mode_code(preferred_eval_mode))
                model.logger.record("train/ent_coef", model.ent_coef)
                model.logger.record("train/stage_learning_rate", current_lr)

                before_timesteps = model.num_timesteps
                remember_progress("training", stage_timesteps=stage_timesteps)
                try:
                    model.learn(
                        total_timesteps=EVAL_FREQ_STEPS,
                        reset_num_timesteps=first_learn_call,
                        callback=train_callback,
                    )
                except KeyboardInterrupt as exc:
                    if handle_training_exception(exc):
                        return
                except Exception as exc:
                    handle_training_exception(exc)
                    raise
                first_learn_call = False
                stage_timesteps += model.num_timesteps - before_timesteps
                remember_progress("post_training_chunk", stage_timesteps=stage_timesteps)
                record_plasticity_metrics(model)
                if cbp is not None:
                    cbp.record(model.logger)
                model.logger.dump(model.num_timesteps)
    
                if is_final:
                    print(
                        f"  [stage {stage_idx + 1}] {stage_timesteps:,} / "
                        f"{min_ts:,} timesteps",
                        end="\r",
                    )
                    if stage_timesteps >= min_ts:
                        print("\n  Final stage budget reached. Done.")
                        break
                    continue
    
                if stage_timesteps < min_ts:
                    print(
                        f"  [stage {stage_idx + 1}] {stage_timesteps:,} ts - "
                        f"warming up (min {min_ts:,})"
                    )
                    continue
    
                consecutive_hits = 0
                while consecutive_hits < CONSECUTIVE_REQUIRED:
                    eval_attempt = consecutive_hits + 1
                    remember_progress(
                        "evaluation",
                        stage_timesteps=stage_timesteps,
                        eval_attempt=eval_attempt,
                        consecutive_hits=consecutive_hits,
                        preferred_eval_mode=preferred_eval_mode,
                    )
                    print(
                        f"  Validation attempt "
                        f"{eval_attempt}/{CONSECUTIVE_REQUIRED}"
                    )
                    try:
                        eval_result = run_dual_coverage_eval(
                            model,
                            size,
                            n_obs,
                            max_steps,
                            stage_timesteps=stage_timesteps,
                            stage_idx=stage_idx,
                            promotion_threshold=threshold,
                            progress_callback=lambda completed: remember_progress(
                                "evaluation",
                                stage_timesteps=stage_timesteps,
                                eval_attempt=eval_attempt,
                                eval_completed_episodes=completed,
                                consecutive_hits=consecutive_hits,
                                preferred_eval_mode=preferred_eval_mode,
                            ),
                        )
                    except KeyboardInterrupt as exc:
                        if handle_training_exception(exc):
                            return
                    except Exception as exc:
                        handle_training_exception(exc)
                        raise
                    rate = eval_result["promotion_rate"]
                    preferred_eval_mode = eval_result["preferred_mode"]
                    remember_progress(
                        "evaluation_result",
                        stage_timesteps=stage_timesteps,
                        eval_attempt=eval_attempt,
                        consecutive_hits=consecutive_hits,
                        last_eval_rate=rate,
                        last_stochastic_eval_rate=eval_result["stochastic_rate"],
                        last_deterministic_eval_rate=eval_result["deterministic_rate"],
                        last_diagnostic_eval_rate=eval_result["diagnostic_rate"],
                        deterministic_promotion_floor=eval_result["deterministic_floor"],
                        coverage_promotion_passed=eval_result["coverage_passed"],
                        quality_promotion_passed=eval_result["quality_passed"],
                        promotion_passed=eval_result["promotion_passed"],
                        stochastic_mean_reward=eval_result["stochastic_mean_reward"],
                        deterministic_mean_reward=eval_result["deterministic_mean_reward"],
                        stochastic_mean_length=eval_result["stochastic_mean_length"],
                        deterministic_mean_length=eval_result["deterministic_mean_length"],
                        promotion_max_mean_length=eval_result["promotion_max_mean_length"],
                        preferred_eval_mode=preferred_eval_mode,
                    )
                    entropy_bump = stage.get("ent_failure_bump", 0.0)

                    if preferred_eval_mode == "stochastic" and entropy_bump > 0.0:
                        entropy_failure_bump = min(
                            entropy_failure_bump + entropy_bump,
                            max(stage["ent_start"] - stage["ent_end"], 0.0),
                        )
                        lr_eval_boost = min(
                            lr_eval_boost + LEARNING_RATE * 0.25,
                            max(stage.get("lr_start", LEARNING_RATE) - LEARNING_RATE, 0.0),
                        )
                    elif preferred_eval_mode == "deterministic":
                        entropy_failure_bump = max(0.0, entropy_failure_bump - entropy_bump)
                        lr_eval_boost = max(0.0, lr_eval_boost - LEARNING_RATE * 0.25)

                    if not eval_result["promotion_passed"]:
                        if entropy_bump > 0.0 and preferred_eval_mode == "balanced":
                            entropy_failure_bump = min(
                                entropy_failure_bump + entropy_bump,
                                max(stage["ent_start"] - stage["ent_end"], 0.0),
                            )
                        model.ent_coef = min(
                            stage["ent_start"],
                            linear_stage_value(
                                stage,
                                stage_timesteps,
                                "ent_start",
                                "ent_end",
                                "ent_decay_timesteps",
                            ) + entropy_failure_bump,
                        )
                        print(
                            f"    target: {threshold * 100:.0f}% | "
                            f"stochastic score: {rate * 100:.1f}% | "
                            "validation failed; training continues"
                        )
                        if entropy_bump > 0.0:
                            print(f"    next entropy target: {model.ent_coef:.4f}")
                        print(
                            f"    stochastic: {eval_result['stochastic_rate'] * 100:.1f}% | "
                            f"deterministic: {eval_result['deterministic_rate'] * 100:.1f}% "
                            f"(floor {eval_result['deterministic_floor'] * 100:.1f}%) | "
                            f"next bias: {preferred_eval_mode}"
                        )
                        print(
                            f"    quality gate: reward>{PROMOTION_MIN_MEAN_REWARD:.2f}, "
                            f"steps<{eval_result['promotion_max_mean_length']:.1f} | "
                            f"stoch r/l={eval_result['stochastic_mean_reward']:.2f}/"
                            f"{eval_result['stochastic_mean_length']:.1f} | "
                            f"det r/l={eval_result['deterministic_mean_reward']:.2f}/"
                            f"{eval_result['deterministic_mean_length']:.1f}"
                        )
                        break
    
                    consecutive_hits += 1
                    remember_progress(
                        "evaluation_hit",
                        stage_timesteps=stage_timesteps,
                        eval_attempt=eval_attempt,
                        consecutive_hits=consecutive_hits,
                        last_eval_rate=rate,
                        last_stochastic_eval_rate=eval_result["stochastic_rate"],
                        last_deterministic_eval_rate=eval_result["deterministic_rate"],
                        deterministic_promotion_floor=eval_result["deterministic_floor"],
                        coverage_promotion_passed=eval_result["coverage_passed"],
                        quality_promotion_passed=eval_result["quality_passed"],
                        promotion_passed=eval_result["promotion_passed"],
                        stochastic_mean_reward=eval_result["stochastic_mean_reward"],
                        deterministic_mean_reward=eval_result["deterministic_mean_reward"],
                        stochastic_mean_length=eval_result["stochastic_mean_length"],
                        deterministic_mean_length=eval_result["deterministic_mean_length"],
                        promotion_max_mean_length=eval_result["promotion_max_mean_length"],
                        preferred_eval_mode=preferred_eval_mode,
                    )
                    print(
                        f"    target: {threshold * 100:.0f}% | "
                        f"stochastic score: {rate * 100:.1f}% | "
                        f"streak: {consecutive_hits}/{CONSECUTIVE_REQUIRED}"
                    )
                    print(
                        f"    stochastic: {eval_result['stochastic_rate'] * 100:.1f}% | "
                        f"deterministic: {eval_result['deterministic_rate'] * 100:.1f}% "
                        f"(floor {eval_result['deterministic_floor'] * 100:.1f}%) | "
                        f"next bias: {preferred_eval_mode}"
                    )
                    print(
                        f"    gate passed: "
                        f"{'coverage' if eval_result['coverage_passed'] else 'quality'} | "
                        f"quality reward>{PROMOTION_MIN_MEAN_REWARD:.2f}, "
                        f"steps<{eval_result['promotion_max_mean_length']:.1f}"
                    )
    
                if consecutive_hits == CONSECUTIVE_REQUIRED:
                    print("  Promotion threshold met. Advancing to next stage.")
                    break
    
            stage_checkpoint = f"data/{run_name}_stage{stage_idx + 1}_{size}x{size}.zip"
            log_checkpoint = (
                checkpoint_dir
                / (
                    f"{run_name}_stage{stage_idx + 1}_trained_"
                    f"{size}x{size}_obs{n_obs}_max{max_steps}.zip"
                )
            )
            model.save(stage_checkpoint)
            model.save(str(log_checkpoint))
            print(f"  Stage checkpoint saved -> {stage_checkpoint}")
            print(f"  Log checkpoint saved   -> {log_checkpoint}")
            model.save(model_path)
        env.save(f"data/{run_name}_vecnorm.pkl")
        close_current_train_env()
        print(f"\n  Final model saved -> {model_path}")
        print("  Training complete.")
    
    
    elif mode == "run":
        from sb3_contrib import RecurrentPPO

        model_name = input("Enter model filename (without .zip): ")
        model_path, best_model_path = resolve_model_paths(model_name)
        load_path = best_model_path or model_path
        print(f"--- Loading model from {load_path}.zip for a visual run ---")
    
        model = RecurrentPPO.load(load_path)
        env = make_cpp_env(TEST_DIM, TEST_OBSTACLES, TEST_MAX_STEPS, render_mode="human")

        try:
            obs, _ = env.reset()
            done = False
            truncated = False
            steps = 0
            lstm_states = None
            episode_starts = np.array([True], dtype=bool)
            while not done and not truncated:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                obs, reward, done, truncated, info = env.step(action.item())
                episode_starts = np.array([done or truncated], dtype=bool)
                print(
                    f"Step {steps + 1:3d} | Action: {print_action(action.item()):5s} | "
                    f"Reward: {reward:+.3f} | Coverage: {info['coverage'] * 100:.1f}% "
                    f"({info['visited_cells']}/{info['total_free_cells']} cells)"
                )
                steps += 1

            final_coverage = info["coverage"] * 100
            print("\n--- Run Finished ---")
            print(f"Final coverage: {final_coverage:.1f}% in {steps} steps.")
            if done:
                print("Result: FULL COVERAGE ACHIEVED")
            else:
                print("Result: Truncated - coverage incomplete.")
        finally:
            close_managed_env(env, "run env")
    
    
    elif mode == "test":
        from sb3_contrib import RecurrentPPO

        test_count = 0
        model_name = input("Enter model filename (without .zip): ")
        model_path, best_model_path = resolve_model_paths(model_name)
        load_path = best_model_path or model_path
        print(f"--- Loading model from {load_path}.zip for batch testing ---")
    
        model = RecurrentPPO.load(load_path)
        env = make_cpp_env(TEST_DIM, TEST_OBSTACLES, TEST_MAX_STEPS)

        try:
            num_episodes = 1000
            full_coverage_count = 0
            coverage_rates = []
            episode_steps = []
            rewards = []

            for ep in range(num_episodes):
                total_reward = 0
                total_steps = 0
                obs, _ = env.reset()
                done = False
                truncated = False
                info = {}
                lstm_states = None
                episode_starts = np.array([True], dtype=bool)
                while not done and not truncated:
                    action, lstm_states = model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=TEST_DETERMINISTIC,
                    )
                    obs, reward, done, truncated, info = env.step(action.item())
                    episode_starts = np.array([done or truncated], dtype=bool)
                    total_reward += reward
                    total_steps += 1

                rewards.append(total_reward)
                episode_steps.append(total_steps)
                coverage = info.get("coverage", 0.0) * 100
                coverage_rates.append(coverage)
                if done:
                    full_coverage_count += 1
                test_count += 1
                print(f"Test progress: {test_count}/1000")

            episode_steps = np.array(episode_steps)
            rewards = np.array(rewards)
            print("\n--- Test Finished ---")
            print(
                f"  Full coverage rate : {full_coverage_count}/{num_episodes} "
                f"({full_coverage_count / num_episodes * 100:.1f}%)"
            )
            print(f"  Mean coverage      : {sum(coverage_rates) / len(coverage_rates):.1f}%")
            print(f"  Min / Max coverage : {min(coverage_rates):.1f}% / {max(coverage_rates):.1f}%")
            print(f"  Average Reward: {rewards.mean():.2f} +/- {rewards.std():.2f}")
            print(f"  Average Length: {episode_steps.mean():.2f} +/- {episode_steps.std():.2f}")
            print("-" * 60)
        finally:
            close_managed_env(env, "test env")
    
    
    elif mode == "random":
        print("=" * 60)
        print("  Random Agent Baseline")
        print("=" * 60)
        print(f"  Grid: {TEST_DIM}x{TEST_DIM} | Obstacles: {TEST_OBSTACLES} | Max steps: {TEST_MAX_STEPS}")
        print("=" * 60)
    
        env = make_cpp_env(TEST_DIM, TEST_OBSTACLES, TEST_MAX_STEPS)

        try:
            num_episodes = 1000
            full_coverage_count = 0
            coverage_rates = []

            for ep in range(num_episodes):
                obs, _ = env.reset()
                done = False
                truncated = False
                last_info = {}
                while not done and not truncated:
                    action = env.action_space.sample()
                    obs, reward, done, truncated, last_info = env.step(action)

                coverage = last_info.get("coverage", 0.0) * 100
                coverage_rates.append(coverage)
                if done:
                    full_coverage_count += 1

            print("\n--- Random Agent Baseline ---")
            print(
                f"  Full coverage rate : {full_coverage_count}/{num_episodes} "
                f"({full_coverage_count / num_episodes * 100:.1f}%)"
            )
            print(f"  Mean coverage      : {sum(coverage_rates) / len(coverage_rates):.1f}%")
            print(f"  Min / Max coverage : {min(coverage_rates):.1f}% / {max(coverage_rates):.1f}%")
            print("-" * 60)
        finally:
            close_managed_env(env, "random env")


if __name__ == "__main__":
    main()
