# python continue_latest_stage_training.py [run_name_or_checkpoint_path]
#
# Resume curriculum training from the newest saved point for a CPP model.
# Prefer an interrupt checkpoint because it carries stage progress and
# VecNormalize state; otherwise continue after the newest completed stage.

import argparse
import atexit
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from train_grid_world_cpp import (
    CBP_ENABLED,
    CLIP_RANGE,
    CONSECUTIVE_REQUIRED,
    CURRICULUM,
    DEVICE,
    EVAL_FREQ_STEPS,
    LEARNING_RATE,
    N_STEPS,
    N_WORKERS,
    PROMOTION_MIN_MEAN_REWARD,
    ContinualBackprop,
    ContinualBackpropCallback,
    close_managed_env,
    constant_schedule,
    get_vec_env_backend,
    linear_stage_value,
    make_parallel_cpp_env,
    record_plasticity_metrics,
    register_cpp_env,
    run_dual_coverage_eval,
    save_emergency_checkpoint,
    set_model_learning_rate,
)

STAGE_CHECKPOINT_RE = re.compile(r"^(?P<run>.+)_stage(?P<stage>\d+)_(?P<size>\d+)x(?P<size2>\d+)\.zip$")


@dataclass
class ResumePoint:
    run_name: str
    model_path: Path
    stage_idx: int
    stage_timesteps: int
    source: str
    mtime: float
    vecnorm_path: Path | None = None
    completed_stage_checkpoint: bool = False


def existing_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None

    path = Path(value)
    if path.exists():
        return path

    if path.suffix != ".zip" and path.with_suffix(".zip").exists():
        return path.with_suffix(".zip")

    return None


def read_interrupt_resume(
    metadata_path: Path,
    model_override: Path | None = None,
) -> ResumePoint | None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  Skipping unreadable metadata {metadata_path}: {exc}")
        return None

    run_name = metadata.get("run_name")
    if not run_name:
        return None

    model_path = model_override or existing_path(
        metadata.get("latest_model_path") or metadata.get("model_path")
    )
    if model_path is None:
        return None

    vecnorm_path = existing_path(
        metadata.get("latest_vecnorm_path") or metadata.get("vecnorm_path")
    )
    mtime = max(metadata_path.stat().st_mtime, model_path.stat().st_mtime)

    return ResumePoint(
        run_name=run_name,
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        stage_idx=int(metadata.get("stage_idx", 0)),
        stage_timesteps=int(metadata.get("stage_timesteps", 0)),
        source=f"interrupt metadata {metadata_path}",
        mtime=mtime,
        completed_stage_checkpoint=False,
    )


def read_stage_checkpoint(path: Path) -> ResumePoint | None:
    match = STAGE_CHECKPOINT_RE.match(path.name)
    if match is None:
        return None

    stage_number = int(match.group("stage"))
    next_stage_idx = stage_number
    if next_stage_idx >= len(CURRICULUM):
        next_stage_idx = len(CURRICULUM)

    return ResumePoint(
        run_name=match.group("run"),
        model_path=path,
        vecnorm_path=None,
        stage_idx=next_stage_idx,
        stage_timesteps=0,
        source=f"completed stage checkpoint {path}",
        mtime=path.stat().st_mtime,
        completed_stage_checkpoint=True,
    )


def metadata_for_interrupt_checkpoint(path: Path) -> Path | None:
    exact_metadata = path.with_name(f"{path.stem}_metadata.json")
    if exact_metadata.exists():
        return exact_metadata

    if path.stem.endswith("_interrupt_latest"):
        latest_metadata = path.with_name(f"{path.stem}_metadata.json")
        if latest_metadata.exists():
            return latest_metadata

    return None


def read_checkpoint_path(path: Path) -> ResumePoint:
    if path.suffix.lower() == ".json":
        point = read_interrupt_resume(path)
        if point is None:
            raise ValueError(f"Could not read interrupt metadata from {path}")
        return point

    if path.suffix.lower() != ".zip":
        raise ValueError(f"Checkpoint path must be a .zip model or .json metadata file: {path}")

    point = read_stage_checkpoint(path)
    if point is not None:
        return point

    metadata_path = metadata_for_interrupt_checkpoint(path)
    if metadata_path is not None:
        point = read_interrupt_resume(metadata_path, model_override=path)
        if point is not None:
            point.source = f"interrupt checkpoint {path}"
            return point

    raise ValueError(
        "Could not infer curriculum stage from checkpoint path. "
        "Pass the matching *_metadata.json file, or use the run name instead."
    )


def find_resume_point(source: str | None) -> ResumePoint:
    if source:
        source_path = Path(source)
        if source_path.exists():
            return read_checkpoint_path(source_path)

    data_dir = Path("data")
    candidates: list[ResumePoint] = []

    for metadata_path in data_dir.glob("*_interrupt_latest_metadata.json"):
        point = read_interrupt_resume(metadata_path)
        if point is not None and (source is None or point.run_name == source):
            candidates.append(point)

    for checkpoint_path in data_dir.glob("*_stage*_*.zip"):
        point = read_stage_checkpoint(checkpoint_path)
        if point is not None and (source is None or point.run_name == source):
            candidates.append(point)

    if not candidates:
        hint = f" for {source!r}" if source else ""
        raise FileNotFoundError(f"No interrupt or stage checkpoint found{hint}.")

    return max(candidates, key=lambda point: point.mtime)


def make_training_env(stage: dict, vecnorm_path: Path | None = None):
    from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

    inner = VecMonitor(
        make_parallel_cpp_env(
            stage["size"],
            stage["obs"],
            stage["max_steps"],
            purpose="train",
        )
    )

    if vecnorm_path is not None and vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), inner)
        env.training = True
        env.norm_reward = True
        return env

    return VecNormalize(inner, norm_obs=False, norm_reward=True, clip_reward=10.0)


def save_stage_checkpoint(model, run_name: str, log_checkpoint_dir: Path,
                          stage_idx: int, stage: dict) -> None:
    size = stage["size"]
    n_obs = stage["obs"]
    max_steps = stage["max_steps"]
    stage_checkpoint = Path("data") / f"{run_name}_stage{stage_idx + 1}_{size}x{size}.zip"
    log_checkpoint = (
        log_checkpoint_dir
        / (
            f"{run_name}_stage{stage_idx + 1}_trained_"
            f"{size}x{size}_obs{n_obs}_max{max_steps}.zip"
        )
    )
    model.save(str(stage_checkpoint))
    model.save(str(log_checkpoint))
    print(f"  Stage checkpoint saved -> {stage_checkpoint}")
    print(f"  Log checkpoint saved   -> {log_checkpoint}")


def reset_optimizer_for_stage(model) -> None:
    model.policy.optimizer = model.policy.optimizer.__class__(
        model.policy.parameters(),
        **model.policy.optimizer.defaults,
    )


def warmup_new_stage(model, stage: dict, train_callback, remember_progress) -> None:
    warmup_steps = 20_000
    original_lr = stage.get("lr_start", LEARNING_RATE)
    original_clip = CLIP_RANGE
    warmup_lr = LEARNING_RATE * 0.5
    warmup_clip_range = CLIP_RANGE * 0.7

    set_model_learning_rate(model, warmup_lr)
    model.clip_range = constant_schedule(warmup_clip_range)
    print(
        f"  Warming up VecNormalize/value function for {warmup_steps:,} steps "
        f"(lr={warmup_lr:.1e}, clip={warmup_clip_range:.3f})..."
    )
    remember_progress("warmup")
    try:
        model.learn(
            total_timesteps=warmup_steps,
            reset_num_timesteps=False,
            callback=train_callback,
        )
    finally:
        set_model_learning_rate(model, original_lr)
        model.clip_range = constant_schedule(original_clip)


def continue_training(resume: ResumePoint, output_run_name: str | None = None) -> None:
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.logger import configure

    # Imported for custom class resolution while loading saved models.
    from CNN import CPPFeatureExtractor  # noqa: F401
    from goal_conditioned_policy import GoalConditionedMultiInputLstmPolicy  # noqa: F401

    if resume.stage_idx >= len(CURRICULUM):
        print("  The latest checkpoint is already at or beyond the final curriculum stage.")
        print(f"  Checkpoint: {resume.model_path}")
        return

    register_cpp_env()
    Path("data").mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_run_name = output_run_name or f"{resume.run_name}_continued_{timestamp}"
    log_dir = Path("log") / output_run_name
    checkpoint_dir = log_dir / "stage_checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_holder = {"env": None}

    def close_current_train_env() -> None:
        close_managed_env(env_holder["env"], "train env")
        env_holder["env"] = None

    atexit.register(close_current_train_env)

    first_stage = CURRICULUM[resume.stage_idx]
    env = make_training_env(
        first_stage,
        resume.vecnorm_path if not resume.completed_stage_checkpoint else None,
    )
    env_holder["env"] = env

    model = RecurrentPPO.load(str(resume.model_path), env=env, device=DEVICE)
    model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))

    cbp = ContinualBackprop(model.policy) if CBP_ENABLED else None
    train_callback = ContinualBackpropCallback(cbp).callback if cbp is not None else None

    progress_state = {
        "phase": "resume_setup_complete",
        "source_run_name": resume.run_name,
        "run_name": output_run_name,
        "resume_source": resume.source,
        "stage_idx": resume.stage_idx,
        "stage_number": resume.stage_idx + 1,
        "stage_timesteps": resume.stage_timesteps,
        "global_timesteps": model.num_timesteps,
        "eval_attempt": 0,
        "consecutive_hits": 0,
        "train_vec_env": get_vec_env_backend("train"),
        "eval_vec_env": get_vec_env_backend("eval"),
        "n_workers": N_WORKERS,
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
            output_run_name,
            progress_state,
            exc,
        )
        close_current_train_env()
        if isinstance(exc, KeyboardInterrupt):
            print("  Training interrupted by Ctrl+C after checkpoint save.")
            return True
        return False

    print("=" * 60)
    print("  CPP Training - CONTINUE LATEST STAGE")
    print("=" * 60)
    print(f"  Source run       : {resume.run_name}")
    print(f"  Resume source    : {resume.source}")
    print(f"  Loaded model     : {resume.model_path}")
    if resume.vecnorm_path is not None and resume.vecnorm_path.exists():
        print(f"  Loaded vecnorm   : {resume.vecnorm_path}")
    print(f"  Output run       : {output_run_name}")
    print(f"  Logging to       : {log_dir}")
    print(
        f"  Train env backend: {get_vec_env_backend('train')} | "
        f"envs={N_WORKERS} | rollout={N_STEPS * N_WORKERS} steps"
    )
    print(f"  Torch device     : {DEVICE}")
    print("=" * 60)

    try:
        for stage_idx in range(resume.stage_idx, len(CURRICULUM)):
            stage = CURRICULUM[stage_idx]
            size = stage["size"]
            n_obs = stage["obs"]
            max_steps = stage["max_steps"]
            threshold = stage["threshold"]
            min_ts = stage["min_timesteps"]
            is_final = threshold is None
            continuing_same_stage = (
                stage_idx == resume.stage_idx
                and not resume.completed_stage_checkpoint
            )

            if continuing_same_stage:
                stage_timesteps = resume.stage_timesteps
            else:
                stage_timesteps = 0
                if stage_idx != resume.stage_idx:
                    close_current_train_env()
                    env = make_training_env(stage)
                    env_holder["env"] = env
                    model.set_env(env)

            remember_progress(
                "stage_start",
                stage_idx=stage_idx,
                stage_number=stage_idx + 1,
                stage=dict(stage),
                stage_timesteps=stage_timesteps,
                eval_attempt=0,
                consecutive_hits=0,
            )

            print(f"\n{'=' * 60}")
            print(f"  STAGE {stage_idx + 1}/{len(CURRICULUM)} - {size}x{size} grid")
            print(f"{'=' * 60}")

            entropy_failure_bump = 0.0
            lr_eval_boost = 0.0
            preferred_eval_mode = "balanced"

            if continuing_same_stage:
                print(
                    f"  Continuing within stage {stage_idx + 1} "
                    f"from {stage_timesteps:,} saved stage timesteps."
                )
            else:
                reset_optimizer_for_stage(model)
                model.ent_coef = stage["ent_start"]
                set_model_learning_rate(model, stage.get("lr_start", LEARNING_RATE))
                try:
                    warmup_new_stage(
                        model,
                        stage,
                        train_callback,
                        lambda phase, **kwargs: remember_progress(
                            phase,
                            stage_timesteps=stage_timesteps,
                            **kwargs,
                        ),
                    )
                except KeyboardInterrupt as exc:
                    if handle_training_exception(exc):
                        return
                except Exception as exc:
                    handle_training_exception(exc)
                    raise

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
                model.logger.record("train/ent_coef", model.ent_coef)
                model.logger.record("train/stage_learning_rate", current_lr)

                before_timesteps = model.num_timesteps
                remember_progress("training", stage_timesteps=stage_timesteps)
                try:
                    model.learn(
                        total_timesteps=EVAL_FREQ_STEPS,
                        reset_num_timesteps=False,
                        callback=train_callback,
                    )
                except KeyboardInterrupt as exc:
                    if handle_training_exception(exc):
                        return
                except Exception as exc:
                    handle_training_exception(exc)
                    raise

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

                if consecutive_hits == CONSECUTIVE_REQUIRED:
                    print("  Promotion threshold met. Advancing to next stage.")
                    break

            save_stage_checkpoint(model, output_run_name, checkpoint_dir, stage_idx, stage)
        final_model_path = Path("data") / f"{output_run_name}.zip"
        final_vecnorm_path = Path("data") / f"{output_run_name}_vecnorm.pkl"
        model.save(str(final_model_path))
        env_holder["env"].save(str(final_vecnorm_path))
        print(f"\n  Final model saved -> {final_model_path}")
        print(f"  Final vecnorm saved -> {final_vecnorm_path}")
        print("  Continued training complete.")
    finally:
        close_current_train_env()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue CPP curriculum training from the newest saved stage."
    )
    parser.add_argument(
        "source",
        nargs="?",
        help=(
            "Optional source run name, .zip checkpoint path, or .json metadata "
            "path. If omitted, the newest interrupt or stage checkpoint in "
            "data/ is used."
        ),
    )
    parser.add_argument(
        "--output-run-name",
        help="Optional name for newly saved checkpoints and logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resume = find_resume_point(args.source)
    continue_training(resume, output_run_name=args.output_run_name)


if __name__ == "__main__":
    main()
