import multiprocessing as mp
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

from cpp_subproc_worker import CloudpickleWrapper, cpp_subproc_worker


class CppSubprocVecEnv(VecEnv):
    """SubprocVecEnv variant whose worker process does not import SB3/Torch."""

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        start_method: str | None = None,
        join_timeout: float = 5.0,
    ):
        self.waiting = False
        self.closed = False
        self.join_timeout = join_timeout

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in env_fns], strict=True)
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns, strict=True):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=cpp_subproc_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self._recv_checked(self.remotes[0])

        super().__init__(len(env_fns), observation_space, action_space)

    def _recv_checked(self, remote):
        result = remote.recv()
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and result[0] == "worker_exception"
        ):
            raise RuntimeError(f"Subprocess env failed:\n{result[1]}")
        return result

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions, strict=True):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [self._recv_checked(remote) for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results, strict=True)
        return (
            _stack_obs(obs, self.observation_space),
            np.stack(rews),
            np.stack(dones),
            infos,
        )

    def reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [self._recv_checked(remote) for remote in self.remotes]
        obs, self.reset_infos = zip(*results, strict=True)
        self._reset_seeds()
        self._reset_options()
        return _stack_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return

        if self.waiting:
            for remote in self.remotes:
                try:
                    if remote.poll(self.join_timeout):
                        self._recv_checked(remote)
                except (EOFError, BrokenPipeError, OSError, RuntimeError):
                    pass
            self.waiting = False

        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (EOFError, BrokenPipeError, OSError):
                pass

        for process in self.processes:
            process.join(timeout=self.join_timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout=self.join_timeout)

        for remote in self.remotes:
            remote.close()

        self.closed = True

    def get_images(self) -> Sequence[np.ndarray | None]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.remotes]
        for remote in self.remotes:
            remote.send(("render", None))
        return [self._recv_checked(remote) for remote in self.remotes]

    def has_attr(self, attr_name: str) -> bool:
        target_remotes = self._get_target_remotes(indices=None)
        for remote in target_remotes:
            remote.send(("has_attr", attr_name))
        return all(self._recv_checked(remote) for remote in target_remotes)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [self._recv_checked(remote) for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            self._recv_checked(remote)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [self._recv_checked(remote) for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        target_remotes = self._get_target_remotes(indices)
        wrapper_id = (wrapper_class.__module__, wrapper_class.__qualname__)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_id))
        return [self._recv_checked(remote) for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> list[Any]:
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _stack_obs(obs_list: list[VecEnvObs] | tuple[VecEnvObs], space: spaces.Space) -> VecEnvObs:
    assert isinstance(obs_list, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs_list) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(space.spaces, dict), "Dict space must have ordered subspaces"
        assert isinstance(obs_list[0], dict), "non-dict observation for environment with Dict observation space"
        return {
            key: np.stack([single_obs[key] for single_obs in obs_list])
            for key in space.spaces.keys()
        }
    if isinstance(space, spaces.Tuple):
        assert isinstance(obs_list[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([single_obs[i] for single_obs in obs_list]) for i in range(obs_len))
    return np.stack(obs_list)
