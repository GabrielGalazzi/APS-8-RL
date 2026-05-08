import traceback
from typing import Any

import cloudpickle


class CloudpickleWrapper:
    """Use cloudpickle for env factories without importing SB3 in workers."""

    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> bytes:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: bytes) -> None:
        self.var = cloudpickle.loads(var)


def get_env_attr(env, name: str):
    if hasattr(env, "get_wrapper_attr"):
        return env.get_wrapper_attr(name)
    return getattr(env, name)


def env_has_attr(env, name: str) -> bool:
    try:
        get_env_attr(env, name)
        return True
    except AttributeError:
        return False


def is_wrapped(env, wrapper_id: tuple[str, str]) -> bool:
    current = env
    while True:
        for cls in current.__class__.mro():
            if (cls.__module__, cls.__qualname__) == wrapper_id:
                return True
        if (
            hasattr(current, "class_name")
            and ("stable_baselines3.common.monitor", current.class_name()) == wrapper_id
        ):
            return True
        if not hasattr(current, "env"):
            return False
        current = current.env


def cpp_subproc_worker(remote, parent_remote, env_fn_wrapper: CloudpickleWrapper) -> None:
    parent_remote.close()
    env = None
    reset_info = {}

    try:
        env = env_fn_wrapper.var()
        while True:
            cmd, data = remote.recv()

            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                if env is not None:
                    env.close()
                    env = None
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = get_env_attr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(get_env_attr(env, data))
            elif cmd == "has_attr":
                remote.send(env_has_attr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:
        try:
            remote.send(("worker_exception", traceback.format_exc()))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
