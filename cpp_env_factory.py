import gymnasium as gym

from gymnasium_env.grid_world_cpp import GridWorldCPPEnv


def register_cpp_env():
    env_id = "gymnasium_env/GridWorldCPP-v0"
    try:
        gym.spec(env_id)
        return
    except gym.error.Error:
        gym.register(
            id=env_id,
            entry_point=GridWorldCPPEnv,
        )


def make_cpp_env(size: int, obs_quantity: int, max_steps: int, render_mode=None):
    # Centralized env factory so training, eval, run, and test modes use the
    # same constructor while freely changing grid size and obstacle count.
    register_cpp_env()
    return gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=size,
        obs_quantity=obs_quantity,
        max_steps=max_steps,
        render_mode=render_mode,
    )


def make_cpp_env_factory(size: int, obs_quantity: int, max_steps: int):
    def _init():
        return make_cpp_env(size, obs_quantity, max_steps)

    return _init
