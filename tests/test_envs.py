import gym
import numpy as np
import pytest

from tetris_ai.envs import TetrisEnv


def test_env_creation(tetris_env):
    assert tetris_env.game is not None


def test_env_reset(tetris_env):
    tetris_env.reset()


def test_env_reset_return(tetris_env):
    assert tetris_env.reset(return_info=True)[1] == {}


def test_env_step_without_reset(tetris_env):
    with pytest.raises(AssertionError):
        tetris_env.step(action=1)


@pytest.mark.parametrize("action", [0, 1, 2, 3, 4, 5, 6])
def test_env_step_with_valid_action(tetris_env, action):
    tetris_env.reset()
    tetris_env.step(action)


def test_env_render_without_reset(tetris_env):
    with pytest.raises(AssertionError):
        tetris_env.render()


def test_env_render_human_mode(tetris_env):
    tetris_env.reset()
    tetris_env.render()
    tetris_env.render()
    tetris_env.close()


def test_env_render_rgb_array_mode(tetris_env):
    tetris_env.reset()
    tetris_env.render(mode="rgb_array")
    tetris_env.close()
