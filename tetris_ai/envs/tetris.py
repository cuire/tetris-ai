from typing import Optional, Union

import gym
import numpy as np
import pygame

from tetris_ai.game import Tetris
from tetris_ai.view import TetrisView


class TetrisEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self):
        self.game = Tetris()
        self.screen = None
        self.state = None
        self.action_space = gym.spaces.Discrete(7)

        low_state = np.zeros((self.game.grid_size_y * self.game.grid_size_x), dtype=int)
        high_state = np.ones((self.game.grid_size_y * self.game.grid_size_x), dtype=int)

        self.observation_space = gym.spaces.Box(
            low=low_state, high=high_state, dtype=int
        )

    def step(self, action):
        assert self.action_space.contains(action)
        assert self.state is not None, "Call reset before using step method."
        self.state, reward, done = self.game.step(action)
        reward += 1 if not done else -5
        return self.state, reward, done, {}

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False):
        super().reset(seed=seed)
        self.game.reset()
        self.state = self.game.get_current_state()

        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

    def render(self, mode="human", width=600, height=600):

        if self.game is None:
            return None

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Tetris")
            self.screen = TetrisView(game=self.game, parent=screen)

        self.screen.render()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
