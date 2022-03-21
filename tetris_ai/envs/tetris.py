from typing import Optional, Union

import gym
import numpy as np
import pygame
from tetris_ai.game import Tetris
from tetris_ai.veiw import TetrisView


class TetrisEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self):
        self.game = None
        self.screen = None
        self.action_space = gym.spaces.Discrete(8)

    def step(self, action):
        assert self.action_space.contains(action)
        self.game.step(action)
        reward = 1.0
        done = False
        return np.array([]), reward, done, {}

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)

        if self.game is None:
            self.game = Tetris()

        self.game.reset()

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
