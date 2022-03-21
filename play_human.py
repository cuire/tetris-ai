from typing import Tuple

import gym
import pygame

from tetris_ai.envs import TetrisEnv


def handle_events() -> Tuple[int, bool]:
    running = True
    action = 0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                action = 1
            elif event.key == pygame.K_DOWN:
                action = 2
            elif event.key == pygame.K_LEFT:
                action = 3
            elif event.key == pygame.K_RIGHT:
                action = 4
            elif event.key == pygame.K_SPACE:
                action = 5
            elif event.key == pygame.K_c:
                action = 6
            elif event.key == pygame.K_r:
                action = 7

    return action, running


def main():
    env = gym.make("Tetris-v0")
    env.reset()
    running = True

    while running:
        env.render()
        action, running = handle_events()
        env.step(action)

    env.close()


if __name__ == "__main__":
    main()
