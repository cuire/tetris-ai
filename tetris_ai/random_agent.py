import gym

import tetris_ai.envs

env = gym.make("Tetris-v0")


def main():
    for _ in range(3):
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(env.action_space.sample())
            env.render()


if __name__ == "__main__":
    main()
