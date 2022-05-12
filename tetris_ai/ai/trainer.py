from collections import OrderedDict
from typing import Dict, List, Tuple

import gym
import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

import tetris_ai.envs
from tetris_ai.ai.agent import Agent
from tetris_ai.ai.dataset import RLDataset
from tetris_ai.ai.memory import ReplayBuffer
from tetris_ai.ai.model import DQN


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(
        self,
        batch_size: int = 32,
        lr: float = 1e-4,
        env: str = "Tetris-v0",
        gamma: float = 0.99,
        sync_rate: int = 10,
        replay_size: int = 100_000,
        warm_start_steps: int = 1000,
        eps_last_frame: int = 600,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        episode_length: int = 200,
    ) -> None:
        """
        Args:
            batch_size: size of the batches")
            lr: learning rate
            env: gym environment tag
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_steps: how many samples do we use to fill our buffer at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
        """
        super().__init__()
        self.save_hyperparameters()

        self.env = gym.make(self.hparams.env)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = DQN(obs_size, n_actions)
        self.target_net = DQN(obs_size, n_actions)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.
        Args:
            steps: number of random steps to populate the buffer with
        """
        for _ in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.
        Args:
            x: environment state
        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.
        Args:
            batch: current mini batch of replay data
        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = (
            self.net(states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        )

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = self.get_epsilon(
            self.hparams.eps_start, self.hparams.eps_end, self.hparams.eps_last_frame
        )
        self.log("epsilon", epsilon)

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward
        self.log("episode reward", self.episode_reward)

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "train_loss": loss,
        }
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
        }

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Evaluate the agent."""
        total_rewards = self.run_n_episodes(n_epsiodes=1)
        avg_rewards = sum(total_rewards) / len(total_rewards)
        return {"avg_rewards": avg_rewards}

    def run_n_episodes(self, n_epsiodes: int = 1, epsilon: float = 1.0) -> List[int]:
        """Carries out N episodes of the environment with the current agent.
        Args:
            n_epsiodes: number of episodes to run
            epsilon: epsilon value for DQN agent
        """
        total_rewards = []

        for _ in range(n_epsiodes):
            print("1")
            self.agent.reset()
            done = False
            episode_reward = 0

            while not done:
                reward, done = self.agent.play_step(
                    self.net, epsilon=epsilon, render=True
                )
                episode_reward += reward

            total_rewards.append(episode_reward)

        return total_rewards

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def test_dataloader(self) -> DataLoader:
        """Get test loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"
