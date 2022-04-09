import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T


class DeepQNetwork(nn.Module):
    def __init__(self, state_size, fc1_dims, fc2_dims, action_size):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.head = nn.Linear(fc2_dims, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.head(x)
