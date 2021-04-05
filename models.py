import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class DQNbn(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQNbn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)


class DQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)
    
    
class DuelingDQN(nn.Module):
    def __init__(self, inputs_shape=4, n_actions=14):
        super(DuelingDQN, self).__init__()
        self.input_shape = inputs_shape
        self.n_actions = n_actions
        self.features = nn.Sequential(
            nn.Conv2d(inputs_shape, 16, kernel_size=8, stride=4),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.MaxPool2d(2)
        )
        # self.hidden = nn.Sequential(
        #     nn.Linear(self.features_size(), 256, bias=True),
        #     nn.ReLU()
        # )
        self.adv = nn.Linear(256, n_actions, bias=True)
        self.val = nn.Linear(256, 1, bias=True)

    def forward(self, x):
        x = x.float() / 255
        x = self.features(x)
        x = x.view(x.size(0), -1)
        #x = self.hidden(x)
        x = F.relu(nn.Linear(x.size(-1), 256, bias=True)(x))
        adv = self.adv(x)
        val = self.val(x).expand(x.size(0), self.n_actions)
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.n_actions)
        return x

    # def features_size(self):
    #     return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
