import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from torch.distributions import Categorical

import DiscreateActionEnv

learning_rate = 0.0001
gamma = 0.98


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(18, 64)
        self.fc2 = nn.Linear(64, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -R * torch.log(prob)
            loss.backward()
        self.optimizer.step()
        self.data = []


def main():
    env = DiscreateActionEnv.GridWorld()
    pi = Policy()
    print_interval = 20
    score = 0.0

    for n_epi in range(10000):
        s = env.reset()
        done = False

        while not done:
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done = env.step(a.item())
            pi.put_data((r, prob[a]))

            s = s_prime
            score += r
        pi.train_net()
        if n_epi % print_interval == 0 and n_epi != 0:
            print("of episode: {}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0


main()