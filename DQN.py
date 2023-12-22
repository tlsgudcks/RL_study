import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import collections
import random

import DiscreateActionEnv
from Qnet import Qnet
from Replaybuffer import ReplayBuffer


class DQN:
    training_size = 10
    learning_rate = 0.0005
    gamma = 0.98
    buffer_limit = 50000
    batch_size = 32
    @classmethod
    def train(cls,q, q_target, memory, optimizer):
        for i in range(cls.training_size):
            s, a, r, s_prime, done_mask = memory.sample(cls.batch_size)

            q_out = q(s)
            q_a = q_out.gather(1, a)
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)


            target = r + cls.gamma * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @classmethod
    def main(cls):
        env = DiscreateActionEnv.GridWorld()
        q = Qnet()
        q_target = Qnet()
        q_target.load_state_dict(q.state_dict())
        memory = ReplayBuffer()
        print_interval = 20
        score = 0.0
        optimizer = optim.Adam(q.parameters(), lr=cls.learning_rate)

        for n_epi in range(10000):
            epsilon = max(0.05, 0.8 - n_epi*0.0005)
            s = env.reset()
            done = False


            while not done:
                a = q.sample_action(torch.from_numpy(s).float(), epsilon)
                s_prime, r, done = env.step(a)
                done_mask = 0.0 if done else 1.0
                memory.put((s, a, r / 100.0, s_prime, done_mask))
                s = s_prime
                score += r
                if done:
                    break
            if memory.size() > 2000:
                cls.train(q, q_target, memory, optimizer)
            #print(s)
            if n_epi % print_interval == 0 and n_epi != 0:
                q_target.load_state_dict(q.state_dict())
                print("n_episode: {}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(n_epi, score / print_interval,
                                                                                           memory.size(), epsilon * 100))
                score = 0.0
DQN.main()