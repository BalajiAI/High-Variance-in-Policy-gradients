'''
This is an implementation of REINFORCE which makes use of multiple trajectories
to estimate the gradient.
'''

import torch
import torch.nn as nn


class Agent(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.policy = nn.Sequential(
                         nn.Linear(input_dim, hidden_dim*2),
                         nn.ReLU(),
                         nn.Linear(hidden_dim*2, hidden_dim),
                         nn.ReLU(),
                         nn.Linear(hidden_dim, out_dim))
        
    def forward(self, x):
        x = self.policy(x)
        return x
    
    def act(self, obs):
        obs = torch.tensor(obs)
        pd_params = self.forward(obs)
        prob_dist = torch.distributions.Categorical(logits=pd_params)
        action = prob_dist.sample()
        #calculate log of probability of taking action(a_t) by the policy(pi) given the obs(s_t)
        log_prob = prob_dist.log_prob(action)
        return action.item(), log_prob


def train_agent(env, agent, optim, nb_episodes, nb_timesteps, gamma, nb_episodes_till_update):
    for episode in range(1, nb_episodes+1):
        obs, _ = env.reset()
        rewards, log_probs = [], []
        for timestep in range(1, nb_timesteps+1):
            action, log_prob = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            if terminated or truncated:
                break

        #"Reward-to-go policy gradient"
        #calculate return at each time step efficiently by using dynamic programming
        returns = []
        future_return = 0.0
        for t in reversed(range(len(rewards))):
            #R[t] = r[t] + gamma * R[t+1]
            future_return = rewards[t] + gamma * future_return
            returns.append(future_return)
        returns.reverse() #Now, the returns are indexed from 0 to nb_timesteps
        log_probs = torch.stack(log_probs)
        returns = torch.tensor(returns)        
        loss = - log_probs * returns
        loss = torch.sum(loss)
        loss_till_update += loss
        
        print(f"Episode {episode} : Total reward - {sum(rewards)}")
        
        if (episode % nb_episodes_till_update == 0):
            optim.zero_grad()
            loss_till_update.backward()
            optim.step()
            loss_till_update = 0
            print(f"Policy has updated!\n")



import argparse
import gymnasium as gym

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="CartPole-v1")
parser.add_argument("--nb_episodes", default=800)
parser.add_argument("--nb_timesteps", default=200)
parser.add_argument("--nb_episodes_till_update", default=10,
                    help="Defines number of trajectories to collect before making the policy update")
parser.add_argument("--gamma", default=1.0)
parser.add_argument("--lr", default=1e-2)
args = parser.parse_args()

env = gym.make(args.env)
agent = Agent(input_dim=env.observation_space.shape[0],
                hidden_dim=32, out_dim=env.action_space.n)
optim = torch.optim.Adam(agent.parameters(), lr=args.lr)
train_agent(env, agent, optim, nb_episodes=args.nb_episodes,
            nb_timesteps=args.nb_timesteps, gamma=args.gamma,
            nb_episodes_till_update=args.nb_episodes_till_update)

