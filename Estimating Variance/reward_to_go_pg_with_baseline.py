'''
Implementation of Reward-to-go policy gradient with baseline value function.
'''

import argparse
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.policy_net = nn.Sequential(
                         nn.Linear(input_dim, hidden_dim*2),
                         nn.ReLU(),
                         nn.Linear(hidden_dim*2, hidden_dim),
                         nn.ReLU(),
                         nn.Linear(hidden_dim, out_dim))

        self.value_net = nn.Sequential(
                         nn.Linear(input_dim, hidden_dim*2),
                         nn.ReLU(),
                         nn.Linear(hidden_dim*2, hidden_dim),
                         nn.ReLU(),
                         nn.Linear(hidden_dim, 1))
    
    def act(self, obs):
        obs = torch.tensor(obs)
        pd_params = self.policy_net(obs)
        prob_dist = torch.distributions.Categorical(logits=pd_params)
        action = prob_dist.sample()
        #calculate log of probability of taking action(a_t) by the policy(pi) given the obs(s_t)
        log_prob = prob_dist.log_prob(action)
        return action.item(), log_prob

    def compute_state_value(self, obs):
        obs = torch.tensor(obs)
        state_value = self.value_net(obs)
        return state_value


def train_agent(env, agent, policy_optim, value_optim, nb_episodes, nb_timesteps, gamma):#, seed):
    for episode in range(1, nb_episodes+1):
        obs, _ = env.reset()#seed=seed)
        rewards, log_probs, state_values = [], [], []
        for timestep in range(1, nb_timesteps+1):
            action, log_prob = agent.act(obs)
            state_value = agent.compute_state_value(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            state_values.append(state_value)
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
        state_values = torch.cat(state_values)

        #calucalte policy loss
        policy_loss = - log_probs * (returns - state_values)
        policy_loss = torch.sum(policy_loss)
        #calculate value loss
        value_loss = F.mse_loss(state_values, returns)
        #update policy network
        policy_optim.zero_grad()
        policy_loss.backward(retain_graph=True)
        policy_optim.step()
        #update value network
        value_optim.zero_grad()
        value_loss.backward()
        value_optim.step()

        if (episode % 10 == 0):
            torch.save(agent.state_dict(), f'checkpoints/agent_ckpt-ep_{episode}.pt')


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="CartPole-v1")
parser.add_argument("--nb_episodes", default=300)
parser.add_argument("--nb_timesteps", default=200)
parser.add_argument("--gamma", default=0.99)
parser.add_argument("--lr", default=1e-2)
args = parser.parse_args()

env = gym.make(args.env)

#seed = 7
#torch.manual_seed(seed)

agent = Agent(input_dim=env.observation_space.shape[0],
                hidden_dim=32, out_dim=env.action_space.n)
policy_optim = torch.optim.Adam(agent.policy_net.parameters(), lr=args.lr)
value_optim = torch.optim.Adam(agent.value_net.parameters(), lr=1e-1)#use a faster lr for the value_net
train_agent(env, agent, policy_optim, value_optim, 
            nb_episodes=args.nb_episodes, nb_timesteps=args.nb_timesteps,
            gamma=args.gamma)#, seed=seed)
