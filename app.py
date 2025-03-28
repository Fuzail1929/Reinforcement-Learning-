import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Hyperparameters
ENV_NAME = "CartPole-v1"
HIDDEN_SIZE = 128
LR = 3e-4
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPSILON = 0.2
EPOCHS = 4
BATCH_SIZE = 64
MAX_EPISODES = 500
PRINT_INTERVAL = 20

# Create environment
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# PPO Network
class PPOModel(nn.Module):
    def __init__(self):
        super(PPOModel, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

# PPO Agent
class PPOAgent:
    def __init__(self):
        self.model = PPOModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = deque(maxlen=BATCH_SIZE)
        
    def get_action(self, state):
        state = torch.FloatTensor(state)
        probs, value = self.model(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value
        
    def update(self):
        states, actions, old_log_probs, returns, advantages = self._process_memory()
        
        for _ in range(EPOCHS):
            for idx in range(0, len(states), BATCH_SIZE):
                batch_states = states[idx:idx+BATCH_SIZE]
                batch_actions = actions[idx:idx+BATCH_SIZE]
                batch_old_log_probs = old_log_probs[idx:idx+BATCH_SIZE]
                batch_returns = returns[idx:idx+BATCH_SIZE]
                batch_advantages = advantages[idx:idx+BATCH_SIZE]
                
                probs, values = self.model(batch_states)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * batch_advantages
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                
                critic_loss = (batch_returns - values.squeeze()).pow(2).mean()
                
                loss = actor_loss + 0.5 * critic_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        self.memory.clear()
    
    def _process_memory(self):
        states = torch.stack([m[0] for m in self.memory])
        actions = torch.tensor([m[1] for m in self.memory])
        old_log_probs = torch.tensor([m[2] for m in self.memory])
        returns = torch.tensor([m[3] for m in self.memory])
        advantages = torch.tensor([m[4] for m in self.memory])
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return states, actions, old_log_probs, returns, advantages

# Training
agent = PPOAgent()
total_rewards = []
average_rewards = []

for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    states = []
    actions = []
    old_log_probs = []
    rewards = []
    values = []
    
    while True:
        action, log_prob, value = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        states.append(torch.FloatTensor(state))
        actions.append(action)
        old_log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value.squeeze())
        
        state = next_state
        episode_reward += reward
        
        if done:
            # Calculate returns and advantages
            returns = []
            advantages = []
            R = 0
            next_value = 0
            
            for r in reversed(rewards):
                R = r + GAMMA * R
                returns.insert(0, R)
            
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            values = values + [next_value]
            for i in range(len(rewards)):
                td_error = rewards[i] + GAMMA * values[i+1] - values[i]
                advantages.append(td_error + (GAMMA * LAMBDA) * (advantages[-1] if i < len(rewards)-1 else 0))
                
            advantages = torch.tensor(advantages[::-1])
            
            # Store in memory
            for i in range(len(states)):
                agent.memory.append((
                    states[i],
                    actions[i],
                    old_log_probs[i],
                    returns[i],
                    advantages[i]
                ))
            
            agent.update()
            total_rewards.append(episode_reward)
            break
    
    # Print progress
    if (episode + 1) % PRINT_INTERVAL == 0:
        avg_reward = np.mean(total_rewards[-PRINT_INTERVAL:])
        average_rewards.append(avg_reward)
        print(f"Episode {episode+1}, Average Reward: {avg_reward:.1f}")

# Plot training results
plt.plot(average_rewards)
plt.xlabel("Episode Batch")
plt.ylabel("Average Reward")
plt.title("Training Progress")
plt.show()

# Evaluation
test_episodes = 100
test_rewards = []

with torch.no_grad():
    for _ in range(test_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            action, _, _ = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                test_rewards.append(episode_reward)
                break

avg_reward = np.mean(test_rewards)
std_dev = np.std(test_rewards)
print(f"\nEvaluation Results ({test_episodes} episodes):")
print(f"Average Reward: {avg_reward:.1f}")
print(f"Standard Deviation: {std_dev:.1f}")

# Save model
torch.save(agent.model.state_dict(), "ppo_cartpole.pth")
