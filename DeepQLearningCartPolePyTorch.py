import gym
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
import os
import matplotlib.pyplot as plt
import gc

class QNeuralNetwork(nn.Module):
    def __init__(self, input_size, action_size):
        super(QNeuralNetwork, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(*input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
    
    def forward(self, state):
        return self.dense(state)

def PreProcess(frame):
    im = rgb2gray(frame)
    resized_frame = resize(im, [84, 84])
    return np.float32(resized_frame)

EPSILON = 1.000
EPSILON_DECAY = 0.0001
EPSILON_MIN = 0.01
GAMMA = 0.950
BATCH_SIZE = 32
LEARNING_RATE = 0.00025
MEMORY_SIZE = 5000

states, actions, rewards, next_states, terminateds = deque(maxlen=MEMORY_SIZE), deque(maxlen=MEMORY_SIZE), deque(maxlen=MEMORY_SIZE), deque(maxlen=MEMORY_SIZE), deque(maxlen=MEMORY_SIZE)

env = gym.make("CartPole-v1")

state, info = env.reset()
#state = PreProcess(state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QNeuralNetwork(state.shape, env.action_space.n).to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.MSELoss()

checkpoint_path = "training_5/model.pth"

frame_count = 0
total_reward = 0
Total_episodes = 0
TimesObjectiveReached = 0

IsTesting = False

while True:
    frame_count += 1
    randNum = np.random.rand()
    action = 0
    if randNum > EPSILON or IsTesting:
        q_values = None
        with torch.no_grad():
            q_values = model(torch.tensor(np.array([state])).unsqueeze(0).to(device))
        action = torch.argmax(q_values).item()
        del q_values
    else:
        action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)
    #next_state = PreProcess(next_state)
    states.append(state)
    actions.append(action)
    rewards.append(reward)
    next_states.append(next_state)
    terminateds.append(terminated)
    state = np.copy(next_state)
    total_reward += reward
    if total_reward >= 200 and not IsTesting:
        print("Modelo chegou ao objetivo!")
        TimesObjectiveReached += 1
        if TimesObjectiveReached >= 3:
            torch.save(model, checkpoint_path)
            break

    if len(states) > BATCH_SIZE and not IsTesting and frame_count % 1 == 0:
        batch_index = np.random.choice(len(states), BATCH_SIZE)
        batch_states = np.array([states[i] for i in batch_index])
        batch_actions = np.array([actions[i] for i in batch_index])
        batch_rewards = np.array([rewards[i] for i in batch_index])
        batch_next_states = np.array([next_states[i] for i in batch_index])
        batch_terminateds = np.array([terminateds[i] for i in batch_index])
        batch_states = torch.tensor(batch_states).to(device)
        #batch_actions = torch.tensor(batch_actions).to(device)
        #batch_rewards = torch.tensor(batch_rewards).to(device)
        batch_next_states = torch.tensor(batch_next_states).to(device)
        #batch_terminateds = torch.tensor(batch_terminateds).to(device)
        q_values = None
        q_values_next = None
        q_values = model(batch_states.unsqueeze(0)).squeeze(0)
        q_values_next = model(batch_next_states.unsqueeze(0)).squeeze(0)
        batch_index = np.arange(BATCH_SIZE, dtype=np.int32)
        q_target = q_values.detach().clone().cpu().numpy()
        #batch_actions = batch_actions.detach().cpu().numpy()
        #batch_rewards = batch_rewards.detach().cpu().numpy()
        q_values_next = q_values_next.detach().cpu().numpy()
        #batch_terminateds = batch_terminateds.detach().cpu().numpy()
        q_target[batch_index, batch_actions] = batch_rewards + GAMMA * np.max(q_values_next, axis=1) * (1 - batch_terminateds)
        loss = loss_function(q_values, torch.tensor(q_target).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del batch_states
        del batch_actions
        del batch_rewards
        del batch_next_states
        del batch_terminateds
        del q_values
        del q_values_next
        del q_target
        del loss
        # "Solve" memory leak
        #gc.collect()

    if EPSILON > EPSILON_MIN:
        EPSILON -= EPSILON_DECAY
    if terminated or truncated:
        Total_episodes += 1
        print(f"Reward: {total_reward}, Total episodes: {Total_episodes}, Epsilon: {EPSILON}")
        total_reward = 0
        observation, info = env.reset()

env.close()