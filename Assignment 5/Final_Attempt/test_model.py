import flappy_bird_gymnasium
import gymnasium as gym
import matplotlib.pyplot as plt
import time
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms

import random
import numpy as np
from collections import deque

from flappyqnetwork import FlappyQNetwork
from humanRendering import HumanRendering

OUTLINE_COLOR = torch.tensor([84,56,71])
def preprocess_state_black_n_white(state):
    state = state[0:state.shape[0]-110,0:state.shape[1]]

    # Removing all but outline
    state[state == OUTLINE_COLOR] = 255 # making outline white
    state[state != 255] = 0             # making everything but outline black


    #resized = cv2.resize(np.array(output),(84,84))
    state = cv2.resize(np.array(state),(84,84))
    state = state[:,:,1]
    normalized = state / 255.0

    return normalized
def stack_frames(frame, stacked_frames, new_epoch):

    if new_epoch:
        stacked_frames = np.stack([frame]*4,axis=0)
    else:
        stacked_frames = np.concatenate((stacked_frames[1:, :, :], np.expand_dims(frame, 0)), axis=0)
    
    return stacked_frames

def test():
    env = gym.make('FlappyBird-v0',render_mode="rgb_array")
    wrapper = HumanRendering(env)
    # constants 

    STATE_SHAPE = (4,84,84)
    NR_ACTIONS = env.action_space.n
    INITIAL_EPSILON = 0.1
    SKIP_FRAMES = 4

    epsilon = INITIAL_EPSILON

    q_network = FlappyQNetwork(STATE_SHAPE,NR_ACTIONS)
    q_network.load_state_dict(torch.load('weights/best_weights.pth'))
    epoch = 0
    while True:
        env.reset()
        wrapper = HumanRendering(env)
        wrapper.reset()
        state = preprocess_state_black_n_white(env.render())

        stacked_frames = stack_frames(state,None,True)
        
        epoch_reward = 0
        done = False
        pipes = 0
        while not done:
            wrapper._render_frame()
            with torch.no_grad():
                    state_tensor = torch.tensor(stacked_frames,dtype=torch.float32).unsqueeze(0)
                    action  = q_network(state_tensor).argmax().item()

            _, reward, done, _, _ = env.step(action)
            pipes += reward == 1
            '''
            if action == 1:              
                if not done:
                    for _ in range(SKIP_FRAMES):
                        _, frame_reward, frame_done, _, _ = env.step(0)
                        pipes += frame_reward == 1
                        wrapper._render_frame()
                        reward += frame_reward
                        done = frame_done
                        if done:
                            break
            '''

            next_state = preprocess_state_black_n_white(env.render())
            next_stacked_frames = stack_frames(next_state,stacked_frames,False)
            stacked_frames = next_stacked_frames

                
            epoch_reward += reward
        print(f'Epoch: {epoch}, Epoch Reward: {epoch_reward}, Epsilon: {epsilon}, Passed {pipes} pipes')
        epoch += 1

        wrapper.close()
        env.close()
        pass
test()