#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_belief(belief):
    
    plt.figure()
    
    ax = plt.subplot(2,1,1)
    ax.matshow(belief.reshape(1, belief.shape[0]))
    ax.set_xticks(np.arange(0, belief.shape[0],1))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks([])
    ax.title.set_text("Grid")
    
    ax = plt.subplot(2, 1, 2)
    ax.bar(np.arange(0, belief.shape[0]), belief)
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.set_ylim([0, 1.05])
    ax.title.set_text("Histogram")

'''
    * with probability 0.7, it moves in the correct direction (i.e.  𝐹→𝐹,𝐵→𝐵 );
    * with probability 0.2 or if the command cannot be exectuted (e.g. end of the world!), it does not move;
    * with probability 0.1, it moves in the opposite direction (i.e.  𝐹→𝐵,𝐵→𝐹 ).
'''
def motion_model(action, belief):
    n = belief.shape[0]
    for idx in range(1, n - 1):
        belief[idx] = 0.7 * belief[idx - 1] + 0.2 * belief[idx] + 0.1 * belief[idx + 1]
    if action == 'F':
        belief[0] = 0.1 * belief[1] + 0.2 * belief[0]
        belief[-1] = 0.2 * belief[-1]
    if action == 'B':
        belief[-1] = 0.7 * belief[-2] + 0.2 * belief[-1]
        belief[0] = 0.2 * belief[0]
    return belief / np.sum(belief)

'''
    * tile is white with probability 0.7
    * tile is black with probability 0.9
'''
def sensor_model(observation, belief, world):
    n = belief.shape[0]
    for idx in range(n):
        if observation == world[idx]:
            belief[idx] *= 0.7 if observation == 1 else 0.9
    return belief

def recursive_bayes_filter(actions, observations, belief, world):
    n = len(actions)
    assert(n == len(observations), 'n != len(observations)')
    
    for idx in range(n):
        belief = motion_model(actions[idx], belief)
        belief = sensor_model(observations[idx], belief, world)
    belief *= 1.0 / np.sum(belief)
    return belief

