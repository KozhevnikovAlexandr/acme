#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import acme
from acme import specs
import tensorflow as tf
import acme.tf.networks as networks 
from acme import wrappers
from acme.wrappers.gym_wrapper import GymWrapper, GymAtariAdapter
import acme.agents.tf.r2d2 as r2d2
from acme.agents.tf import dqn
import dm_env
import sonnet as snt
from acme.tf.networks.atari import R2D2AtariNetwork
import functools
from absl import app
from absl import flags
from examples.atari.helpers import make_environment as me
import numpy as np
import copy
import imageio 
import base64
import IPython
import matplotlib.pyplot as plt
from acme import specs


# In[9]:


level = 'TimePilot-v0'
num_episodes = 1000
num_evaluate = 10


# In[3]:


def make_environment(level, evaluation: bool = False) -> dm_env.Environment:
    env = gym.make(level, full_action_space=False)
    max_episode_len = 108000 if evaluation else 50000
    return wrappers.wrap_all(env, [
      wrappers.GymAtariAdapter,
      functools.partial(
          wrappers.AtariWrapper,
          to_float=True,
          max_episode_len=max_episode_len,
          zero_discount_on_life_loss=True,
      ),
      wrappers.SinglePrecisionWrapper,
      wrappers.ObservationActionRewardWrapper,  # Adds prev actions and rewards to observation

  ])


# In[4]:


def render(env):
    return env._physics.render(camera_id=0)


# In[5]:


def evaluate(env, agent, steps=100):
    frames = []
    score = 0
    state = env.reset()
    done = False
    for step in range(steps):
        ep_frames = []
        score = 0 
        while state.step_type != 2:
            ep_frames.append(env.environment.render(mode='rgb_array'))
            action = agent.select_action(state.observation)
            state = env.step(action)
            score += state.reward
        state = env.reset()
        frames.append([ep_frames, score])
    return frames


# In[6]:


def save_video(frames, score=0, filename=None):
    if filename is None:
        filename = r'videos/' + str(int(score)) + '.mp4'
    with imageio.get_writer(filename, fps=15) as video:
        for frame in frames:
            video.append_data(frame)


# In[7]:


def display_video(frames, score=0, filename=None):
    if filename is None:
        filename = str(score) + '.mp4'
    with imageio.get_writer(filename, fps=10) as video:
        for frame in frames:
            video.append_data(frame)
    video = open(filename, 'rb').read()
    b64_video = base64.b64encode(video)
    video_tag = ('<video  width="640" height="480" controls alt="test" '
               'src="data:video/mp4;base64,{0}">').format(b64_video.decode())
    return IPython.display.HTML(video_tag)


# In[8]:


if __name__=='__main__':
    env = make_environment(level)
    env_spec = acme.make_environment_spec(env)
    network = networks.R2D2AtariNetwork(env_spec.actions.num_values)
    agent = r2d2.R2D2(env_spec, network, burn_in_length=40, trace_length=40, replay_period=1)
    loop = acme.EnvironmentLoop(env, agent)
    loop.run(num_episodes)
    frames = evaluate(env, agent, num_evaluate)
    for ep in range(len(frames)):
        save_video(frames[ep][0], frames[ep][1])


# In[30]:


#def display_video(frames, filename='temp.mp4'):
    #with imageio.get_writer(filename, fps=30) as video:
        #for frame in frames:
            #video.append_data(frame)
  # Read video and display the video
    #video = open(filename, 'rb').read()
    #b64_video = base64.b64encode(video)
    #video_tag = ('<video  width="640" height="480" controls alt="test" '
               #'src="data:video/mp4;base64,{0}">').format(b64_video.decode())
    #return IPython.display.HTML(video_tag)

