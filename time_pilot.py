#!/usr/bin/env python
# coding: utf-8

# In[1]:

import gym
import acme
import acme.tf.networks as networks 
from acme import wrappers
import acme.agents.tf.r2d2 as r2d2
import dm_env
import functools
import imageio 
import base64
import IPython


# In[9]:


level = 'TimePilot-v0'
num_episodes = 10
num_evaluate = 10


# In[3]:


def make_environment(level) -> dm_env.Environment:
    env = gym.make(level)
    max_episode_len = 10_000
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


def save_video(frames, score=0, iter=0, filename=None):
    if filename is None:
        filename = r'videos/{0}_{1}.mp4'.format(iter, score)
    with imageio.get_writer(filename, fps=15) as video:
        for frame in frames:
            video.append_data(frame)


# In[7]:


def display_video(frames, score=0, iter=0, filename=None):
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
    for i in range(100):
        print(i)
        loop.run(1)
    #loop.run(num_episodes)
        if i % 25 == 0:
            frames = evaluate(env, agent, 5)
            for ep in range(len(frames)):
                save_video(frames[ep][0], frames[ep][1], i)


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

