import tensorflow as tf
import gymnasium as gym
import keras
import numpy as np
from model import DQN
from agent import dqn_agent

def load_model():
    model = DQN(4)
    model(tf.random.uniform(shape=(1,8)))
    user_input = input("Do you want to use DQN or DDQN? Enter 1 for DQN and 2 for DDQN:")
    if user_input == 1:
        print('Chose DQN')
        model.load_weights('models/checkpoints/dqn_checkpoint_250')
    else: 
        print('Chose DDQN')
        model.load_weights('models/checkpoints/double_dqn_checkpoint_250')
    return model

def run_env(model):
    env = gym.make("LunarLander-v2", render_mode="human")
    score = 0 
    observation, _ = env.reset()

    episode_over = False
    #while not episode_over:
    for _ in range(3000):
        q_values = model(tf.reshape(observation, shape=(1,8)))
        greedy_action = tf.argmax(q_values, axis=1) # agent policy that uses the observation and info
        observation, reward, terminated, truncated, _ = env.step(int(greedy_action))
        score += reward

        if terminated or truncated:
            observation, _ = env.reset()
            print(f'Reward: {score}')
            score = 0

    env.close()


if __name__ == "__main__":
    model = load_model()
    run_env(model)

