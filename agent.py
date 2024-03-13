import pickle
import tensorflow as tf
from tensorflow.keras.losses import MSE
import numpy as np
import random
from collections import deque, namedtuple
import gymnasium as gym
from model import DQN

class dqn_agent():
  '''
  The agent handles all interaction with the environment and contains the online and target network
  '''
  def __init__(self, env_name, erp_size, tau, gamma, learning_rate, double_q, eps_min=0.01, eps_decay=0.99):
    self.env = gym.make(env_name)
    self.num_actions = self.env.action_space.n
    self.erp = deque(maxlen=erp_size)
    self.tau = tau
    self.gamma = gamma
    self.epsilon = 1.0
    self.eps_min = eps_min
    self.eps_decay = eps_decay
    self.double_q = double_q
    self.online_network = DQN(self.num_actions, learning_rate)
    self.target_network = DQN(self.num_actions, learning_rate)

    self.online_network(tf.random.uniform(shape=(1,8)))
    self.target_network(tf.random.uniform(shape=(1,8)))

  def get_experiences(self):
    '''sample experiences from the ERP convert them to a tensor and return a tuple of tensors'''
    experiences = random.sample(self.erp, k=64)
    states = tf.convert_to_tensor(np.array([e.state for e in experiences if e is not None]),dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array([e.action for e in experiences if e is not None]), dtype=tf.float32)
    rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences if e is not None]),dtype=tf.float32)
    done_vals = tf.convert_to_tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
                                     dtype=tf.float32)
    return (states, actions, rewards, next_states, done_vals)

  def get_action(self, q_values, epsilon=0):
    '''epsilon greedy policy: take random action with probability of epsilon, else take best action'''
    if random.random() > epsilon:
      return np.argmax(q_values.numpy()[0])
    else:
      return random.choice(np.arange(4))

  def compute_loss(self, experiences):
    '''compute loss obetween q_target and q_pred'''
    states, actions, rewards, next_states, done_vals = experiences
    if self.double_q: #DDQN target calculation
      #get prediction for best_action in next state from online network
      best_actions = tf.argmax(self.online_network(next_states), axis=1)
      #predict q_vals for next state with target network
      q_vals = self.target_network(next_states)
      #take q value of the best action chosen from the online network
      eval_q = tf.gather_nd(q_vals, tf.stack([tf.range(best_actions.shape[0]), tf.cast(best_actions, tf.int32)], axis=1))
      #calculate targets with bellman equation
      y_targets = rewards + (self.gamma * eval_q * (1-done_vals))
    else: #DQN target calculation
      #take action with the highest q values from target network
      max_qsa = tf.reduce_max(self.target_network(next_states), axis=-1)
      #calculate targets with bellman equation
      y_targets = rewards + (self.gamma * max_qsa * (1-done_vals))
    #get q value prediction from online network and take q value of action that was chosen
    q_values = self.online_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]), tf.cast(actions, tf.int32)], axis=1))
    #calculate loss with MSE 
    loss = MSE(y_targets, q_values)

    return loss

  @tf.function
  def agent_learn(self, experiences):
    '''calculate loss and apply gradients to online network'''
    with tf.GradientTape() as tape:
      loss = self.compute_loss(experiences)
    gradients = tape.gradient(loss, self.online_network.trainable_variables)
    self.online_network.optimizer.apply_gradients(zip(gradients, self.online_network.trainable_variables))
    self.polyak_average()

  def polyak_average(self):
    '''soft update of target network with polyak averaging'''
    for target_weights, q_network_weights in zip(self.target_network.weights, self.online_network.weights):
      target_weights.assign((1-self.tau) * q_network_weights + self.tau * target_weights)

  def check_update_conditions(self, j, steps_per_update):
    '''check for enough samples in buffer and whether enough timesteps have passed for the agent to learn'''
    if(j+1) % steps_per_update == 0 and len(self.erp) > 64:
      return True
    else:
      return False

  def get_new_epsilon(self):
    '''epsilon decay'''
    return max(self.eps_min, self.eps_decay * self.epsilon)

  def train(self, model_path, reward_path, num_epochs=2000, max_steps_per_iter=1000, erp_size=100000, steps_per_update=4):
    '''train loop for the agent'''

    #initialize variables
    self.target_network.set_weights(self.online_network.get_weights())
    experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    points_history = []
    solved = False

    #one epoch equals one game in the environment
    for i in range(num_epochs):

      state, _ = self.env.reset()
      total_points = 0

      #in game steps
      for j in range(max_steps_per_iter):

        #get action from epsilon greedy policy
        state_qn = np.expand_dims(state, axis=0)
        q_values = self.online_network(state_qn)
        action = self.get_action(q_values, self.epsilon)
        #step with chosen action
        next_state, reward, done, _, _ = self.env.step(action)
        #add experience to ERP
        self.erp.append(experience(state, action, reward, next_state, done))

        update = self.check_update_conditions(j, steps_per_update)

        if update:
          #get experiences and do train step for experiences in ERP
          experiences = self.get_experiences()
          self.agent_learn(experiences)

        state = next_state.copy()
        total_points += reward

        if done:
          break

      points_history.append(total_points)
      avg_points = np.mean(points_history[-100:])

      self.epsilon = self.get_new_epsilon()

      print(f"\rEpisode {i+1} | Total point average of the last {100} episodes: {avg_points:.2f}", end="")

      if (i+1) % 100 == 0:
          print(f"\rEpisode {i+1} | Total point average of the last {100} episodes: {avg_points:.2f}")
          self.online_network.save_weights(model_path + str(i+1))

      #env is solved if last 100 episodes had an average reward >= 200
      if not solved:
        if(avg_points >= 200):
          print(f"Environment solved in {i+1} episodes!")
          self.online_network.save_weights(model_path + '_200')
          solved = True

      #stop train if agent reaches avg reward of min. 250
      if(avg_points >= 250):
        self.online_network.save_weights(model_path + '_250')
        if self.double_q:
          with open(reward_path + 'double_dqn_rewards.pkl', 'wb') as f:
            pickle.dump(points_history, f)
        else:
          with open(reward_path + 'dqn_rewards.pkl', 'wb') as f:  
            pickle.dump(points_history, f)
        break

