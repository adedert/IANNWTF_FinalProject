import tensorflow as tf

class DQN(tf.keras.Model):
  '''
  Simple feed forward model which is used as online and target network
  '''
  def __init__(self, num_actions, lr=0.001):
    super().__init__()

    self.fc1 = tf.keras.layers.Dense(64, activation="relu")
    self.fc2 = tf.keras.layers.Dense(64, activation="relu")
    self.out = tf.keras.layers.Dense(num_actions)

    self.metrics_list = [tf.keras.metrics.Mean(name="loss")]
    self.lr = lr

    self.optimizer = tf.keras.optimizers.Adam(self.lr)

  @property
  def metrics(self):
      return self.metrics_list

  def reset_metrics(self):
      for metric in self.metrics:
          metric.reset_state()

  def call(self, input):
    x = self.fc1(input)
    x = self.fc2(x)
    x = self.out(x)
    return x