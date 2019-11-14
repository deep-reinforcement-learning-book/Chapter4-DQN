import argparse
import time

from wrappers import build_env
from config import *
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--seed', help='random seed', type=int, default=0)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)  # reproducible

env = build_env(env_id, seed=args.seed)
in_dim = env.observation_space.shape
action_dim = env.action_space.n


# ##############################  Network  ####################################
class QFunc(tf.keras.Model):

    def __init__(self, name):
        super(QFunc, self).__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(
            32, kernel_size=(8, 8), strides=(4, 4),
            padding='valid', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(
            64, kernel_size=(4, 4), strides=(2, 2),
            padding='valid', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(
            64, kernel_size=(3, 3), strides=(1, 1),
            padding='valid', activation='relu')
        self.flat = tf.keras.layers.Flatten()
        self.fc1q = tf.keras.layers.Dense(512, activation='relu')
        self.fc2q = tf.keras.layers.Dense(action_dim, activation='linear')
        self.fc1v = tf.keras.layers.Dense(512, activation='relu')
        self.fc2v = tf.keras.layers.Dense(1, activation='linear')

    def call(self, pixels, **kwargs):
        # scale observation
        pixels = tf.divide(tf.cast(pixels, tf.float32), tf.constant(255.0))
        # extract features by convolutional layers
        feature = self.flat(self.conv3(self.conv2(self.conv1(pixels))))
        # calculate q-value
        qvalue = self.fc2q(self.fc1q(feature))
        svalue = self.fc2v(self.fc1v(feature))

        return svalue + qvalue - tf.reduce_mean(qvalue, 1, keepdims=True)


# ###############################  DQN  #####################################
class DQN(object):
    def __init__(self):
        self.qnet = QFunc('q')
        self.targetqnet = QFunc('targetq')
        sync(self.qnet, self.targetqnet)
        self.niter = 0
        self.optimizer = tf.optimizers.Adam(lr, epsilon=1e-5, clipnorm=clipnorm)

    def get_action(self, obv):
        eps = epsilon(self.niter)
        if random.random() < eps:
            return int(random.random() * action_dim)
        else:
            obv = np.expand_dims(obv, 0).astype('float32')
            return self._qvalues_func(obv).numpy().argmax(1)[0]

    @tf.function
    def _qvalues_func(self, obv):
        return self.qnet(obv)

    def train(self, b_o, b_a, b_r, b_o_, b_d):
        self._train_func(b_o, b_a, b_r, b_o_, b_d)

        self.niter += 1
        if self.niter % target_q_update_freq == 0:
            sync(self.qnet, self.targetqnet)

    @tf.function
    def _train_func(self, b_o, b_a, b_r, b_o_, b_d):
        with tf.GradientTape() as tape:
            td_errors = self._tderror_func(b_o, b_a, b_r, b_o_, b_d)
            loss = tf.reduce_mean(huber_loss(td_errors))

        grad = tape.gradient(loss, self.qnet.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.qnet.trainable_weights))

        return td_errors

    @tf.function
    def _tderror_func(self, b_o, b_a, b_r, b_o_, b_d):
        b_q_ = (1 - b_d) * tf.reduce_max(self.targetqnet(b_o_), 1)
        b_q = tf.reduce_sum(self.qnet(b_o) * tf.one_hot(b_a, action_dim), 1)

        return b_q - (b_r + reward_gamma * b_q_)


# #############################  Trainer  ###################################
if __name__ == '__main__':
    dqn = DQN()
    buffer = ReplayBuffer(buffer_size)

    o = env.reset()
    nepisode = 0
    t = time.time()
    for i in range(1, number_timesteps + 1):
        a = dqn.get_action(o)

        # execute action and feed to replay buffer
        # note that `_` tail in var name means next
        o_, r, done, info = env.step(a)
        buffer.add(o, a, r, o_, done)

        if i >= warm_start and i % train_freq == 0:
            transitions = buffer.sample(batch_size)
            dqn.train(*transitions)

        if done:
            o = env.reset()
        else:
            o = o_

        # episode in info is real (unwrapped) message
        if info.get('episode'):
            nepisode += 1
            reward, length = info['episode']['r'], info['episode']['l']
            print(
                'Time steps so far: {}, episode so far: {}, '
                'episode reward: {:.4f}, episode length: {}'
                .format(i, nepisode, reward, length)
            )
