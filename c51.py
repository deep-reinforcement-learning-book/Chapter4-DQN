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
vrange = tf.reshape(tf.linspace(min_value, max_value, atom_num), [1, atom_num])
vrange = tf.cast(vrange, tf.float32)
vrange_broadcast = tf.tile(vrange, tf.constant([action_dim, 1]))
deltaz = (max_value - min_value) / (atom_num - 1)


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
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.fc2 = tf.keras.layers.Dense(action_dim * atom_num,
                                         activation='linear')

    def call(self, pixels, **kwargs):
        # scale observation
        pixels = tf.divide(tf.cast(pixels, tf.float32), tf.constant(255.0))
        # extract features by convolutional layers
        feature = self.flat(self.conv3(self.conv2(self.conv1(pixels))))
        # calculate q-value
        qvalue = self.fc2(self.fc1(feature))

        return tf.keras.activations.softmax(
            tf.reshape(qvalue, [-1, action_dim, atom_num]), axis=2)


# ###############################  DQN  #####################################
class DQN(object):
    def __init__(self):
        self.qnet = QFunc('q')
        self.targetqnet = QFunc('targetq')
        sync(self.qnet, self.targetqnet)
        self.niter = 0
        self.optimizer = tf.optimizers.Adam(lr, clipnorm=clipnorm,
                                            epsilon=0.01 / batch_size)

    def get_action(self, obv):
        eps = epsilon(self.niter)
        if random.random() < eps:
            return int(random.random() * action_dim)
        else:
            obv = np.expand_dims(obv, 0).astype('float32')
            dist = self._qvalues_func(obv)
            qvalue = tf.reduce_sum(dist * vrange_broadcast, axis=2)
            return qvalue.numpy().argmax(1)[0]

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
            kl_divergence = self._kl_divergence_func(b_o, b_a, b_r, b_o_, b_d)
            loss = tf.reduce_mean(kl_divergence)

        grad = tape.gradient(loss, self.qnet.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.qnet.trainable_weights))

        return kl_divergence

    @tf.function
    def _kl_divergence_func(self, b_o, b_a, b_r, b_o_, b_d):
        b_r = tf.tile(
            tf.reshape(b_r, [-1, 1]),
            tf.constant([1, atom_num])
        )  # batch_size * atom_num
        b_d = tf.tile(
            tf.reshape(b_d, [-1, 1]),
            tf.constant([1, atom_num])
        )

        z = b_r + (1 - b_d) * reward_gamma * vrange
        z = tf.clip_by_value(z, min_value, max_value)
        b = (z - min_value) / deltaz
        index_help = tf.expand_dims(tf.tile(
            tf.reshape(tf.range(batch_size), [batch_size, 1]),
            tf.constant([1, atom_num])
        ), -1)

        b_u = tf.cast(tf.math.ceil(b), tf.int32)  # upper
        b_uid = tf.concat([index_help, tf.expand_dims(b_u, -1)], 2)  # indexes
        b_l = tf.cast(tf.math.floor(b), tf.int32)
        b_lid = tf.concat([index_help, tf.expand_dims(b_l, -1)], 2)  # indexes

        b_dist_ = self.targetqnet(b_o_)  # whole distribution
        b_q_ = tf.reduce_sum(b_dist_ * vrange_broadcast, axis=2)
        b_a_ = tf.cast(tf.argmax(b_q_, 1), tf.int32)
        b_adist_ = tf.gather_nd(  # distribution of b_a_
            b_dist_,
            tf.concat([tf.reshape(tf.range(batch_size), [-1, 1]),
                       tf.reshape(b_a_, [-1, 1])], axis=1)
        )
        b_adist = tf.gather_nd(  # distribution of b_a
            self.qnet(b_o),
            tf.concat([tf.reshape(tf.range(batch_size), [-1, 1]),
                       tf.reshape(b_a, [-1, 1])], axis=1)
        ) + 1e-8

        b_l = tf.cast(b_l, tf.float32)
        mu = b_adist_ * (b - b_l) * tf.math.log(tf.gather_nd(b_adist, b_uid))
        b_u = tf.cast(b_u, tf.float32)
        ml = b_adist_ * (b_u - b) * tf.math.log(tf.gather_nd(b_adist, b_lid))
        kl_divergence = tf.negative(tf.reduce_sum(mu + ml, axis=1))

        return kl_divergence


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
