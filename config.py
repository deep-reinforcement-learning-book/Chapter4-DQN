# configs

# common config
env_id = 'BreakoutNoFrameskip-v4'
number_timesteps = int(1e7)  # total number of time steps to train on
explore_frac = 0.1
# epsilon-greedy schedule, final exploit prob is 0.99
epsilon = lambda i: 1 - 0.99 * min(1, i / (number_timesteps * explore_frac))
lr = 1e-4  # learning rate
buffer_size = 100000  # replay buffer size
target_q_update_freq = 1000  # how frequency target q net update
train_freq = 4
clipnorm = 10
reward_gamma = 0.99  # reward discount
batch_size = 32  # batch size for sampling from replay buffer
warm_start = 10000  # sample times before learning

# config for per
prioritized_alpha = 0.6  # alpha in PER
prioritized_beta0 = 0.4  # initial beta in PER

# config for c51
atom_num = 51
min_value = -1.0  # allow some approximation error
max_value = 19.0
