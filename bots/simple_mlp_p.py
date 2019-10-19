import tensorflow.compat.v1 as tf
import numpy as np
import logging
import os.path

tf.disable_eager_execution()

class Env:
	def __init__(self):
		self.observation_space = 15
		self.action_space = 7

	def reset(self):
		# 	Retreive first observation


		return np.random.rand(15)
	def compute_reward(self, state):

	def step(self, action):
		# 	Send action to server
		#	Retreive new observation
		#	Compute Reward
		#	Return new state, reward, and whether the game is over.
		logging.info(action)
		return np.random.rand(15), \
			np.random.rand(), \
			[False if np.random.rand()>1e-1 else True][0], \
			np.random.rand()

def mlp(x, sizes, activation=tf.tanh, output_activation=None):
	# feedforward neural network.
	for size in sizes[:-1]:
		x = tf.layers.dense(x, units=size, activation=activation)
	return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

def train(hidden_sizes=[32], lr=1e-2, premodel=False 
		  epochs=50, batch_size=5000):

	# make environment, check spaces, get obs / act dims
	"""
	env = gym.make(env_name)
	assert isinstance(env.observation_space, Box), \
		"This example only works for envs with continuous state spaces."
	assert isinstance(env.action_space, Discrete), \
		"This example only works for envs with discrete action spaces."
	"""
	models_dir = "models/"
	env = Env()
	obs_dim = env.observation_space # 
	n_acts = env.action_space # q w e a s d and fire
	if premodel:
		premodel_file = intput("Name of pre trained model file: ")
		os.path.isfile(models_dir + premodel_file)
		# TODO: we need to set the variables/placeholders so that the model can be restored and used.
	else:
		file = input("Name the new RL model: ")
		# make core of policy network
		obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
		logits = mlp(obs_ph, sizes=hidden_sizes+[n_acts])

		# make action selection op (outputs int actions, sampled from policy)
		actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)

		# make loss function whose gradient, for the right data, is policy gradient
		weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
		act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
		action_masks = tf.one_hot(act_ph, n_acts)
		log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
		loss = -tf.reduce_mean(weights_ph * log_probs)

		# make train op
		train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

	# save model ops
	saver = tf.train.Saver()

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	if premodel:
		saver.restore(sess, models_dir+premodel_file)
# for training policy
	def train_one_epoch():
		# training > epochs[50] > batches[100] > episodes > steps
		# the policy gradient update happens once each epoch 
		# make some empty lists for logging --> trajectory (tau) for each batch
		batch_obs = []          # for observations
		batch_acts = []         # for actions
		batch_weights = []      # for R(tau) weighting in policy gradient
		batch_rets = []         # for measuring episode returns
		batch_lens = []         # for measuring episode lengths

		# reset episode-specific variables
		obs = env.reset()       # observation vector from game
		done = False            # signal that the game is over (when 5 minutes have passed)
		ep_rews = []            # list for rewards accrued throughout episode

		# collect experience by acting in the environment with current policy
		while True:
			# save obs
			batch_obs.append(obs.copy()) # obs has to be a numpy array
			obs_p = obs.reshape(1,-1) # this reshapes the obs matrix into a single vector
			# act in the environment
			act = sess.run(actions, feed_dict={obs_ph: obs_p})[0]
			obs, rew, done, _ = env.step(act) # how about <done>??

			# save action, reward
			batch_acts.append(act)
			ep_rews.append(rew)

			if done:
				# if episode is over, record info about episode
				ep_ret, ep_len = sum(ep_rews), len(ep_rews)
				batch_rets.append(ep_ret)
				batch_lens.append(ep_len)

				# the weight for each log-prob(a|s) is R(tau)
				batch_weights += [ep_ret] * ep_len

				# reset episode-specific variables
				obs, done, ep_rews = env.reset(), False, [] ## from our environment, reteive new obs

				# end experience loop if we have enough of it
				if len(batch_obs) > batch_size:
					break

		# take a single policy gradient update step
		batch_loss, _ = sess.run([loss, train_op],
								 feed_dict={
									obs_ph: np.array(batch_obs),
									act_ph: np.array(batch_acts),
									weights_ph: np.array(batch_weights)
								 })
		return batch_loss, batch_rets, batch_lens

	# training loop
	for i in range(epochs):
		batch_loss, batch_rets, batch_lens = train_one_epoch()
		print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
				(i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
		# Save the variables to disk.
		save_path = saver.save(sess, models_dir+file)
		print("Model saved in path: %s" % save_path)



if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
	parser.add_argument('-H', '--hostname', default='127.0.0.1', help='Hostname to connect to')
	parser.add_argument('-p', '--port', default=8052, type=int, help='Port to connect to')
	parser.add_argument('-n', '--name', default='TeamA:RandomBot', help='Name of bot')
	parser.add_argument('--lr', type=float, default=1e-2)
	parser.add_argument('--premodel', type=bool, default=False)
	args = parser.parse_args()

	print('\nUsing simplest formulation of policy gradient.\n')

	# Set up console logging
	if args.debug:
		logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.DEBUG)
	else:
		logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
	
	train(lr=args.lr, premodel=args.premodel)

