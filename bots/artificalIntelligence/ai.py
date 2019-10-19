import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam


import sys
sys.path.append("C:\\Users\\polek\\Desktop\\CodeBlockProjects\\insert-a-creative-name-here-2\\bots\\")


from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy


ENV_NAME = 'CartPole-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

input = Input(shape=(1,) + env.observation_space.shape)
lastLayer = Flatten()(input)
lastLayer = Dense(16)(lastLayer)
lastLayer = Activation('relu')(lastLayer)
lastLayer = Dense(16)(lastLayer)
lastLayer = Activation('relu')(lastLayer)
lastLayer = Dense(16)(lastLayer)
connectingLayer = Activation('relu')(lastLayer)

movement = Dense(6)(connectingLayer)
movementOutput = Activation('relu')(movement)
firing = Dense(1)(connectingLayer)
firingOutput = Activation('softmax')(firing)
output = concatenate([movementOutput, firingOutput])

model = Model(input, output)
print(model.summary())


# SARSA does not require a memory.
policy = BoltzmannQPolicy()
sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
sarsa.fit(env, nb_steps=5000, visualize=False, verbose=2)
sarsa.fit(env, nb_steps=50000, visualize=True, verbose=2)

# After training is done, we save the final weights.
sarsa.save_weights('sarsa_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
sarsa.test(env, nb_episodes=5, visualize=True)
