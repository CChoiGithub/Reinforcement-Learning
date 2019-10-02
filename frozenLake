import numpy as np
import gym
import random

#create environment
env = gym.make("FrozenLake-v0")
actionSize = env.action_space.n
stateSize = env.observation_space.n

#create qtable and init with zeros
qtable = np.zeros((stateSize, actionSize))
print(qtable)

#create hyperparameters
totalEpisodes = 15000
learningRate = 0.8
maxSteps = 99   #max steps per episode
gamma = 0.95 # discount rate

#exploration parameters
epsilon = 1
maxEpsilon = 1.0
minEpsilon = 0.01
decayRate = 0.005

#list of rewards
rewards = []

for episode in range(totalEpisodes):
    #reset environment
    state = env.reset()
    step = 0
    done = False #game not finished
    totalRewards = 0

    for step in range(maxSteps):
        #choose action to take
        #randomise the number
        expExpTradeoff = random.uniform(0,1) #random number in (0,1), uniformly distributed
        #if this number > epsilon then exploitation
        if expExpTradeoff > epsilon:
            action = np.argmax(qtable[state, :])
        #otherwise random choice -> exploration
        else:
            action = env.action_space.sample()
        #take action and observe the outcome state and reward
        newState, reward, done, info = env.step(action)
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        qtable[state, action] = qtable[state, action] + learningRate * (reward + gamma + np.max(qtable[newState, :]) - qtable[state, action])
        totalRewards += reward

        if done == True:
            break

        #redute epsilon: to shrink the exploration area
        epsilon = minEpsilon + (maxEpsilon - minEpsilon) * np.exp(-decayRate * episode) #np.exp: e^
        rewards.append(totalRewards)

    print("Score over time: " + str(sum(rewards)/totalEpisodes))
    print(qtable)

env.reset()
for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(maxSteps):

        action = np.argmax(qtable[state, :])
        newState, reward, done, info = env.step(action)

        if done:
            env.render()

            print("Number of steps", step)
            break
        state = newState
    env.close()
