import gym
import numpy as np

env = gym.make ( 'FrozenLake-v0')

#Initialize table with all zeros
# env.observation_space.n is the number of elements in the obs and action space
Q = np.zeros([env.observation_space.n,env.action_space.n])
#Set learning paramaters
lr = .85
y = .99
num_episodes = 2000
# create lists to contain total rewards and steps per epsidoe
#jList = []
rList =[]
for i in range(num_episodes):
    # Reset enviroment and get first new observation
    s = env.reset() # env reset return the observation
    rAll = 0
    d = False # if the episode is done?
    j = 0
    #The Q-table learning algorithm
    while j<99:
        j+=1
        #Choose an action by greedily(with noise epsilon greeedy) picking
        # from Q tabl
        a=np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        # Get new state and reward from the enviroment
        s1,r,d,_ = env.step(a)
        #Update Q-table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r+y*np.max(Q[s1,:]-Q[s,a]))
        rAll+= r
        s=s1
        if d == True:
            break
    rList.append(rAll)

print "Score over time:" + str(sum(rList)/num_episodes)
print "Final Q-table Values"
print Q
