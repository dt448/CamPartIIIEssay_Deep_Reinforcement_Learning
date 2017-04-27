import gym
import numpy as np


env = gym.make ('FrozenLake8x8-v0')

#Initialize table with all zeros
# env.observation_space.n is the number of elements in the obs and action space
# in this case it is [16x4 table/array]
Q = np.zeros([env.observation_space.n,env.action_space.n])
tpolicy = np.zeros([env.observation_space.n])
#Set learning paramaters
alpha = .85
gamma = .99
num_episodes = 50000
# create lists to contain total rewards and steps per epsidoe
rList =[]
for i in range(num_episodes):
    # Reset enviroment and get first new observation
    s = env.reset() # env reset return the observation
    rAll = 0
    d = False # if the episode is done?
    j = 0
    #The Q-table learning algorithm
    while j<100:
        j+=1
        #Behaviour Policy : Choose an action by greedily(with noise epsilon greeedy) picking
        # from Q tabl
        a=np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        # Get new state and reward from the enviroment
        s1,r,d,_ = env.step(a)
        #Update Q-table with new knowledge
        Q[s,a] = Q[s,a] + alpha*(r+gamma*np.max(Q[s1,:]-Q[s,a]))
        #Target policy
        rAll+= r
        s=s1
        if d == True or j==99:
            if i % 400 == 0:
                #for i in range(tpolicy.size):
                #    tpolicy[i] = np.argmax(Q[i,:])
                #print tpolicy
                print "Episode number",i
            #     print "Time step of end:",j
                print "Total Reward:",rAll
            #     print Q
            break
    rList.append(rAll)

print "Score over time:" + str(sum(rList)/num_episodes)
print "Final Q-table Values"
print Q
