import gym
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter

num_trials= 1000                      #Enter the number of trails(experiments) you wish to carry out
num_episodes=500                   #Enter the number of episodes for each trail of the experiment to have


def runEnvironment(gList,num_episodes,lAlgo='Q-Learning'):
    env = gym.make ('FrozenLake8x8-v0')
    Q = np.zeros([env.observation_space.n,env.action_space.n])
    alpha = .85                                                     #Stepsize alpha
    gamma = .99                                                     #Discount rate
    for i in range(num_episodes):
        s = env.reset()
        gRet = 0                                                    #This will be the return/score
        d = False
        t = 0                                                       #time steps, limited to 100 to prevent loops

        #The Q-table learning algorithm
        while t<100:
            a=np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./((i+1))))     #Behaviour Policy : Choose an action epsilon-greedily
                                                                                        # from Q tabl
            s1,r,d,_ = env.step(a)
            if lAlgo=='Q-Learning':                                                      # Get new state and reward from the enviroment
                Q[s,a] = Q[s,a] + alpha*(r+gamma*np.max(Q[s1,:]-Q[s,a]))                    #Update Q-table with new knowledge
            elif lAlgo== 'SARSA':
                Q[s,a] = Q[s,a] + alpha*(r+gamma*(Q[s1,a]-Q[s,a]))
            gRet+= r
            s=s1
            if d == True:
                #if i % 400 == 0:
                    #for i in range(tpolicy.size):
                    #    tpolicy[i] = np.argmax(Q[i,:])
                    #print tpolicy
                    #print "Episode number",i
                #   print "Time step of end:",j
                    #print "Total Reward:",rAll
                #   print Q
                break
            t+=1
        gList[i]=gRet

def runExperiment(num_episodes, num_trials,lAlgo):
    avgList = np.zeros([num_episodes,1])  #Used for incremental sample averaging of the score over the episodes
    for i in range(num_trials):
        gList = np.zeros([num_episodes,1])
        runEnvironment(gList,num_episodes,lAlgo)
        avgList = avgList + ((gList-avgList)/(i+1))
        print (lAlgo,i)
    return avgList

def createSpreadSheet(data,lAlgo):
#Making SpreadSheet to store data locally -this is just used for record purposes
    workbook = xlsxwriter.Workbook(lAlgo + 'Comparision' '.xlsx')
    worksheet = workbook.add_worksheet()

    for row, data in enumerate(data):
        worksheet.write_row(row,0,data)

    workbook.close()

for _ in range(2):
    if _==0:
        lAlgo ='Q-Learning'
        data1 = runExperiment(num_episodes, num_trials,'Q-Learning')
        createSpreadSheet(data1,lAlgo)
    else:
        lAlgo ='SARSA'
        data2 =runExperiment(num_episodes, num_trials,'SARSA')
        createSpreadSheet(data2,lAlgo)

plt.plot(range(num_episodes),data1,'r-',range(num_episodes),data2,'b--')
plt.xlabel('Episodes')
plt.ylabel('Average Return per Episode')
plt.title('Q-Learning vs SARSA')
plt.show()
