import gym
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter

num_trials= 1000                      #Enter the number of trails(experiments) you wish to carry out
num_episodes=500                   #Enter the number of episodes for each trail of the experiment to have

gamma = .99                     #Discount rate


def runEnvironment(gList,num_episodes,alpha):
    env = gym.make ('FrozenLake-v0')
    Q = np.zeros([env.observation_space.n,env.action_space.n])
    for i in range(num_episodes):
        s = env.reset()
        gRet = 0                                                    #This will be the return/score
        d = False
        t = 0                                                       #time steps, limited to 100 to prevent loops

        #The Q-table learning algorithm
        while t<100:
            a=np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./((i+1))))    #Behaviour Policy : Choose an action epsilon-greedily
                                                                                        #from Q tabl
            s1,r,d,_ = env.step(a)                                                      #Get new state and reward from the enviroment
            Q[s,a] = Q[s,a] + alpha*(r+gamma*np.max(Q[s1,:]-Q[s,a]))                    #Update Q-table with new knowledge
            gRet+= r
            s=s1
            if d == True:
                break
            t+=1
        gList[i]=gRet

def runExperiment(num_episodes, num_trials,alpha):
    avgList = np.zeros([num_episodes,1])  #Used for incremental sample averaging of the score over the episodes
    for i in range(num_trials):
        gList = np.zeros([num_episodes,1])
        runEnvironment(gList,num_episodes,alpha)
        avgList = avgList + ((gList-avgList)/(i+1))
        print (alpha,i)
    return avgList

def createSpreadSheet(data,alpha):
#Making SpreadSheet to store data locally -this is just used for record purposes
    workbook = xlsxwriter.Workbook(alpha+'Alpha'+'Comparision' '.xlsx')
    worksheet = workbook.add_worksheet()

    for row, data in enumerate(data):
        worksheet.write_row(row,0,data)

    workbook.close()

for _ in range(3):
    if _==0:
        alpha = 0.01
        data1 = runExperiment(num_episodes, num_trials,alpha)
        createSpreadSheet(data1,str(alpha))
    elif _==1:
        alpha = 0.5
        data2 =runExperiment(num_episodes, num_trials,alpha)
        createSpreadSheet(data2,str(alpha))
    elif _==2:
        alpha = 0.99
        data3 =runExperiment(num_episodes, num_trials,alpha)
        createSpreadSheet(data3,str(alpha))

plots = plt.plot(range(num_episodes),data1,'g-',range(num_episodes),data2,'b--',range(num_episodes),data3,'r-')
plt.setp(plots, linewidth=.75)
plt.xlabel('Episodes')
plt.ylabel('Average Return per Episode')
plt.title('Q-Learning with Different Step Size')

plt.show()
