import gym
import numpy as np
import random
import tensorflow as tf
import xlsxwriter
import matplotlib.pyplot as plt

num_trials= 50
num_episodes=1000

gamma=.99

def runEnvironment(gList,trail):
    env = gym.make ('FrozenLake-v0')

    # Setting the learning paramaters
    e=1.
    n_stateSpace= env.observation_space.n
    n_actionSpace = env.action_space.n
    n_nodes_hl1 = 50
    inputs1 = tf.placeholder(shape=[1,n_stateSpace],dtype=tf.float32)

    hidden_1_layer={'weights':tf.Variable(tf.random_normal([n_stateSpace,n_nodes_hl1])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_actionSpace])),
                  'biases':tf.Variable(tf.random_normal([n_actionSpace]))}
    l1 = tf.add(tf.matmul(inputs1,hidden_1_layer['weights']),hidden_1_layer['biases'])     # summation
    l1 =tf.nn.relu(l1)
    Qout = tf.matmul(l1,output_layer['weights'])+output_layer['biases']    # Gives a 16-dimensional vector
    predict = tf.argmax(Qout,1)

    # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
    nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)                # We will be using the Q-learning predictor here, note this the
    loss = tf.reduce_sum(tf.square(nextQ-Qout))
    #trainer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    trainer = tf.train.AdamOptimizer(learning_rate=0.001)
    updateModel = trainer.minimize(loss)

    init = tf.global_variables_initializer()


    #### Create arrays(lists) to contain total rewards and steps per episode
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            # Resets the enviroment and gets the first new observation
            s=env.reset()
            gRet=0
            d=False
            t=0
            #The Q-Network
            while t<99:
                t+=1
                # Choose an action by greedily ( with e chance of random action) from
                # from the Q-Network
                a,allQ = sess.run([predict,Qout],feed_dict = {inputs1:np.identity(n_stateSpace)[s:s+1]})
                if np.random.rand(1)<e:
                    a[0]=env.action_space.sample()
                    #Get new state and reward from enviroment
                s1,r,d,_ = env.step(a[0])
                    #Obtain the Q' values by feeding the new state through our network
                Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(n_stateSpace)[s1:s1+1]})
                #Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0,a[0]] = r + gamma*maxQ1
                #Train our network using target and predicted Q values
                _ = sess.run(updateModel,feed_dict={inputs1:np.identity(n_stateSpace)[s:s+1],nextQ:targetQ})
                gRet += r
                s = s1
                if d == True:
                    #Reduce chance of random action as we train the model.
                    e = 1./(i/1000+10)
                    print 'Trail:',trail,'Episode:',i,'Return',gRet
                    break
            gList[i]=gRet


def runExperiment(num_trials):
    avgList = np.zeros([num_episodes,1])  #Used for incremental sample averaging of the score over the episodes
    for trail in range(num_trials):
        gList = np.zeros([num_episodes,1])
        runEnvironment(gList,trail)
        avgList = avgList + ((gList-avgList)/(trail+1))
    return avgList

def createSpreadSheet(data):
    workbook = xlsxwriter.Workbook('Q-Network_Data.xlsx')
    worksheet = workbook.add_worksheet()

    for row, data in enumerate(data):
        worksheet.write_row(row,0,data)

    workbook.close()

data = runExperiment(num_trials)
createSpreadSheet(data)

plots = plt.plot(data,'ro')
plt.xlabel('Episodes')
plt.ylabel('Average Return per Episode')
plt.title('Q-Network')
plt.show()
