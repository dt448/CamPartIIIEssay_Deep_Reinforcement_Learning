import gym
import numpy as np
import random
import tensorflow as tf
import experienceBuffer as exB
import matplotlib.pyplot as plt
import xlsxwriter

num_trials= 50
num_episodes=1000

gamma=.99
batch_size = 128 #How many experiences to use for each training step.
update_freq = 5 #How often to perform a training step.
startE = 1. #Starting chance of random action
endE = 0.1 #Final chance of random action
anneling_steps = 1000 #How many steps of training to reduce startE to endE.
pre_train_steps = 1000 #How many steps of random actions before training begins.


def runEnvironment(gList,trail):
    env = gym.make ('FrozenLake-v0')
    myBuffer = exB.experience_buffer()

    # Setting the learning paramaters
    e = startE
    stepDrop = (startE - endE)/anneling_steps
    total_steps = 0

    n_stateSpace= env.observation_space.n
    n_actionSpace = env.action_space.n
    n_nodes_hl1 = 100
    n_nodes_hl2 = 100

    inputs1 = tf.placeholder(shape=[1,n_stateSpace],dtype=tf.float32)

    hidden_1_layer={'weights':tf.Variable(tf.random_normal([n_stateSpace,n_nodes_hl1])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_actionSpace])),
                  'biases':tf.Variable(tf.random_normal([n_actionSpace]))}

    l1 = tf.add(tf.matmul(inputs1,hidden_1_layer['weights']),hidden_1_layer['biases'])     # summation
    l1 =tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])     # summation
    l2 =tf.nn.relu(l2)

    Qout = tf.matmul(l2,output_layer['weights'])+output_layer['biases']    # Gives a 16-dimensional vector
    predict = tf.argmax(Qout,1)

    # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
    nextQ = tf.placeholder(shape=[1,n_actionSpace],dtype=tf.float32)                # We will be using the Q-learning predictor here, note this the
    loss = tf.reduce_sum(tf.square(nextQ-Qout))
    #trainer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
    trainer = tf.train.AdamOptimizer(learning_rate=0.001)
    updateModel = trainer.minimize(loss)

    init = tf.global_variables_initializer()


    #### Creeate arrays(lists) to contain total rewards and steps per episode
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            episodeBuffer = exB.experience_buffer()
            # Resets the enviroment and gets the first new observation
            s = env.reset()
            randStart=s
            ran=0
            d1 = False
            while ran<3:
                ran +=1
                randAct = env.action_space.sample()
                randStart,_,__,d1 =env.step(randAct)
                if d1 == True:
                    ran = 0
                    env.reset()
            s = randStart
            gRet=0
            d=False
            j=0
            #The Q-Network
            while j<150:
                #print(gw.dispGrid(s))
                j+=1
                # Choose an action by greedily ( with e chance of random action) from
                # from the Q-Network
                sarray = np.identity(n_stateSpace)[s:s+1]
                a,allQ = sess.run([predict,Qout],feed_dict = {inputs1:sarray})
                if np.random.rand(1)<e or total_steps < pre_train_steps:
                    a[0]=env.action_space.sample()
                    #Get new state and reward from enviroment
                s1,r,d,_ = env.step(a[0])
                s1array = np.identity(n_stateSpace)[s1:s1+1]
                total_steps += 1
                episodeBuffer.add(np.reshape(np.array([sarray,a[0],r,s1array,d]),[1,5])) #Save the experience to our episode buffer.
                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= stepDrop
                    if total_steps % (update_freq) == 0:
                        #print 'Updating network'
                        trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                        trainBatchs = np.vstack(trainBatch[:,0])
                        trainBatcha=np.vstack(trainBatch[:,1])
                        trainBatchr=np.vstack(trainBatch[:,2])
                        trainBatchs1 = np.vstack(trainBatch[:,3])

                        for eps in  range(batch_size):
                            #print trainBatchs1[eps]
                            Q1 = sess.run(Qout,feed_dict={inputs1:np.array([trainBatchs1[eps]])})
                            #print "q1:",Q1
                            #Obtain maxQ' and set our target value for chosen action.
                            oldQ = sess.run(Qout,feed_dict={inputs1:np.array([trainBatchs[eps]])})
                            #print oldQ
                            a1=trainBatcha[eps]
                            r1=trainBatchr[eps]
                            maxQ1 = np.max(Q1)
                            targetQ = oldQ
                            targetQ[0,a1] = r1 + gamma*maxQ1
                            #Train our network using target and predicted Q values
                            _ = sess.run(updateModel,feed_dict={inputs1:np.array([trainBatchs[eps]]),nextQ:targetQ})
                gRet += r
                s = s1
                if d == True:
                    #Reduce chance of random action as we train the model.
                    print 'Trail:',trail,'Episode:',i,'Return',gRet
                    break
            gList[i]=gRet
            myBuffer.add(episodeBuffer.buffer)

def runExperiment(num_trials):
    avgList = np.zeros([num_episodes,1])  #Used for incremental sample averaging of the score over the episodes
    for trail in range(num_trials):
        gList = np.zeros([num_episodes,1])
        runEnvironment(gList,trail)
        avgList = avgList + ((gList-avgList)/(trail+1))
    return avgList

def createSpreadSheet(data):
    workbook = xlsxwriter.Workbook('Experince_Replay_Data.xlsx')
    worksheet = workbook.add_worksheet()

    for row, data in enumerate(data):
        worksheet.write_row(row,0,data)

    workbook.close()

data = runExperiment(num_trials)
createSpreadSheet(data)

plots = plt.plot(gList,'ro')
plt.xlabel('Episodes')
plt.ylabel('Average Return per Episode')
plt.title('Experience Replay on Stochastic Environment')
plt.show()
