# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:15:54 2020

@author: jisuk
"""

# %% import modules
import tensorflow as tf
import numpy as np
import gym 
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr
#%%
# register(
#     id='FrozenLake-v3',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name': '4x4',
#             'is_slippery': False}
# )
#%%
env = gym.make('FrozenLake-v0')


# %% Q-network parameters
input_size = env.observation_space.n
output_size = env.action_space.n
hidden_n = 50;
learning_rate = 0.1

def one_hot(x):
    return np.identity(16)[x:x+1]

#%% Q-learning parameters
dis = 0.99
num_episodes = 2000


#%% prepare network

class NeuralNet(Model):
    # set layers
    def __init__(self):
        super(NeuralNet,self).__init__()
        # first fully connected layer
        #self.fc1 = layers.Dense(hidden_n,activation=tf.nn.relu)
        self.out = layers.Dense(output_size,activation=None,use_bias=False,)
        
    def call(self,x):
        #x = self.fc1(x)
        x = self.out(x)
        return x

# create instance of net
neural_net = NeuralNet()

# calculating loss
def loss(Q_pred,Y):
    return tf.reduce_sum(tf.square(Y - Q_pred))

optimizer = tf.optimizers.SGD(learning_rate)

def run_optimization(x,y):
    with tf.GradientTape() as g:
        #forward
        x = neural_net(x)
        # loss
        l = loss(x,y)
    
    trainable_variables = neural_net.trainable_variables
    gradients = g.gradient(l, trainable_variables)
    optimizer.apply_gradients(zip(gradients,trainable_variables))

#%% do learning

rList = []

for i in range(num_episodes):
    #reset env and get first new observation
    s = env.reset()
    e = 1./((i/50.)+10)
    
    rAll = 0
    done = False
    local_loss = []
    
    # the Q-network training
    
    j=0
    while not done:
        j+=1
        # choose and action by greedly (with a chance of random action) from the Q-network
        Qs = neural_net(one_hot(s)).numpy()
        
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)
        
        # get new state and reward from environment
        s1,reward, done,_ = env.step(a)
        
        if done:
            # update Q and no Qs+1, since if were terminate state
            Qs[0,a] = reward
        else:
            # obtain the Q_s1 values by feeding the new state through our network
            Qs1 = neural_net(one_hot(s1)).numpy()
            # update Q
            Qs[0,a] = reward + dis*np.max(Qs1)
        
        # train our network using target(Y) and predicted Q(Qpred) values
        run_optimization(tf.constant(one_hot(s)),tf.constant(Qs))
        
        rAll += reward
        s = s1    
        print(i,j)
    
    rList.append(rAll)
    print(rAll)
    
#%% visualize result
print("percent of succesful episodes: " + str(sum(rList)/num_episodes))
plt.bar(range(len(rList)), rList,color="blue")
plt.show()

#%% 
QS_list=  []
for i in range(16):
    Qs = neural_net(one_hot(i)).numpy()
    
    QS_list.append(Qs)
    
    
QS_list = np.array(QS_list)
real_viz = QS_list.reshape([4,4,-1])
real_viz_ = np.zeros(shape=(12,12))

for i in range(4):
    for j in range(4):
        real_viz_[1+3*i     , 1+3*j - 1] = real_viz[i,j,0]
        real_viz_[1+3*i + 1 , 1+3*j    ] = real_viz[i,j,1]
        real_viz_[1+3*i     , 1+3*j + 1] = real_viz[i,j,2]
        real_viz_[1+3*i - 1 , 1+3*j    ] = real_viz[i,j,3]
        real_viz_[1+3*i     , 1+3*j    ] = (4*i + j + 1) / 100.
        
env.render()