# -*- coding: utf-8 -*-
"""

"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import numpy.matlib
import random


import scipy.io as sio

import os
current_dir = os.path.dirname(os.path.realpath(__file__))
load_fn ="Env_data.mat"
load_data = sio.loadmat(os.path.join(current_dir, load_fn))

# Load data
state_H_data=torch.tensor(load_data['state_H'])
H_data=torch.tensor(load_data['H'])
users_position=torch.tensor(load_data['user_position'])
target_position=torch.tensor(load_data['target_position'])
CODEBOOK=torch.tensor(load_data['D_Nt'])


# parameters setting

EPISODE=2000
length_traj=20 # T
size_codebook=32  # G_t
N_t=16
U=4
N_r=U
f_c=28*10**9
M=8
delta_f=20*10**3
sigma_n=1
sigma_s=1
P_T=10**1.0
N_angle=60  
N_range=40  
eta=0.5
# TODO: 最优值待定
Fisher_opt=100
sum_rate_opt=20


def complexExp(x):
    x1=torch.exp(x[:,0])*torch.cos(x[:,1])
    y1=torch.unsqueeze(x1, 1)
    x2=torch.exp(x[:,0])*torch.sin(x[:,1])
    y2=torch.unsqueeze(x2, 1)
    y=torch.cat((y1,y2),1)
    return y

def dim1to2(x,mode):
    temp=torch.zeros(x.shape[0],2)
    if x.dim()==1:
        if mode=='real':
            temp[:,0]=x
        elif mode=='imag':
            temp[:,1]=x
    elif x.dim()==2:
        if x.shape[1]==1:
            if mode=='real':
                temp[:,0]=x.reshape(-1)
            elif mode=='imag':
                temp[:,1]=x.reshape(-1)
        elif x.shape[1]==2:
            temp=x
    return temp


def complex_matrix_multiply(matrix1, matrix2):
    real1=torch.real(matrix1)
    real1=real1.to(torch.float64)
    imag1=torch.imag(matrix1)
    imag1=imag1.to(torch.float64)
    
    real2=torch.real(matrix2)
    real2=real2.to(torch.float64)
    imag2=torch.imag(matrix2)
    imag2=imag2.to(torch.float64)
    
    real_product=torch.mm(real1,real2)-torch.mm(imag1,imag2)
    imag_product=torch.mm(real1,imag2)+torch.mm(imag1,real2)
    complex_product = real_product+1j*imag_product
    return complex_product



class Env_ISAC(gym.Env):     
    
    def reset(self):
        # state:  initial channel information
        state_H=torch.zeros(N_r, N_t)
        # state: historical positions
        state_P=torch.zeros(3,N_angle, N_range)
        return state_H, state_P
    
    
    def step(self, state_H, state_P, action, F_RF_this, H_this, users_position_this, target_position_this, sum_rate_opt, Fisher_opt):
        
        ### state transition
        state_H_next=state_H
        state_P_next=torch.zeros(3, N_angle, N_range)
        state_P_next[0:2]=state_P[1:3]

        angle_user=0
        range_user=0
        angle_target=0
        range_target=0
        for u in range(U):
            angle_user=int((users_position_this[u][0]*180/math.pi + 90)/3)  # coordinates in the spectrum
            range_user= int(users_position_this[u][1]/2.5)
            state_P_next[2][angle_user][range_user]=1
            
        angle_target=int((target_position_this[0]*180/math.pi + 90)/3)
        range_target=int(target_position_this[1]/2.5)
        state_P_next[2,angle_target,range_target]=2
        state_P_next[2]=torch.abs( state_P_next[2] + 0.1*torch.randn(N_angle, N_range) )  
        state_P_next[2]=state_P_next[2]/torch.max(state_P_next[2])   #normalization
        

        ### Precoding update
        action=torch.round(action)
        for i in range(action_dim):
            if action[i] >1.0:
                action[i]=torch.tensor(1.0)
            if action[i] <0.0:
                action[i]=torch.tensor(0.0)
        if U<2:
            action[0]=0
            action[1]=0
            action[2]=0
            action[3]=0
        elif U<3:
            action[0]=0
            action[1]=0
            action[2]=0
        elif U<5:
            action[0]=0
            action[1]=0
        elif U<9:
            action[0]=0
        index_column=int(action[0]*(2**3)+action[1]*(2**2)+action[2]*(2**1)+action[3]*(2**0) )
        index_codebook=int( action[4]*(2**4)+action[5]*(2**3)+action[6]*(2**2)+action[7]*(2**1)+action[8]*(2**0) )
        
        
        F_RF_0=F_RF_this
        F_RF_0[:,index_column]=CODEBOOK[:,index_codebook]  
        F_RF_next=F_RF_0
        
        ###reward calculation
        # Fisher information
        steering_phase=torch.zeros(1,N_t)
        for i in range(N_t):
            steering_phase[0,i]=-math.pi*torch.sin(target_position_this[0])*(i)  #位置信息待定
        y=complexExp(dim1to2(steering_phase[0],'imag'))
        steering_vec=y[:,0]+1j*y[:,1]
        a=steering_vec.unsqueeze(0).t()
        a_diff=torch.zeros(N_t,1)+1j*torch.zeros(N_t,1)
        for i in range(N_t):
            a_diff[i,0]=-1j*math.pi*a[i,0]*i*torch.cos(target_position_this[0])  
            
        A_diff=complex_matrix_multiply(a_diff, a.t())+complex_matrix_multiply(a, a_diff.t())
        Fisher=0
        for m in range(M):
            f_m=f_c+delta_f*m
            complex_loss=1/(target_position_this[1]**1.5)*f_c/f_m  
            AFFA=complex_matrix_multiply( complex_matrix_multiply(torch.conj(F_RF_0.t()), torch.conj(A_diff.t()) ), complex_matrix_multiply(A_diff, F_RF_0) )  #这里有问题
            Fisher=Fisher+2*complex_loss**2*torch.real(AFFA.trace())
            
        # sum rate
        metric_com_rate=0+0*1j
        F_wideband=torch.zeros([N_t*M,N_r*M])+0*1j*torch.zeros([N_t*M,N_r*M])
        F_wideband=torch.block_diag(F_RF_0, F_RF_0, F_RF_0, F_RF_0, F_RF_0, F_RF_0 , F_RF_0 , F_RF_0)

        for ii in range(N_r*M):
            inference=0+0*1j
            for jj in range(N_r*M):
                #inference=inference+( (H_this[ii,:]*F_wideband[:,jj]*F_wideband[:,jj].H*H_this[ii,:].H)[0,0])
                inference=inference+  torch.norm( complex_matrix_multiply(H_this[ii,:].unsqueeze(0), F_wideband[:,jj].unsqueeze(0).t()) ,2 )**2
            #inference=inference-((H_this[ii,:]*F_wideband[:,ii]*F_wideband[:,ii].H*H_this[ii,:].H)[0,0])
            inference=inference- torch.norm(complex_matrix_multiply(H_this[ii,:].unsqueeze(0), F_wideband[:,ii].unsqueeze(0).t()) , 2)**2
            #metric_com_rate += math.log2(1 + ((H_this[ii,:]*F_wideband[:,ii]*F_wideband[:,ii].H*H_this[ii,:].H)[0,0])/(inference+sigma_n**2))
            metric_com_rate+= math.log2(1 + (torch.norm( complex_matrix_multiply(H_this[ii,:].unsqueeze(0), F_wideband[:,ii].unsqueeze(0).t()),2)**2))
            sum_rate=metric_com_rate
        
        # weighted sum as reward
        reward=eta* (Fisher/Fisher_opt) + (1-eta)* (sum_rate/sum_rate_opt)
        return state_H_next, state_P_next, F_RF_next, reward, Fisher, sum_rate
    
    def close(self):
        pass


env=Env_ISAC()

### training parameters
# Hyper Parameters
BATCH_SIZE = 64
LR = 0.05                  # learning rate
EPSILON = 0.6              # greedy policy
GAMMA = 0.6                # reward discount
TARGET_REPLACE_ITER = 10   # target update frequency
MEMORY_CAPACITY = 400

env = env.unwrapped
N_ACTIONS=2*N_r
epi_decay=(1-EPSILON)/EPISODE


## Model structure 

action_dim=9
class Actor(nn.Module):
    def __init__(self, action_dim):
        super(Actor,self).__init__()
        self.CNN_H_1=nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3,  stride=1,padding=1)
        self.CNN_H_2=nn.Conv2d(4, 8, kernel_size=3,  stride=1,padding=1)
        self.CNN_P_1=nn.Conv2d(3, 4, kernel_size=3,  stride=1,padding=1)
        self.CNN_P_2=nn.Conv2d(4, 6, kernel_size=3,  stride=1,padding=1)
        self.pool=nn.MaxPool2d(2,2)

        self.fc1=nn.Linear(int(N_t/4)*int(N_r/4)*8+int(N_angle/4)*int(N_range/4)*6, 64)
        self.fc2=nn.Linear(64, action_dim)
        
    def forward(self, state_H, state_P):
        x_1=self.pool(torch.relu(self.CNN_H_1(state_H)))
        x_1=self.pool(torch.relu(self.CNN_H_2(x_1)))
        x_2=self.pool(torch.relu(self.CNN_P_1(state_P)))
        x_2=self.pool(torch.relu(self.CNN_P_2(x_2)))
        
        x_1 = x_1.view(-1, int(N_t/4)*int(N_r/4)*8 )
        x_2 = x_2.view(-1, int(N_angle/4)*int(N_range/4)*6 ) 
        x=torch.cat((x_1,x_2), 1)
        x=torch.relu(self.fc1(x))
        action=(torch.tanh(self.fc2(x))+ torch.ones(1,action_dim) )/2   #action: [0,1]    
        return action

class Critic(nn.Module):
    def __init__(self, action_dim):
        super(Critic, self).__init__()
        self.CNN_H_1=nn.Conv2d(1, 4, kernel_size=3,  stride=1,padding=1)
        self.CNN_P_1=nn.Conv2d(3, 4, kernel_size=3,  stride=1,padding=1)
        self.pool=nn.MaxPool2d(2,2)

        self.fc1=nn.Linear(int(N_t/2)*int(N_r/2)*4 + int(N_angle/2)*int(N_range/2)*4 + action_dim, 64) 
        self.fc2=nn.Linear(64, 32)
        self.fc3=nn.Linear(32, 1)
        
    def forward(self, state_H, state_P, action):
        x_1=self.pool(torch.relu(self.CNN_H_1(state_H)))
        x_2=self.pool(torch.relu(self.CNN_P_1(state_P)))
        
        x_1 = x_1.view(-1, int(N_t/2)*int(N_r/2)*4 ) 
        x_2 = x_2.view(-1, int(N_angle/2)*int(N_range/2)*4 ) 
        x=torch.cat((x_1,x_2), 1)
        xx=torch.cat((x, action), 1)
        xx=torch.relu(self.fc1(xx))
        xx=torch.relu(self.fc2(xx))
        Q=self.fc3(xx)
        return Q
    
    
# O-U noise
class OrnsteinUhlenbeckNoise:
    def __init__(self, action_dim, mu=0, theta=0.5, sigma=0.3):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state
    
# DDPG training
class DDPG:
    def __init__(self, action_dim, actor_lr, critic_lr, gamma):
        self.action_dim = action_dim
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        # ACtor and critic networks
        self.actor = Actor(action_dim)
        self.critic = Critic(action_dim)
        # target networks
        self.target_actor = Actor(action_dim)
        self.target_critic = Critic(action_dim)
        # copy
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        # optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.noise = OrnsteinUhlenbeckNoise(action_dim)
    # Action selection
    def select_action(self, s_H, s_P):  
        state_H = s_H.unsqueeze(0).unsqueeze(0)  #修改维度适配网络， 为什么原来就不适配呢？
        state_P = s_P.unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_H, state_P).squeeze(0)
        # add O-U noise
        action += self.noise.sample()
        return action
    # Update network
    def update(self, replay_buffer, batch_size):
        state_H, state_P, action, next_state_H, next_state_P, reward = replay_buffer.sample(batch_size)
        # Q-value calculation
        with torch.no_grad():
            target_action = self.target_actor(next_state_H, next_state_P)
            target_q = self.target_critic(next_state_H, next_state_P, target_action)
            target_q = target_q.squeeze(1)
            target_q = reward.real.float() + self.gamma * target_q 
  
        current_q = self.critic(state_H, state_P, action)
        current_q = current_q.squeeze(1)
        # UPdate Critic
        critic_loss = F.mse_loss(current_q, target_q) 
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Update Actor
        pred_action = self.actor(state_H, state_P)
        actor_loss = -self.critic(state_H, state_P, pred_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Update target networks
        self.soft_update()
    # soft update
    def soft_update(self, tau=0.06):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
# replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    def push(self, state_H, state_P, action, next_state_H, next_state_P, reward):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        state_H = torch.unsqueeze(state_H, 0) 
        next_state_H = torch.unsqueeze(next_state_H, 0)
        action = action.float()
        self.buffer[self.position] = (state_H, state_P, action, next_state_H, next_state_P, reward)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):  
        sample_index=np.random.choice(self.__len__(), batch_size)
        b_memory=[self.buffer[i] for i in sample_index] 
        state_H, state_P, action, next_state_H, next_state_P, reward = zip(*b_memory)
        return (torch.stack(state_H), torch.stack(state_P), torch.stack(action), torch.stack(next_state_H), torch.stack(next_state_P), torch.stack(reward))

    def __len__(self):
        return len(self.buffer)
    

## Record training
STATUS_EVERY=20
epi_status_dqn={'ep':[],'avg':[],'max':[],'min':[]}
epi_metric_com={'ep':[],'avg':[],'max':[]}
epi_metric_sensing={'ep':[],'avg':[],'min':[]}
rewards_status_every_dqn=[]
metric_com_every=[]
metric_sensing_every=[]


record_angle=[]
record_range=[]
record_action=[]
record_reward=[]



agent = DDPG(action_dim, actor_lr=0.01, critic_lr=0.01, gamma=GAMMA )

replay_buffer = ReplayBuffer(capacity=MEMORY_CAPACITY)

count_rate=np.array([0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0])

for i_episode in range(EPISODE):
    s_H, s_P = env.reset()
    record_angle.clear()
    record_range.clear()
    record_action.clear()
    record_reward.clear()
    ep_r = 0
    ep_metric_c=0
    ep_metric_s=0
    
    # initialize a precoding matrix
    F_RF_this=torch.ones([N_t, N_r])+1j*0*torch.ones([N_t, N_r])

    for tt in range(length_traj):
        a = agent.select_action(s_H, s_P)
        # take action
        s_H_next, s_P_next, F_RF_next, r, metric_sensing, metric_com=env.step(s_H, s_P, a, F_RF_this, H_data[i_episode,tt], users_position[i_episode,tt], target_position[i_episode,tt], sum_rate_opt, Fisher_opt) 
        F_RF_this=F_RF_next
        
        replay_buffer.push(s_H, s_P, a, s_H_next, s_P_next, r)
        
        if i_episode>EPISODE/2:
            count_rate[int(np.ceil(metric_com))] +=1
        ep_r += r.real.float() 
        ep_metric_c += metric_com.real 
        ep_metric_s += metric_sensing
        
        if len(replay_buffer) > BATCH_SIZE:
            agent.update(replay_buffer, BATCH_SIZE)  
            
        s_H = s_H_next
        s_P = s_P_next
        record_action.append(a)
        record_reward.append(r)
        
    rewards_status_every_dqn.append(ep_r)
    metric_com_every.append(ep_metric_c/length_traj)
    metric_sensing_every.append(ep_metric_s/length_traj)
    
    if not i_episode%STATUS_EVERY:
        avg=np.mean(rewards_status_every_dqn)
        epi_status_dqn['ep'].append(i_episode)
        epi_status_dqn['avg'].append(avg)
        maxx=max(rewards_status_every_dqn)
     
        epi_status_dqn['max'].append(max(rewards_status_every_dqn))
        epi_status_dqn['min'].append(min(rewards_status_every_dqn))
        
        epi_metric_com['ep'].append(i_episode)
        epi_metric_com['avg'].append(np.mean(metric_com_every))
        epi_metric_com['max'].append(max(metric_com_every))
        
        epi_metric_sensing['ep'].append(i_episode)
        epi_metric_sensing['avg'].append(np.mean(metric_sensing_every))
        epi_metric_sensing['min'].append(min(metric_sensing_every))
        
        rewards_status_every_dqn.clear()
        metric_com_every.clear()
        metric_sensing_every.clear()
        
        print(f'episode:{i_episode},avg:{avg}')

count_rate=count_rate/np.sum(count_rate)
