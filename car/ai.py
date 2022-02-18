#Importing necessary libraries
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional  as Func
from torch.optim import Adam
from torch.autograd import Variable

temperature = 100

class NeuralNewtwork(nn.Module):
    def __init__(self,input_size,nb_action):
        super(NeuralNewtwork,self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        #For connection between input and hidden layer
        self.fc1 = nn.Linear(input_size,30)

        #For connection between hidden layer and output layer
        self.fc2 = nn.Linear(30,nb_action)

    #Function for forward propagation
    def forward(self,state):
        #Activating hidden neurons
        x = Func.relu(self.fc1(state))

        #Q-values ([left,front,right])
        q_values = self.fc2(x)

        return q_values

#Experience replay
class ReplayMemory(object):
    def __init__(self,capacity):
        #The maximum size of memory
        self.capacity = capacity 

        #Memory
        self.memory = []

    def push(self,event):
        #Appendind event to the memory
        # event-> [last state, new state, last action, last reward]

        self.memory.append(event)

        #Retaining events only upto the capacity
        if(len(self.memory)>self.capacity):
            del self.memory[0]

    def sample(self,batch_size):
        #Taking random events from memory
        samples = zip(*random.sample(self.memory,batch_size))
        return map(lambda x: Variable(torch.cat(x,0)),samples)

#Deep Q-Learning
class DeepQNetwork:
    def __init__(self,input_size ,nb_action ,gamma = 0.01,capacity = 100000):
        self.gamma = gamma
        self.reward_window = []

        #Creating NN,memory and optimizer and initializing state ,action and reward
        self.model = NeuralNewtwork(input_size,nb_action)
        self.memory = ReplayMemory(capacity)
        self.optimizer = Adam(self.model.parameters(),lr=0.001)

        #Initializing stste with zeros(tensor with zeros)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = torch.LongTensor([0])
        self.last_reward = torch.LongTensor([0])

    def select_action(self,state):
        #Getting probabilities of output
        #Temp param = 7 => Scaling
        with torch.no_grad():
            probabilities = Func.softmax(self.model(state)*temperature)

            #Random draw from the distribution
            action = probabilities.multinomial(num_samples =1)

            return action.data[0,0]

    def learn(self,batch_state,batch_next_state,batch_reward,batch_action):
        #batch_state -> Input , batch_action -> Output
        outputs = self.model(batch_state).gather(1,batch_action.unsqueeze(1)).squeeze(1)

        #Calculating output
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward

        #Computing loss
        temporal_diff_loss = Func.smooth_l1_loss(outputs,target)

        #Re-initializing the optimizer
        self.optimizer.zero_grad()

        #Back-Propagation for weights updation
        temporal_diff_loss.backward(retain_graph=True)
        self.optimizer.step()

    def update(self,last_reward,state):
        #Converting list of event into tensor
        new_state = torch.Tensor(state).float().unsqueeze(0)

        #Updating the new state to the memory
        self.memory.push((self.last_state,new_state,torch.LongTensor([int(self.last_action)]),torch.LongTensor([self.last_reward])))

        #Selecting action
        action = self.select_action(new_state)

        #Learning if memory size greater than 100
        if len(self.memory.memory) > 100:
            batch_state,batch_next_state,batch_action,batch_reward = self.memory.sample(100)
            self.learn(batch_state,batch_next_state,batch_reward,batch_action)
        
        #Updation
        self.last_action = action
        self.last_state = new_state
        self.last_reward = torch.LongTensor([last_reward])
        self.reward_window.append(last_reward)

        #Maintaining the reward_window size to 1000
        if len(self.reward_window) > 1000:
            del self.reward_window[0]

        return action
    
    #Calculating the score(mean of the rewards)
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+ 1.0)   

    #Saving the model
    def save(self):
        torch.save({
            'state_dict':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict()
        },'last_brain.pth')

    #Loading the model if it exits
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print('=> Loading checkpoint...')
            #Loading the existing brain
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('No checkpoint found!')







    
