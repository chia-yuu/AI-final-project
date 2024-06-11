import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import gym
import dataset
from gym import spaces, core
from tqdm import tqdm
import os


BATCH_SIZE = 32
CAPACITY = 10000

from collections import namedtuple
Tr = namedtuple('tr', ('name_a', 'value_b'))
Tr_object = Tr('是名稱A', 100)

from collections import namedtuple
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

#設定常數
ENV = 'Movieselect-v0'  # 使用的課題名稱
GAMMA = 0.99  # 時間折扣率
MAX_STEPS = 200  # 1回合的step數
NUM_EPISODES = 500  # 最大執行回合數


class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  #記憶體最大長度
        self.memory = []  # 儲存經驗的變數
        self.index = 0  # 儲存index的變數

    def push(self, state, action, state_next, reward): #把transition = (state,action,state_next,reward)存於記憶體
        if len(self.memory) < self.capacity: #若記憶體還有剩的空間
            self.memory.append(None) 

        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity  #讓Index遞增1

    def sample(self, batch_size): #依照batch_size的大小，隨機取出儲存的內容
        return random.sample(self.memory, batch_size)

    def __len__(self): #把變數memory目前的長度傳給函數Len
        return len(self.memory)
    
    
class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY) #產生可以記憶經驗的記憶體

        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001) #設定最佳化手法

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        self.model.eval()

        state_action_values = self.model(state_batch).gather(1, action_batch)

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,batch.next_state)))
        
        next_state_values = torch.zeros(BATCH_SIZE)

        next_state_values[non_final_mask] = self.model(
            non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + GAMMA * next_state_values

        self.model.train()

        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()  # 重設梯度
        loss.backward()  # 反向傳播演算法
        self.optimizer.step()  # 更新連結參數

    def decide_action(self, state, episode): #依照目前的狀態決定動作
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()  # 將神經網路切換成計算模式
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
        else:
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])
        return action

class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)  #agent的腦袋

    def update_q_function(self): #更新Q函數
        self.brain.replay()

    def get_action(self, state, episode): #決定動作
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward): #將state,action,state_next,reward存入memory
        self.brain.memory.push(state, action, state_next, reward)

class Environment:
    
    databaserating = dataset.load_data()
    databaseuser = dataset.load_user()
    databasemovie = dataset.load_movie()
    
    def __init__(self):
        self.action_space = spaces.Discrete(5,start=1) 
        self.observation_space = spaces.Box(low=np.array([1,1,0,10,1,999]), high=np.array([943,1682,1,60,21,99999]), dtype=np.int64)
        num_states = self.observation_space.shape[0] 
        num_actions = self.action_space.n
        self.agent = Agent(num_states, num_actions)
        
    def run(self):
        '''実行'''
        complete_episodes = 0  # 連續195step成功的回合數量
        episode_final = False  # 最後一回合的旗標
        ret = []

        for episode in tqdm(range(NUM_EPISODES)):
            
            observation = np.random.uniform(low=np.array([1,1,0,10,1,999]), high=np.array([943,1682,1,60,21,99999]), size=(6,)).astype(np.int64)
            done = False
            state = torch.from_numpy(observation).type(torch.FloatTensor)  # NumPy變數轉換成Pytorch的張量
            state = torch.unsqueeze(state, 0)  # size n 變成 size 1xn
            r = 0

            for step in range(MAX_STEPS):
                    action = self.agent.get_action(state, episode)  # 求出動作
                    
                    current_userid = observation[0]
                    current_movieid = observation[1]
                    
                    next_movie = dataset.find_next_movie(current_userid, current_movieid)
                    
                    if (next_movie == 1700):
                        state = torch.from_numpy(observation).type(torch.FloatTensor)  # NumPy變數轉換成Pytorch的張量
                        state = torch.unsqueeze(state, 0)
                        observation[0] = observation[0]+1
                        user_info = dataset.find_userinfo(observation[0])
                        observation[2] =  user_info[2]
                        observation[3] = user_info[1]
                        observation[4] = user_info[3]
                        observation[5] = user_info[4]
                        
                    else:
                        observation[1] = next_movie
                        
                    observation_next =  observation
                    
                    real_rating = dataset.find_rating(current_userid,current_movieid)
                    
                    if(real_rating == "none"):
                        k = dataset.find_average_rating(int(current_userid))
                        d = abs(k - int(action))
                        complete_episodes = 0
                        
                    else:
                        d = abs(int(real_rating) - int(action))

                    reward = torch.FloatTensor([0.0])
                    if d == 0: 
                        reward = torch.FloatTensor([3.0])
                        complete_episodes = complete_episodes + 3
                    elif d == 1:
                        reward = torch.FloatTensor([1.0])
                        complete_episodes = complete_episodes + 1
                    elif d == 2: 
                        reward = torch.FloatTensor([0.0])
                        complete_episodes = complete_episodes
                    elif d == 3:
                        reward = torch.FloatTensor([-1.0])
                        complete_episodes = 0
                    elif d == 4:
                        reward = torch.FloatTensor([-3.0])
                        complete_episodes = 0
                            
                    if(step > 195 or int(reward)<0 ):
                        done = True  

                    if done:
                        torch.save(self.agent.brain.model.state_dict(), "cj_DQN.pt")
                        ret.append(r)
                        break
                    else:
                        state_next = observation_next
                        state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                        state_next = torch.unsqueeze(state_next, 0)
                        r += reward

                    self.agent.memorize(state, action, state_next, reward)

                    self.agent.update_q_function()

                    state = state_next

            if complete_episodes >= 10:
                    #print('連續10次推薦成功')
                    episode_final = True
        return ret
     
    def recommend(self):
        self.agent.brain.model.load_state_dict(torch.load("DQN.pt"))
        user_id_input = int(input("Enter your user ID (1-943): "))                 
        while(True):
            movie_name_input = input("Please enter a movie name (enter 0 to exit): ")
            if(movie_name_input == "0"):   break
            
            movie_id_rec = dataset.find_movie_id(movie_name_input)

            if(movie_id_rec==-1):
                print("Invalid movie, please enter again.")
                continue  # invalid movie name (movie is not in the dataset)
            
            array = []
            for episode in range (1682):
                user_info_rec = dataset.find_userinfo(user_id_input)
                observation_rec = np.array([user_id_input,movie_id_rec,user_info_rec[2],user_info_rec[1],user_info_rec[3],user_info_rec[4]])
                state_rec = torch.from_numpy(observation_rec).type(torch.FloatTensor)  # NumPy變數轉換成Pytorch的張量
                state_rec = torch.unsqueeze(state_rec, 0)
                action_rec = self.agent.get_action(state_rec, episode)
                array.append(int(action_rec))
                
            max_index = np.argmax(array)
            print("you may like ",dataset.find_movie_name(int(max_index+1)))

def DQN():
    selectmovie_env = Environment()
    if(not os.path.isfile("DQN.pt")):
        for i in tqdm(range(3)):
            print(f"Training process {i}.")
            reward = selectmovie_env.run()
    selectmovie_env.recommend()

    # plot the result
    # x = list(range(len(reward)))
    # plt.plot(x, reward)
    # plt.show()

if __name__ == "__main__":
    DQN()
