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


BATCH_SIZE = 32
CAPACITY = 10000

#使用namedtuple，即可同時儲存值與對應的欄位名稱
from collections import namedtuple
Tr = namedtuple('tr', ('name_a', 'value_b'))
Tr_object = Tr('是名稱A', 100)
#print(Tr_object) 
#print(Tr_object.value_b) 

#產生nametuple
from collections import namedtuple
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

#設定常數
ENV = 'Movieselect-v0'  # 使用的課題名稱
GAMMA = 0.99  # 時間折扣率
MAX_STEPS = 200  # 1回合的step數
NUM_EPISODES = 500  # 最大執行回合數


# 定義儲存經驗的記憶體class
class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  #記憶體最大長度
        self.memory = []  # 儲存經驗的變數
        self.index = 0  # 儲存index的變數

    def push(self, state, action, state_next, reward): #把transition = (state,action,state_next,reward)存於記憶體
        if len(self.memory) < self.capacity: #若記憶體還有剩的空間
            self.memory.append(None) 

        #使用namedtuple的Transition儲存值與對應的欄位名稱
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity  #讓Index遞增1

    def sample(self, batch_size): #依照batch_size的大小，隨機取出儲存的內容
        return random.sample(self.memory, batch_size)

    def __len__(self): #把變數memory目前的長度傳給函數Len
        return len(self.memory)
    
    
# 人工智慧的大腦，可以執行DQN
class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # 取得CartPole的動作
        self.memory = ReplayMemory(CAPACITY) #產生可以記憶經驗的記憶體

        #建置神經網絡
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))

        #print(self.model)  #輸出神經網路的形狀
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001) #設定最佳化手法

    def replay(self):
        #利用Experience Replay 學習神經網路的連結參數
        
        # -----------------------------------------
        # 1.確認記憶體大小
        # -----------------------------------------
        # 1.1 記憶體大小過小，什麼都不執行
        if len(self.memory) < BATCH_SIZE:
            return

        # -----------------------------------------
        # 2.建立小批次資料
        # -----------------------------------------
        # 2.1 從記憶體中取得資料
        transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 格式轉換
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # -----------------------------------------
        # 3. 計算指令訊號的Q(s_t, a_t)値
        # -----------------------------------------
        # 3.1 讓神經網路切換成推論模式
        self.model.eval()

        # 3.2 計算神經網路輸出的Q(s_t,a_t) ???我還是不懂gather怎麼做的QAQ
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # 3.3 計算max{Q(s_t+1, a)}的值，不過要注意是否還有下個狀態 ???這行也看不懂QAQ
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,batch.next_state)))
        
        # 初始化所有狀態為0
        next_state_values = torch.zeros(BATCH_SIZE)

        # 接著計算下一個狀態的index的最大Q值，存取輸出值，以max(1)計算欄方向的最大值[值、Index]，接著輸出該Q值(index=0)
        #以detach取出該值
        #還是不懂??????QAQ QAQ QAQ QAQ QAQ QAQ
        next_state_values[non_final_mask] = self.model(
            non_final_next_states).max(1)[0].detach()

        # 3.4 根據Q學習的公式學習指令訊號Q(s_t,a_t)
        expected_state_action_values = reward_batch + GAMMA * next_state_values

        # -----------------------------------------
        # 4. 更新連結參數
        # -----------------------------------------
        # 4.1 將神經網路切換成訓練模式
        self.model.train()

        # 4.2 計算損失函數（smooth_l1_loss的Huberloss）
        # expected_state_action_values 的size已是minbatch，所以利用unsqueeze遞增為minibatch*1
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # 4.3 更新連結參數
        self.optimizer.zero_grad()  # 重設梯度
        loss.backward()  # 反向傳播演算法
        self.optimizer.step()  # 更新連結參數

    def decide_action(self, state, episode): #依照目前的狀態決定動作
        # 以ε-greedy法採用最佳動作
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()  # 將神經網路切換成計算模式
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
            # 取得神經網路的最大輸出的Index=max(1)[1]
            #.view(1,1) 會將[torch.LongTensor of size 1]轉換成size 1*1
        else:
            # 隨機回傳0,1的動作
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
        self.action_space = spaces.Discrete(5,start=1) #defines a one-dimensional continuous action space where the action can take any value between 1 and 5
        #observation_low = np.array([1,1,0,1,1,0,1])
        #observation_high = np.array([944,1683,2,100,21,99999,19])
        #self.observation_space = spaces.Box(observation_low,observation_high,shape=(6,))
        self.observation_space = spaces.Box(low=np.array([1,1,0,10,1,999]), high=np.array([943,1682,1,60,21,99999]), dtype=np.int64)
        num_states = self.observation_space.shape[0] 
        num_actions = self.action_space.n
        self.agent = Agent(num_states, num_actions)
        
    def run(self):
        '''実行'''
        complete_episodes = 0  # 連續195step成功的回合數量
        episode_final = False  # 最後一回合的旗標

        for episode in tqdm(range(NUM_EPISODES)):
        #for episode in range(NUM_EPISODES):  #episode = 回合數
            
            # 環境初始化
            observation = np.random.uniform(low=np.array([1,1,0,10,1,999]), high=np.array([943,1682,1,60,21,99999]), size=(6,)).astype(np.int64)
            done = False
            #state = observation #將觀測結果直接當成狀態S使用
            state = torch.from_numpy(observation).type(torch.FloatTensor)  # NumPy變數轉換成Pytorch的張量
            state = torch.unsqueeze(state, 0)  # size n 變成 size 1xn

            for step in range(MAX_STEPS):  # 單一回合的迴圈
                    
                    #if episode_final is True: #最後一回合
                        #print(state,action,state_next,reward)
                        #print("end?")

                    action = self.agent.get_action(state, episode)  # 求出動作
                    
                    # 執行動作a_t之後，算出s_t+1，以及是否done
                    current_userid = observation[0]
                    current_movieid = observation[1]
                    
                    next_movie = dataset.find_next_movie(current_userid, current_movieid)
                    
                    if (next_movie == 1700):
                        #state = observation
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
                    #print(observation_next)
                    
                    real_rating = dataset.find_rating(current_userid,current_movieid)
                    
                    if(real_rating == "none"):
                        k = dataset.find_average_rating(int(current_userid))
                        d = abs(k - int(action))
                        complete_episodes = 0
                        
                    else:
                        d = abs(int(real_rating) - int(action))

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
                        break
                        #state_next = None
                        #episode_10_list = np.hstack(
                        #    (episode_10_list[1:], step + 1))

                    else:
                        #reward = torch.FloatTensor([0.0])
                        state_next = observation_next
                        state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                        state_next = torch.unsqueeze(state_next, 0)

                    # 將學習經驗存入記憶體
                    self.agent.memorize(state, action, state_next, reward)

                    # 以Experience Replay 更新Q函數
                    self.agent.update_q_function()

                    # 觀測狀態更新
                    state = state_next

                    # 結束時的處理
                    #if done:
                        #print('%d Episode: Finished after %d steps：10試行の平均STEP数 = %.1lf' % (
                        #    episode, step + 1, episode_10_list.mean()))
                        #break

                # 10連続で200step経ち続けたら成功
            if complete_episodes >= 10:
                    #print('連續10次推薦成功')
                    episode_final = True  # 次の試行を描画を行う最終試行とする
     
    def recommend(self):
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
            print("you may like ",dataset.find_movie_name(int(max_index)))

def DQN():
    selectmovie_env = Environment()
    selectmovie_env.run()
    selectmovie_env.recommend()

if __name__ == "__main__":
    DQN()