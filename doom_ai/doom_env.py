
from vizdoom import *
import numpy as np
import matplotlib.pyplot as plt
from gym import Env
from gym.spaces import Discrete, Box
import cv2

class VizDoomGym(Env):
    def __init__(self,render=True,height=64,width=64,gray_scale=True):
        super().__init__()
        self.game = DoomGame()
        self.game.load_config('viz/ViZDoom/scenarios/deadly_corridor.cfg')
        self.img_height = height
        self.img_width = width
        self.gray_scale = gray_scale

        #Actions possible(left ,right,shoot)
        self.actions = np.eye(7,dtype=np.uint8)
        self.no_actions = self.actions.shape[0]
        
        if render:
            self.game.set_window_visible(True)
        else:
            self.game.set_window_visible(False)
            
        self.game.init()
        color_channel = 1 if gray_scale else 3
        
        #Creating action and observation space
        self.observation_space = Box(low=0,high=255,shape=(color_channel,height,width),dtype=np.uint8)
        self.action_space = Discrete(7)
             
        
    def step(self,action):
        reward = self.game.make_action(self.actions[action],4)  
        info = {"ammo":0}

        done = self.game.is_episode_finished()
 
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.to_gray_scale(state) if self.gray_scale else state
            info["ammo"] = self.game.get_state().game_variables[0]
        else:
            state = np.zeros(self.observation_space.shape,dtype=np.uint8)
 
        return state,reward,done,info
    
    def close(self):
        self.game.close()
        
    def render(self):
        pass
    
    def to_gray_scale(self,observation):
        #Rearranging the image array for the cv2 requirements
        gray_img = cv2.cvtColor(np.moveaxis(observation,0,-1),cv2.COLOR_BGR2GRAY)
        state = cv2.resize(gray_img,(self.img_width,self.img_height),interpolation = cv2.INTER_CUBIC)
        state = np.reshape(state,(1,self.img_height,self.img_width))
        return state
    
    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        if self.gray_scale:
            return self.to_gray_scale(state)
        return state
