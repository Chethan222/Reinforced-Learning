from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color,Line
from kivy.config import Config
from kivy.properties import NumericProperty,ReferenceListProperty,ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
import numpy as np
import matplotlib.pyplot as plt

#Importing Dqn object from ai.py file
from ai import DeepQNetwork


#Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse','mouse,multitouch_on_demand')

#Adding last_x,last_y used to keep last point in memory when we draw the sand map on the graph
last_x = 0
last_y = 0

#Total no of points in the last drawing
n_points = 0 

#The length of the last drawing
length = 0 

#Getting our AI,which we call "brain",and that containes our neural network that represents Q-function
# 5 sensors(state variables), 3 actions(left,right,straight),gamma = 0.9
brain = DeepQNetwork(5,3,0.9)

#action = 0 => no rotation,action = 1 => rotate 20 degree(right), action = 2 => rotate -20 degree(left)
action2rotation = [0,20,-20]

#Initializing the last reward 
last_reward = 0 

#Initializing the mean score curve (sliding window of the rewards) with respect to time
scores = [] 



   
#Initializing the last distance 
last_distance = 0


#Initializing the map
#Using this trick to initialize the map only once
FIRST = True

def init():
    #Sand is an array that has as many cells as our graphic interface has pixels.
    global sand 

    #Goal Co-ordinates
    #x-coordinate of the goal
    global goal_x 

    #y-coordinate of the goal
    global goal_y

    #Initializing the sand array with only zeros
    sand = np.zeros((longueur,largeur))
    print('Sand   ',longueur,largeur)

    #The goal to reach is upper left of the  map
    goal_x = 20  
    goal_y = largeur - 20

    global FIRST
    FIRST = False
 

#Creating car class
class Car(Widget):
    #Initializing the car properties
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)   
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    #Sensor for detecting the sand infront of the car
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)

    #Sensor for detecting the sand at the left of the car
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x,sensor2_y)

    #Sensor for detecting the sand at the right of the car
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)

    #Signal received by each of the sensor (signal is the density of sand arround the sensor)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self,rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30,0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30,0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30,0).rotate((self.angle-30)%360) + self.pos
        #Calculating the signal density
        #Between area of 20*20 => 400pixesls
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10,int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.0
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10,int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.0
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10,int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.0
        
        #10 -- longeur-10
        #10 -- largeur-10
        #Setting the signals when car gets closer to the walls
        
        if(self.sensor1_x > longueur -10) or (self.sensor1_x < 10) or (self.sensor1_y > largeur -10) or (self.sensor1_y < 10):
            #sensor1 detects full sand
            self.signal1 = 1.0
            
        if(self.sensor2_x > longueur -10) or (self.sensor2_x < 10) or (self.sensor2_y > largeur -10) or (self.sensor2_y < 10):
            #sensor2 detects full sand
            self.signal2 = 1.0 
            
        if self.sensor3_x > longueur -10 or self.sensor3_x < 10 or self.sensor3_y > largeur -10 or self.sensor3_y < 10:
            #sensor3 detects full sand
            self.signal3 = 1.0 
 
#Sensor1    
class Ball1(Widget): 
    pass 
           
#Sensor2   
class Ball2(Widget): 
    pass   
         
#Sensor3    
class Ball3(Widget): 
    pass            

#creating the game class 
class Game(Widget):
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    
    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6,0)
        
    def update(self,dt):
        
        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        
        longueur = self.width
        largeur = self.height
        
        #Updating the game for the first time
        if FIRST:
            init()
        
        #Difference of x-coordinates of goal and car points
        xx = goal_x - self.car.x

        #Difference of y-coordinates of goal and car points
        yy = goal_y - self.car.y

        #Direction of the car with respect to the goal and the car
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.0

        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation] 

        #Upating the brain with reward and signal and getting corresponding action
        action = brain.update(last_reward,last_signal)

        print('Last Reward : ',last_reward)
        #Getting score 
        scores.append(brain.score())

        #selecting the action from the rotation list
        rotation = action2rotation[action]

        self.car.move(rotation)
        
        #Getting the distace between the car and the goal point
        goalDistance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)

        #Setting ball positions as the sensor position in the game(simulation)
        self.ball1.pos = self.car.sensor1   
        self.ball2.pos = self.car.sensor2   
        self.ball3.pos = self.car.sensor3   
        
        #Checking whether the car is on the san or not
        print('sand[int(self.car.x),int(self.car.y)]',sand[int(self.car.x),int(self.car.y)])
        if sand[int(self.car.x),int(self.car.y)] > 0:
            #Slowing down the car if the car is on the sand
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)

            #Gets bad reward
            last_reward = -1    
        else:
            #Increasing the car's speed if it's not on the sand
            self.car.velocity = Vector(6,0).rotate(self.car.angle)

            #Gets bad reward
            last_reward = 0.5
            
            #Gets positive reward if it approaches the goal
            if goalDistance < last_distance:
                last_reward = 0.3 
        
        #Handling car's border conditions an giving -nagative rewards
        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
            
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
            
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
            
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        #Reversing the goal when it reaches the goal 
        if goalDistance < 100:  
            goal_x = self.width - goal_x  
            goal_y = self.height - goal_y 
        
        #Updating the last distance
        last_distance = goalDistance     

#Painting for graphic interfaces
class PaintingWidget(Widget):
    
            
    def on_touch_down(self,touch):
        global length, n_points,last_x,last_y
        
        with self.canvas:
            Color(0.8,0.7,0)
            touch.ud['line'] = Line(points = (touch.x,touch.y),width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            try:
                sand[int(touch.x),int(touch.y)] = 1
            except IndexError:
                pass

    
    def on_touch_move(self,touch):
        global length, n_points, last_x, last_y
        
        if(touch.button == 'left'):
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y 

#API ans Switches interface
class CarApp(App):
    
    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/30.0)
        self.painter = PaintingWidget()
        clearBtn = Button(text = 'Clear')
        saveBtn = Button(text = 'Save',pos = (parent.width,0))
        loadBtn = Button(text = 'Load',pos = (2 * parent.width,0))

        clearBtn.bind(on_release = self.clear_canvas)
        saveBtn.bind(on_release = self.save)
        loadBtn.bind(on_release = self.load)

        parent.add_widget(self.painter)
        parent.add_widget(clearBtn)
        parent.add_widget(saveBtn)
        parent.add_widget(loadBtn)

        return parent
    
    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))
        
    def save(self, obj):
        print("Saving brain...\n")
        brain.save()
        plt.plot(scores)
        plt.show()
        
    def load(self,obj):
        print("Loading last saved brain...\n")
        brain.load()    
        


if __name__ == '__main__':
    CarApp().run()       
        
        
        
                
                        
            
          
            
                      