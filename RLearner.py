import networkx as nx #for various graph parameters, such as eigenvalues, macthing number, etc
import random
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import load_model
from statistics import mean
import pickle
import time
import math
import matplotlib.pyplot as plt





class RLearner:


  def __init__(self, N, calcScore,
                learning_rate=0.0001, n_sessions =1000, agent = None,
                elite_percentile=93, super_percentile=94):

    self.calcScore = calcScore #reward function
    self.N = N #number of vertices in graph
    self.n_actions = 2 #The size of the alphabet. In this file we will assume this is 2. There are a few things we need to change when the alphabet size is larger,
        #such as one-hot encoding the input, and using categorical_crossentropy as a loss function.
    self.LR = learning_rate #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
    self.n_sessions =n_sessions #number of new sessions per iteration
    self.elite_percentile = elite_percentile #top 100-X percentile we are learning from
    self.super_percentile = super_percentile #top 100-X percentile that survives to next iteration



    # number of possible edges(jn)
    self.MYN = int(N*(N-1)/2)  #The length of the word we are generating. Here we are generating a graph, so we create a 0-1 word of length (N choose 2)

            
    self.observation_space = 2*self.MYN #Leave this at 2*MYN. The input vector will have size 2*MYN, where the first MYN letters encode our partial word (with zeros on
                  #the positions we haven't considered yet), and the next MYN bits one-hot encode which letter we are considering now.
                  #So e.g. [0,1,0,0,      0,0,1,0] means we have the partial word 01 and we are considering the third letter now.
                  #Is there a better way to format the input to make it easier for the neural network to understand things?


                  
    state_dim = (self.observation_space,)


    if not agent:
      FIRST_LAYER_NEURONS = 128 #Number of neurons in the hidden layers.
      SECOND_LAYER_NEURONS = 64
      THIRD_LAYER_NEURONS = 4


      #Model structure: a sequential network with three hidden layers, sigmoid activation in the output.
      #I usually used relu activation in the hidden layers but play around to see what activation function and what optimizer works best.
      #It is important that the loss is binary cross-entropy if alphabet size is 2.

      self.agent = Sequential()
      self.agent.add(Dense(FIRST_LAYER_NEURONS,  activation="relu"))
      self.agent.add(Dense(SECOND_LAYER_NEURONS, activation="relu"))
      self.agent.add(Dense(THIRD_LAYER_NEURONS, activation="relu"))
      self.agent.add(Dense(1, activation="sigmoid"))


    self.agent.build((None, self.observation_space))
    self.agent.compile(loss="binary_crossentropy", optimizer=SGD(learning_rate = self.LR)) #Adam optimizer also works well, with lower learning rate

    print(self.agent.summary())




  def generate_session(self, verbose = 1):
    """
    Play n_session games using agent neural network.
    Terminate when games finish 
    
    Code inspired by https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
    """
    n_sessions= self.n_sessions
    observation_space = self.observation_space
    MYN = self.MYN

    #Initiate Variables (JN)
    states =  np.zeros([n_sessions, observation_space, MYN], dtype=int)
    actions = np.zeros([n_sessions, MYN], dtype = int)
    state_next = np.zeros([n_sessions, observation_space], dtype = int)
    prob = np.zeros(n_sessions)
    states[:,MYN,0] = 1
    step = 0
    total_score = np.zeros([n_sessions])
    recordsess_time = 0
    play_time = 0
    scorecalc_time = 0
    pred_time = 0

    #loop until reach last step aka have considered last edge (JN)
    while (step<MYN):
      step += 1   
      tic = time.time()
      prob = self.agent.predict(states[:,:,step-1], batch_size = n_sessions) #have agent predict moves for current step for all n games (JN)
      pred_time += time.time()-tic
      
      for i in range(n_sessions):
        
        #if probability is larger than some random decimal, approve the action (add the edge) (JN)
        if np.random.rand() < prob[i]: 
          action = 1
        else:
          action = 0

        #save the action in action array (JN)
        actions[i][step-1] = action 


        tic = time.time() # start timer
        state_next[i] = states[i,:,step-1]
        play_time += time.time()-tic

        #if action approved, change graph-state (JN) 
        if (action > 0):
          state_next[i][step-1] = action    

        #change step indicator in state to next step (JN)
        state_next[i][MYN + step-1] = 0
        if (step < MYN):
          state_next[i][MYN + step] = 1 


        #if at last step, calculate reward and time it , leave loop(JN)         
        if step == MYN:
          tic = time.time() 
          total_score[i] = self.calcScore(state_next[i])
          scorecalc_time += time.time()-tic
        else: #if not at last step, record state (JN)
          tic = time.time()
          states[i,:,step] = state_next[i]      
          recordsess_time += time.time()-tic
    #end while (JN)

    #If you want, print out how much time each step has taken. This is useful to find the bottleneck in the program.    
    if (verbose):
      print("Predict: "+str(pred_time)+", play: " + str(play_time) +", scorecalc: " + str(scorecalc_time) +", recordsess: " + str(recordsess_time))
    
    return states, actions, total_score


  def select_elites(self, states_batch, actions_batch, rewards_batch): #%todo make this better
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i]

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions
    
    This function was mostly taken from https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
    If this function is the bottleneck, it can easily be sped up using numba
    """
    counter = self.n_sessions * (100.0 - self.elite_percentile) / 100.0
    reward_threshold = np.percentile(rewards_batch, self.elite_percentile)

    elite_states = []
    elite_actions = []
    elite_rewards = []

    
    for i in range(len(states_batch)):
      if rewards_batch[i] >= reward_threshold-0.0000001:    
        if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
          for item in states_batch[i]:
            elite_states.append(item.tolist())
          for item in actions_batch[i]:
            elite_actions.append(item)      
        counter -= 1
    elite_states = np.array(elite_states, dtype = int)  
    elite_actions = np.array(elite_actions, dtype = int)  

    return elite_states, elite_actions


  def select_super_sessions(self, states_batch, actions_batch, rewards_batch, percentile=90): #%todo make better
    """
    Select all the sessions that will survive to the next generation
    Similar to select_elites function
    If this function is the bottleneck, it can easily be sped up using numba
    """
    
    counter = self.n_sessions * (100.0 - self.super_percentile) / 100.0
    reward_threshold = np.percentile(rewards_batch,self.super_percentile)

    super_states = []
    super_actions = []
    super_rewards = []
    for i in range(len(states_batch)):
      if rewards_batch[i] >= reward_threshold-0.0000001:
        if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
          super_states.append(states_batch[i])
          super_actions.append(actions_batch[i])
          super_rewards.append(rewards_batch[i])
          counter -= 1
    super_states = np.array(super_states, dtype = int)
    super_actions = np.array(super_actions, dtype = int)
    super_rewards = np.array(super_rewards)
    return super_states, super_actions, super_rewards
  

  def run(self, log_file_suffix=None):
    #make empty variables
    super_states =  np.empty((0,self.MYN,self.observation_space), dtype = int)
    super_actions = np.array([], dtype = int)
    super_rewards = np.array([])
    sessgen_time = 0
    fit_time = 0
    score_time = 0


    if log_file_suffix==None:
      log_file_suffix= str(random.randint(0,1000)) #used in the filename

    for i in range(1000000): #1000000 generations should be plenty

      # generate new sessions
      # performance can be improved with joblib
      tic = time.time()
      sessions = self.generate_session() #change 0 to 1 to print out how much time each step in generate_session takes 
      sessgen_time = time.time()-tic
      tic = time.time()
      
      # seperate states, actions, rewards into seperate arrays (JN)
      states_batch = np.array(sessions[0], dtype = int)
      actions_batch = np.array(sessions[1], dtype = int)
      rewards_batch = np.array(sessions[2])
      states_batch = np.transpose(states_batch,axes=[0,2,1])

      #Add in the super 'sessions' from last round to this round's sessions(JN)
      states_batch = np.append(states_batch,super_states,axis=0)
      if i>0:
        actions_batch = np.append(actions_batch,np.array(super_actions),axis=0) 
      rewards_batch = np.append(rewards_batch,super_rewards)
        
      randomcomp_time = time.time()-tic 
      tic = time.time()


      #select 'elite' sessions. These sessions will be used to train model (JN)
      elite_states, elite_actions = self.select_elites(states_batch, actions_batch, rewards_batch) #pick the sessions to learn from
      select1_time = time.time()-tic


      #get 'super' sessions. These sessions will be added to the newly generated sessions in the next round (JN)
      tic = time.time()
      super_sessions = self.select_super_sessions(states_batch, actions_batch, rewards_batch) #pick the sessions to survive
      select2_time = time.time()-tic
      
      tic = time.time()
      super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
      super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)
      select3_time = time.time()-tic
      

      # Train model on 'elite' sessions (JN)
      tic = time.time()
      self.agent.fit(elite_states, elite_actions) #learn from the elite sessions
      fit_time = time.time()-tic
      
      
      tic = time.time()
      
      super_states = [super_sessions[i][0] for i in range(len(super_sessions))]
      super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
      super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]
      
      rewards_batch.sort()
      mean_all_reward = np.mean(rewards_batch[-100:]) 
      mean_best_reward = np.mean(super_rewards) 

      score_time = time.time()-tic
      
      print("\n" + str(i) +  ". Best individuals: " + str(np.flip(np.sort(super_rewards))))
      
      #uncomment below line to print out how much time each step in this loop takes. 
      print(  "Mean reward: " + str(mean_all_reward) + "\nSessgen: " + str(sessgen_time) + ", other: " + str(randomcomp_time) + ", select1: " + str(select1_time) + ", select2: " + str(select2_time) + ", select3: " + str(select3_time) +  ", fit: " + str(fit_time) + ", score: " + str(score_time)) 
      
      
      if (i%20 == 1): #Write all important info to files every 20 iterations
        with open('best_species_pickle_'+ log_file_suffix +'.txt', 'wb') as fp:
          pickle.dump(super_actions, fp)
        with open('best_species_txt_'+log_file_suffix+'.txt', 'w') as f:
          for item in super_actions:
            f.write(str(item))
            f.write("\n")
        with open('best_species_rewards_'+ log_file_suffix+'.txt', 'w') as f:
          for item in super_rewards:
            f.write(str(item))
            f.write("\n")
        with open('best_100_rewards_'+log_file_suffix+'.txt', 'a') as f:
          f.write(str(mean_all_reward)+"\n")
        with open('best_elite_rewards_'+log_file_suffix+'.txt', 'a') as f:
          f.write(str(mean_best_reward)+"\n")
      if (i%200==2): # To create a timeline, like in Figure 3
        with open('best_species_timeline_txt_'+log_file_suffix+'.txt', 'a') as f:
          f.write(str(super_actions[0]))
          f.write("\n")
      




