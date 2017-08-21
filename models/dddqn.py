# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:38:25 2017

@author: nivradmin
"""


import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework import get_variables
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #so that TF doesn't show its warnings
import math
import sys
#====own classes====
from myprint import myprint as print
from utils import convolutional_layer, fc_layer, variable_summary, netCopyOps



class DuelDQN():
    
    ############################BUILDING THE COMPUTATION GRAPH#################
    def __init__(self, conf, agent, name, isInference=False, isPretrain=False):  
        self.conf = conf
        self.agent = agent
        self.name = name  
        self.isInference = isInference
        self.isPretrain = isPretrain
        self.pretrain_episode = 0
        self.run_inferences = 0
        self.step = 0
        self.h_size = 256
        self.stood_frames_ago = 0 #das wird benutzt damit er, wenn er einmal stand, sich merken kann ob erst kurz her ist (für settozero)
        with tf.variable_scope(name, reuse=None):
            self.pretrain_episode_tf = tf.Variable(tf.constant(self.pretrain_episode), dtype=tf.int32, name='pretrain_episode_tf', trainable=False)
            self.pretrain_step_tf = tf.Variable(tf.constant(0), dtype=tf.int32, name='pretrain_step_tf', trainable=False) #diese ist hier nur zum backupen
            self.step_tf = tf.Variable(tf.constant(0), dtype=tf.int32, name='step_tf', trainable=False)
            self.run_inferences_tf = tf.Variable(tf.constant(self.run_inferences), dtype=tf.int32, name='run_inferences_tf', trainable=False) #diese ist hier nur zum backupen
            
            self.num_actions = self.conf.steering_steps*4 if self.conf.INCLUDE_ACCPLUSBREAK else self.conf.steering_steps*3
            self.conv_stacksize = (self.conf.history_frame_nr*2 if self.conf.use_second_camera else self.conf.history_frame_nr) if self.agent.conv_stacked else 1
            self.ff_stacksize = self.conf.history_frame_nr if self.agent.ff_stacked else 1
            
            #THIS IS FORWARD STEP
            self.phase = tf.placeholder(tf.bool, name='phase') #for batchnorm
            self.conv_inputs = tf.placeholder(tf.float32, shape=[None, self.conf.image_dims[0], self.conf.image_dims[1], self.conv_stacksize], name="conv_inputs")  if self.agent.usesConv else None
            self.ff_inputs = tf.placeholder(tf.float32, shape=[None, self.ff_stacksize*self.agent.ff_inputsize], name="ff_inputs")  if self.agent.ff_inputsize else None
            self.stands_input = tf.placeholder(tf.bool, name="stands_input") #necessary for settozero            
            self.Qout, self.Qmax, self.predict = self._inference(self.conv_inputs, self.ff_inputs, self.stands_input, self.phase)
#            self.Qout = fc_layer(self.ff_inputs, self.agent.ff_inputsize, self.num_actions, "FC0", True, False, False, False, tf.nn.relu, 1, {}, variable_summary, initializer=tf.random_normal_initializer(0, 1e-100))               
#            self.Qmax = tf.reduce_max(self.Qout, axis=1) 
#            self.predict = tf.argmax(self.Qout,1)            
            
            
            #THIS IS SV_LEARN 
            self.targetA = tf.placeholder(shape=[None],dtype=tf.int32)
            self.targetA_OH = tf.one_hot(self.targetA, self.num_actions, dtype=tf.float32)
            self.sv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.targetA_OH, logits=self.Qout))
            self.sv_OP = self._sv_training(self.sv_loss)  
            
            
            #THIS IS DDQN-LEARN
            #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
            self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
            #self.targetA = tf.placeholder(shape=[None],dtype=tf.int32)
            #self.targetA_OH = tf.one_hot(self.targetA, self.num_actions, dtype=tf.float32)
            self.compareQ = tf.reduce_sum(tf.multiply(self.Qout, self.targetA_OH), axis=1) #der td_error von den actions über die wir nicht lernen wollen ist null
            self.td_error = tf.square(self.targetQ - self.compareQ) 
            self.q_loss = tf.reduce_mean(self.td_error)
            self.q_OP = self._q_training(self.q_loss, isPretrain)
            
        
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.saver = tf.train.Saver(var_list=get_variables(name))
        

            
    def _inference(self, conv_inputs, ff_inputs, stands_input, is_training):
        assert (conv_inputs is not None or ff_inputs is not None)
#        ini = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.conf.image_dims[0]*self.conf.image_dims[1])))
        ini = tf.random_normal_initializer(0, 1e-3)
        do_batchnorm = False        

        if conv_inputs is not None:
            rs_input = tf.reshape(conv_inputs, [-1, self.conf.image_dims[0], self.conf.image_dims[1], self.conv_stacksize]) #final dimension = number of color channels*number of stacked (history-)frames                  
            #convolutional_layer(input_tensor, input_channels, kernel_size, stride, output_channels, name, act, is_trainable, batchnorm, is_training, weightdecay=False, pool=True, trainvars=None, varSum=None, initializer=None)
            self.conv1 = convolutional_layer(rs_input, self.conv_stacksize, [4,6], [2,3], 32, "Conv1", tf.nn.relu, True, do_batchnorm, is_training, False, False, {}, variable_summary, initializer=ini) #(?, 14, 14, 32)
            self.conv2 = convolutional_layer(self.conv1, 32, [4,4], [2,2], 64, "Conv2", tf.nn.relu, True, do_batchnorm, is_training, False, False, {}, variable_summary, initializer=ini)                     #(?, 8, 8, 64)
            self.conv3 = convolutional_layer(self.conv2, 64, [3,3], [2,2], 64, "Conv3", tf.nn.relu, True, do_batchnorm, is_training, False, True, {}, variable_summary, initializer=ini)                      #(?, 2, 2, 64)
            self.conv4 = convolutional_layer(self.conv3, 64, [4,4], [2,2], self.h_size, "Conv4", tf.nn.relu, True, do_batchnorm, is_training, False, False, {}, variable_summary, initializer=ini)            #(?, 1, 1, 256)
            self.conv4_flat = tf.reshape(self.conv4, [-1, self.h_size])
        
        if ff_inputs is not None:
            fc_in = fc_layer(ff_inputs, self.ff_stacksize*self.agent.ff_inputsize, self.ff_stacksize*self.agent.ff_inputsize, "FC0", True, do_batchnorm, is_training, False, tf.nn.relu, 1, {}, variable_summary, initializer=ini)   

        if conv_inputs is not None and ff_inputs is not None:
            fc0 = tf.concat([self.conv4_flat, fc_in], 1)
        elif conv_inputs is not None:
            fc0 = self.conv4_flat
        else:
            fc0 = fc_in
        
        length = fc0.get_shape()[1]
        fc1 = fc_layer(fc0, length, self.h_size*2, "FC1", True, do_batchnorm, is_training, False, tf.nn.relu, 1, {}, variable_summary, initializer=ini)                 

        #Dueling DQN: split into separate advantage and value stream
        self.streamA,self.streamV = tf.split(fc1,2,1) 
        xavier_init = tf.contrib.layers.xavier_initializer()
        neutral_init = tf.random_normal_initializer(0, 1e-50)
        self.AW = tf.Variable(xavier_init([self.h_size,self.num_actions]))
        self.VW = tf.Variable(neutral_init([self.h_size,1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        if do_batchnorm:
            self.Advantage = tf.contrib.layers.batch_norm(self.Advantage, center=True, scale=True, is_training=is_training)
        self.Value = tf.matmul(self.streamV,self.VW)
        if do_batchnorm:
            self.Value = tf.contrib.layers.batch_norm(self.Value, center=True, scale=True, is_training=is_training)
        Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        if do_batchnorm:
            Qout = tf.contrib.layers.batch_norm(Qout, center=True, scale=True, is_training=is_training)

        def settozero(q):
            ZEROIS = -sys.maxsize-1 if self.agent.isSupervised else 0
            q = tf.squeeze(q) #stands_input ist nur dann True wenn es nur um ein sample geht
            if not self.conf.INCLUDE_ACCPLUSBREAK: #dann nimmste nur das argmax von den mittleren neurons (was die mit gas sind)
                q = tf.slice(q,tf.shape(q)//3,tf.shape(q)//3)
                q = tf.concat([tf.multiply(tf.ones(tf.shape(q)),ZEROIS), q, tf.multiply(tf.ones(tf.shape(q)), ZEROIS)], axis=0)
            else:
                q = tf.slice(q,tf.shape(q)//2,(tf.shape(q)//4)*3)
                q = tf.concat([tf.multiply(tf.ones(tf.shape(q)*2), ZEROIS), q, tf.multiply(tf.ones(tf.shape(q)), ZEROIS)], axis=0)                   
            q = tf.expand_dims(q, 0)            
            return q        
            
        if self.isInference:
            Qout = tf.cond(self.stands_input, lambda: settozero(Qout), lambda: Qout) #wenn du stehst, brauchste dich nicht mehr für die ohne gas zu interessieren

        Qmax = tf.reduce_max(Qout, axis=1) #not necessary anymore because we use Double-Q, only used for the stateval for the evaluator
        predict = tf.argmax(Qout,1)

        return Qout, Qmax, predict 
    

    def _sv_training(self, loss): #svtrain is necessarily pretrain, but pretrain is not necessarily sv
        init_lr = self.conf.pretrain_sv_initial_lr
        self.sv_lr = tf.Variable(tf.constant(init_lr), trainable=False)
        self.sv_new_lr = tf.placeholder(tf.float32, shape=[])                   
        self.sv_lr_update = tf.assign(self.sv_lr, self.sv_new_lr)  
        trainer = tf.train.AdamOptimizer(self.sv_lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops): #because of batchnorm, see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
            train_op = trainer.minimize(loss, global_step=self.pretrain_step_tf)
        return train_op
        
        
    def _q_training(self, loss, isPretrain):
        if isPretrain:
            init_lr = self.conf.pretrain_q_initial_lr
            self.lr = tf.Variable(tf.constant(init_lr), trainable=False)
            self.new_lr = tf.placeholder(tf.float32, shape=[]) 
            self.lr_update = tf.assign(self.lr, self.new_lr)    
            q_trainer = tf.train.AdamOptimizer(learning_rate=self.lr)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops): #because of batchnorm, see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
                q_OP = q_trainer.minimize(loss, global_step=self.pretrain_step_tf) 
        else:
            init_lr = self.conf.initial_lr
            self.lr = tf.Variable(tf.constant(init_lr), trainable=False)
            self.new_lr = tf.placeholder(tf.float32, shape=[]) 
            self.lr_update = tf.assign(self.lr, self.new_lr)    
            q_trainer = tf.train.AdamOptimizer(learning_rate=self.lr)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops): #because of batchnorm, see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
                q_OP = q_trainer.minimize(loss, global_step=self.step_tf) 
        return q_OP
       
    ############################METHODS FOR RUNNING############################

    def save(self, session):
        folder = self.conf.pretrain_checkpoint_dir if self.isPretrain else self.conf.checkpoint_dir
        checkpoint_file = os.path.join(self.agent.folder(folder), 'model.ckpt')
        session.run(self.pretrain_episode_tf.assign(self.pretrain_episode))
        session.run(self.run_inferences_tf.assign(self.run_inferences))
        self.saver.save(session, checkpoint_file, global_step=self.pretrain_step_tf if self.isPretrain else self.step_tf)
        print("Saved Model.", level=6) 
        
        
    def load(self, session, from_pretrain=False):
         folder = self.conf.pretrain_checkpoint_dir if from_pretrain else self.conf.checkpoint_dir
         ckpt = tf.train.get_checkpoint_state(self.agent.folder(folder))
         if ckpt and ckpt.model_checkpoint_path:
             self.saver.restore(session, ckpt.model_checkpoint_path)
             print("Loaded",("from pretrain" if from_pretrain else "from RL-train"), level=10)
             self.pretrain_step = self.pretrain_step_tf.eval(session)
             self.pretrain_episode = self.pretrain_episode_tf.eval(session)
             self.step = self.step_tf.eval(session)
             self.run_inferences = self.run_inferences_tf.eval(session)
             print("Pretrain-Step:",self.pretrain_step, "Pretrain-Episode:",self.pretrain_episode,"Main-Step:",self.step, "Run'n Iterations:", self.run_inferences, level=10)
             return True
         else:
             print("Couldn't load", ("from pretrain" if from_pretrain else "from RL-train"), level=10)
             return False
        
        
    #carstands ist true iff (single inference & carstands), in jedem anderem Fall false
    def make_inputs(self, inputs, targetQ=None, targetA=None, carstands = False, decay_lr=False, is_training=True):
        conv_inputs = np.array([inputs[i][0] for i in range(len(inputs))])
        ff_inputs   = np.array([inputs[i][1] for i in range(len(inputs))])
        feed_dict = {self.phase: is_training}
        if not is_training and self.isInference:   
            self.stood_frames_ago = 0 if carstands else self.stood_frames_ago + 1
            if self.stood_frames_ago < 4: #wenn du vor einigen frames stands, gib jetzt auch gas
                carstands = True
        else:
            if targetQ is not None: #targetQ und targetA werden nur beim learning verwendet, und dann ist ebennicht inference
                feed_dict[self.targetQ] = targetQ
            if targetA is not None:
                feed_dict[self.targetA] = targetA
        if self.agent.usesConv:
            feed_dict[self.conv_inputs] = conv_inputs
        if self.agent.ff_inputsize:
            feed_dict[self.ff_inputs] = ff_inputs 
        feed_dict[self.stands_input] = carstands 
        if decay_lr == "sv":
            lr_decay = self.conf.pretrain_sv_lr_decay ** max(self.pretrain_episode-self.conf.pretrain_lrdecayafter, 0.0)
            new_lr = max(self.conf.pretrain_sv_initial_lr*lr_decay, self.conf.pretrain_sv_minimal_lr)
            feed_dict[self.sv_new_lr] = new_lr
        elif decay_lr == "q":
            if self.isPretrain:
                lr_decay = self.conf.pretrain_q_lr_decay ** max(self.pretrain_episode-self.conf.pretrain_lrdecayafter, 0.0)
                new_lr = max(self.conf.pretrain_q_initial_lr*lr_decay, self.conf.pretrain_q_minimal_lr)
                feed_dict[self.new_lr] = new_lr                 
            else:
                lr_decay = self.conf.lr_decay ** max(self.step-self.conf.lrdecayafter, 0.0)
                new_lr = max(self.conf.initial_lr*lr_decay, self.conf.minimal_lr)
                feed_dict[self.new_lr] = new_lr      
        return feed_dict
        
    


##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
        
        
        
class DDDQN_model():
    #this is the class for Double-Dueling-DQN, containing BOTH the online and the target DuelDQN-Network        
        
    def __init__(self, conf, agent, session, isPretrain=False):
        self.conf = conf
        self.agent = agent
        self.session = session        
        self.isPretrain = isPretrain
        self.onlineQN = DuelDQN(conf, agent, "onlineNet", isPretrain=isPretrain)
        self.targetQN = DuelDQN(conf, agent, "targetNet", isInference=(not isPretrain), isPretrain=isPretrain)        
        self.smoothTargetNetUpdate = netCopyOps(self.onlineQN, self.targetQN, self.conf.target_update_tau)
                
            
    def initNet(self, load=False):
        self.session.run(tf.global_variables_initializer())
        if load == "preTrain":
            self.targetQN.load(self.session, from_pretrain=True)     
            self.session.run(self.onlineQN.pretrain_step_tf.assign(self.targetQN.pretrain_step_tf))
        elif load == "noPreTrain":
            self.targetQN.load(self.session, from_pretrain=False)   
            self.session.run(self.onlineQN.step_tf.assign(self.targetQN.step_tf))
        elif load != False: #versuche RLLearn, wenn das nicht geht pretrain
            if self.targetQN.load(self.session, from_pretrain=False):
                self.session.run(self.onlineQN.step_tf.assign(self.targetQN.step_tf))
            else:
                self.targetQN.load(self.session, from_pretrain=True)     
                self.session.run(self.onlineQN.pretrain_step_tf.assign(self.targetQN.pretrain_step_tf))
        self.session.run(netCopyOps(self.targetQN, self.onlineQN))
        self.lastTrained = None
            
        
    def save(self):
        if self.lastTrained == self.onlineQN: #falls zuletzt q_train gemacht wurde
            self.session.run(self.targetQN.pretrain_step_tf.assign(self.onlineQN.pretrain_step_tf))
            self.session.run(self.targetQN.step_tf.assign(self.onlineQN.step_tf))
        self.targetQN.save(self.session)            
           
        
    def pretrain_episode(self):
        return self.targetQN.pretrain_episode
    def inc_episode(self):
        if self.isPretrain:
            self.targetQN.pretrain_episode += 1
            self.onlineQN.pretrain_episode += 1
    def step(self):
        return self.onlineQN.step #wird bei jedem qlearn-step ge-evalt
    def run_inferences(self):
        return self.targetQN.run_inferences #wird bei jeder inference increased
        
        
    #expects a whole s,a,r,s,t - tuple, needs however only s & a
    def getAccuracy(self, batch, likeDDPG=True):
        oldstates, actions, _, _, _ = batch
        predict = self.session.run(self.targetQN.predict,feed_dict=self.targetQN.make_inputs(oldstates, is_training=False))
        if likeDDPG:
            return np.mean(np.array([abs(predict[i][0] -actions[i][0]) for i in range(len(actions))]))
        else:
            return round(np.mean(np.array(actions == predict, dtype=int))*100, 2)
            
    #expects only a state (with stands_input)
    def inference(self, oldstates):                                                    
        assert not self.isPretrain, "Please reload this network as a non-pretrain-one!"
        self.targetQN.run_inferences += 1
        carstands = oldstates[0][2] if len(oldstates) == 1 and len(oldstates[0]) > 2 else False
        return self.session.run([self.targetQN.predict, self.targetQN.Qout], feed_dict=self.targetQN.make_inputs(oldstates, carstands = carstands, is_training=False))
        
        
    #expects only a state (and no stands_input)
    def statevalue(self, oldstates):                                                  
        return self.session.run(self.targetQN.Qmax, feed_dict=self.targetQN.make_inputs(oldstates, is_training=False))
    
    
    #expects a whole s,a,r,s,t - tuple, needs however only s & a
    def sv_train_step(self, batch, decay_lr = True):
        assert self.isPretrain, "Supervised-Learning is only allowed as Pre-training!"
        oldstates, actions, _, _, _ = batch
        if decay_lr:
            _, loss, _ = self.session.run([self.targetQN.sv_OP, self.targetQN.sv_loss, self.targetQN.sv_lr_update], feed_dict=self.targetQN.make_inputs(oldstates, targetA=actions, decay_lr="sv"))
        else:
            _, loss, _ = self.session.run([self.targetQN.sv_OP, self.targetQN.sv_loss], feed_dict=self.targetQN.make_inputs(oldstates, targetA=actions))
        self.lastTrained = self.targetQN
        #print("Learning rate:",self.session.run(self.targetQN.sv_lr))
        return loss
    
    
    #expects a whole s,a,r,s,t - tuple  
    def q_train_step(self, batch, decay_lr = False):
        oldstates, actions, rewards, newstates, terminals = batch
        action = self.session.run(self.onlineQN.predict,feed_dict=self.onlineQN.make_inputs(newstates)) #TODO: im text schreiben wie das bei non-doubleDQN anders wäre
        folgeQ = self.session.run(self.targetQN.Qout,feed_dict=self.targetQN.make_inputs(newstates)) #No reduceMax anymore, but instead the action-prediciton because DDQN: instead of taking the max over Q-values when computing the target-Q value for our training step, we use our primary network to chose an action, and our target network to generate the target Q-value for that action. 
        consider_stateval = -(terminals - 1)
        doubleQ = folgeQ[range(len(terminals)),action]  
        targetQ = rewards + (self.conf.q_decay * doubleQ * consider_stateval)
        #Update the network with our target values.
        if decay_lr:
            _, _ = self.session.run([self.onlineQN.q_OP, self.onlineQN.lr_update], feed_dict=self.onlineQN.make_inputs(oldstates, targetQ=targetQ, targetA=actions, decay_lr="q"))
        else:
            _ = self.session.run(self.onlineQN.q_OP, feed_dict=self.onlineQN.make_inputs(oldstates, targetQ=targetQ, targetA=actions))
        self.session.run(self.smoothTargetNetUpdate) #Update the target network toward the primary network.
        if not self.isPretrain:
            self.onlineQN.step = self.onlineQN.step_tf.eval(self.session)
        self.lastTrained = self.onlineQN
        #print("Learning rate:",self.session.run(self.onlineQN.lr))
        return
        
###########################################################################################################
###########################################################################################################
###########################################################################################################
    
#sind die noch richtig?        
############################## helper functions ###############################

#takes as input a batch of ENV-STATES, and returns batch of AGENT-STATES
#def EnvStateBatch_to_AgentStateBatch(self, agent, stateBatch):
#    presentStates = list(zip(*stateBatch))
#    conv_inputs, other_inputs, _ = list(zip(*[agent.getAgentState(*presentState) for presentState in presentStates]))
#    other_inputs = [agent.makeNetUsableOtherInputs(i) for i in other_inputs]
#    return conv_inputs, other_inputs, False
#    
##takes as input batch of ENV-STATES, and return batch of AGENT-ACTIONS
#def EnvStateBatch_to_AgentActionBatch(self, agent, stateBatch):
#    presentStates = list(zip(*stateBatch))
#    targets = [agent.makeNetUsableAction(agent.getAction(*presentState)) for presentState in presentStates]
#    return targets
#                      


def TPSample(conf, agent, batchsize):
    return read_supervised.create_QLearnInputs_from_PTStateBatch(*trackingpoints.next_batch(conf, agent, batchsize), agent)
    #returns [[s],[a],[r],[s2],[t]], so to only get the actions it's result[1], to only get the first three actions it's result[1][:3]
    #to get the first three of each it's result[:][:3], however pay attention that every state s and s2 = (conv, ff, stands). 
    #so to get the ff's of the first three items it is [result[0][:3][i][1] for i in range(3)]

###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == '__main__':       
    import config
    conf = config.Config()
    import read_supervised
    from server import Containers; containers = Containers()
    import dqn_rl_agent
    myAgent = dqn_rl_agent.Agent(conf, containers, True)
    trackingpoints = read_supervised.TPList(conf.LapFolderName, conf.use_second_camera, conf.msperframe, conf.steering_steps, conf.INCLUDE_ACCPLUSBREAK)
    
    BATCHSIZE = 32   

    #PRETRAINING:
#    tf.reset_default_graph()
#    model = DDDQN_model(conf, myAgent, tf.Session(), isPretrain=True)
#    model.initNet(load="preTrain")
#    for i in range(11-model.pretrain_episode()):
#        trackingpoints.reset_batch()
#        trainBatch = TPSample(conf, myAgent, trackingpoints.numsamples)
#        print("Iteration",model.pretrain_episode(),"Accuracy",model.getAccuracy(trainBatch),"%")
#        model.inc_episode()
#        trackingpoints.reset_batch()
#        while trackingpoints.has_next(BATCHSIZE):
#            trainBatch = TPSample(conf, myAgent, BATCHSIZE)
#            #model.sv_train_step(trainBatch, True)
#            model.q_train_step(trainBatch, True)    
#        if (i+1) % 5 == 0:
#            model.save()
           
            
#   #Fake real training
#    tf.reset_default_graph()
#    model = DDDQN_model(conf, myAgent, tf.Session(), isPretrain=False)
#    model.initNet(load=False)
#    for i in range(100):
#        trackingpoints.reset_batch()
#        trainBatch = TPSample(conf, myAgent, trackingpoints.numsamples)
#        print("Step",model.step(),"Accuracy",model.getAccuracy(trainBatch),"%") 
#        if i % 10 == 0:
#            print(model.inference(trainBatch[0][:2])) #die ersten 2 states
#        trackingpoints.reset_batch()
#        while trackingpoints.has_next(BATCHSIZE):
#            trainBatch = TPSample(conf, myAgent, BATCHSIZE)
#            model.q_train_step(trainBatch, True)    
#        if (i+1) % 5 == 0:
#            model.save()