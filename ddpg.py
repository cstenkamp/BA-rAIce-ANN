# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 18:23:09 2017

@author: nivradmin
"""
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #so that TF doesn't show its warnings
import numpy as np
from tensorflow.contrib.framework import get_variables 
import math
from utils import convolutional_layer, fc_layer, variable_summary

#bases heavily on http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html


class criticNet() :
    
    #TODO: ff_inputs, global_step
    def __init__(self, conf, agent, outerscope="critic", name="online"):
        with tf.variable_scope(name, reuse=None):
            self.conf = conf
            self.agent = agent
            self.name = name  
            self.num_actions = conf.num_actions
            self.h_size = 256
            self.conv_stacksize = (self.conf.history_frame_nr*2 if self.conf.use_second_camera else self.conf.history_frame_nr) if self.agent.conv_stacked else 1
            
            self.conv_inputs = tf.placeholder(tf.float32, shape=[None, self.conf.image_dims[0], self.conf.image_dims[1], self.conv_stacksize], name="conv_inputs")  
            self.actions = tf.placeholder(tf.float32, shape=[None, 3], name="action_inputs")  
            
            rs_input = tf.reshape(self.conv_inputs, [-1, self.conf.image_dims[0], self.conf.image_dims[1], self.conv_stacksize]) #final dimension = number of color channels*number of stacked (history-)frames                  
            #convolutional_layer(input_tensor, input_channels, kernel_size, stride, output_channels, name, act, is_trainable, batchnorm, is_training, weightdecay=False, pool=True, trainvars=None, varSum=None, initializer=None)
            conv1 = convolutional_layer(rs_input, self.conv_stacksize, [4,6], [2,3], 32, "Conv1", tf.nn.relu, True, True, True, False, False, {}, variable_summary, initializer="fanin") #(?, 14, 14, 32)
            conv2 = convolutional_layer(conv1, 32, [4,4], [2,2], 64, "Conv2", tf.nn.relu, True, True, True, False, False, {}, variable_summary, initializer="fanin")                     #(?, 8, 8, 64)
            conv3 = convolutional_layer(conv2, 64, [3,3], [2,2], 64, "Conv3", tf.nn.relu, True, True, True, False, True, {}, variable_summary, initializer="fanin")                      #(?, 2, 2, 64)
            conv4 = convolutional_layer(conv3, 64, [4,4], [2,2], self.h_size, "Conv4", tf.nn.relu, True, True, True, False, False, {}, variable_summary, initializer="fanin")            #(?, 1, 1, 256)
            conv4_flat = tf.reshape(conv4, [-1, self.h_size])
            fc1 = fc_layer(conv4_flat, self.h_size, 97, "FC1", True, True, True, False, tf.nn.relu, 1, {}, variable_summary, initializer="fanin")                 
            fc1 = tf.concat([fc1, self.actions], 1) 
            self.Q = fc_layer(fc1, 100, 1, "FC2", True, True, True, False, tf.nn.relu, 1, {}, variable_summary, initializer="fanin")                 
        
            self.ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=outerscope+"/"+self.name)      
            self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outerscope+"/"+self.name)
        



class actorNet():
    def __init__(self, conf, agent, bounds, outerscope="actor", name="online"):
        tanh_min_bounds,tanh_max_bounds = np.array([-1, -1, -1]), np.array([1, 1, 1])
        min_bounds, max_bounds = np.array(list(zip(*bounds)))
        
        with tf.variable_scope(name, reuse=None):
            self.conf = conf
            self.agent = agent
            self.name = name  
            self.num_actions = conf.num_actions
            self.h_size = 256
            self.action_bound = bounds #TODO: muss noch anders für brake & accelearation 
            self.conv_stacksize = (self.conf.history_frame_nr*2 if self.conf.use_second_camera else self.conf.history_frame_nr) if self.agent.conv_stacked else 1
            
            self.conv_inputs = tf.placeholder(tf.float32, shape=[None, self.conf.image_dims[0], self.conf.image_dims[1], self.conv_stacksize], name="conv_inputs")  

            rs_input = tf.reshape(self.conv_inputs, [-1, self.conf.image_dims[0], self.conf.image_dims[1], self.conv_stacksize]) #final dimension = number of color channels*number of stacked (history-)frames                  
            #convolutional_layer(input_tensor, input_channels, kernel_size, stride, output_channels, name, act, is_trainable, batchnorm, is_training, weightdecay=False, pool=True, trainvars=None, varSum=None, initializer=None)
            conv1 = convolutional_layer(rs_input, self.conv_stacksize, [4,6], [2,3], 32, "Conv1", tf.nn.relu, True, True, True, False, False, {}, variable_summary, initializer="fanin") #(?, 14, 14, 32)
            conv2 = convolutional_layer(conv1, 32, [4,4], [2,2], 64, "Conv2", tf.nn.relu, True, True, True, False, False, {}, variable_summary, initializer="fanin")                     #(?, 8, 8, 64)
            conv3 = convolutional_layer(conv2, 64, [3,3], [2,2], 64, "Conv3", tf.nn.relu, True, True, True, False, True, {}, variable_summary, initializer="fanin")                      #(?, 2, 2, 64)
            conv4 = convolutional_layer(conv3, 64, [4,4], [2,2], self.h_size, "Conv4", tf.nn.relu, True, True, True, False, False, {}, variable_summary, initializer="fanin")            #(?, 1, 1, 256)
            conv4_flat = tf.reshape(conv4, [-1, self.h_size])
            fc1 = fc_layer(conv4_flat, self.h_size, 100, "FC1", True, True, True, False, tf.nn.relu, 1, {}, variable_summary, initializer="fanin")                 
            self.fc2 = fc_layer(fc1, 100, 3, "FC2", True, False, True, False, tf.nn.tanh, 1, {}, variable_summary, initializer="fanin")                  #kein batchnorm im letzten layer sonst ist tanh größer 1!
            self.scaled = (((self.fc2 - tanh_min_bounds)/ (tanh_max_bounds - tanh_min_bounds)) * (max_bounds - min_bounds) + min_bounds) #broadcasts the action_bound array


        self.ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=outerscope+"/"+self.name)      
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outerscope+"/"+self.name)

##########################################################################################          
            


def _netCopyOps(fromNet, toNet, tau = 1):
    toCopy = fromNet.trainables
    toPast = toNet.trainables
    op_holder = []
    for idx,var in enumerate(toCopy[:]):
        if tau == 1:
            op_holder.append(toPast[idx].assign(var.value()))
        else:
            op_holder.append(toPast[idx].assign((var.value()*tau) + ((1-tau)*toPast[idx].value())))
    return op_holder

    
def _runMultipleOps(op_holder,sess):
    for op in op_holder:
        sess.run(op)      



##########################################################################################


class actor():
    
    def __init__(self, conf, agent, session):
        with tf.variable_scope("actor"):
            self.conf = conf
            self.agent = agent
            self.session = session
            self.learning_rate = 0.0001
            bounds = [(0, 1), (0, 1), (-1, 1)]
            
            self.online = actorNet(conf, agent, bounds)
            self.target = actorNet(conf, agent, bounds, name="target")
            
            self.smoothTargetUpdate = _netCopyOps(self.online, self.target, self.conf.target_update_tau)
            
            # This gradient will be provided by the critic network
            self.action_gradient = tf.placeholder(tf.float32, [None, self.conf.num_actions], name="actiongradient")        
            
            # Combine the gradients here
            self.actor_gradients = tf.gradients(self.online.scaled, self.online.trainables, -self.action_gradient)  #we want gradient AScent
            
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.online.trainables))

        
    def make_inputs(self, inputs):
        conv_inputs = [inputs[i][0] for i in range(len(inputs))]
        return conv_inputs
                
    def train(self, inputs, actiongrad):
        self.session.run(self.optimize, feed_dict={self.online.conv_inputs: self.make_inputs(inputs), self.action_gradient: actiongrad})
        
    def predict(self, inputs, which="target"):
        net = self.online if which == "online" else self.target
        return self.session.run(net.scaled, feed_dict={net.conv_inputs: self.make_inputs(inputs)})
        
    def update_target(self):
        _runMultipleOps(self.smoothTargetUpdate, self.session)
    

        
class critic():
    def __init__(self, conf, agent, session):
        with tf.variable_scope("critic"):
            self.conf = conf
            self.agent = agent
            self.session = session
            self.learning_rate = 0.001
            
            self.online = criticNet(conf, agent)
            self.target = criticNet(conf, agent, name="target")
            
            self.smoothTargetUpdate = _netCopyOps(self.online, self.target, self.conf.target_update_tau)      
                
            # Network target (y_i)
            self.predicted_q_value = tf.placeholder(tf.float32, [None, 1], name="predQVal")        
                
            self.loss = tf.reduce_mean(tf.square(self.predicted_q_value - self.online.Q))
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)    
            
            self.action_grads = tf.gradients(self.online.Q, self.online.actions)

    def make_inputs(self, inputs):
        conv_inputs = [inputs[i][0] for i in range(len(inputs))]
        return conv_inputs        
    
    def train(self, inputs, action, predQVal):
        return self.session.run([self.online.Q, self.optimize], feed_dict={self.online.conv_inputs: self.make_inputs(inputs), \
                                                                      self.online.actions: action, self.predicted_q_value: predQVal})        
        
    def predict(self, inputs, action, which="target"):
        net = self.online if which == "online" else self.target
        return self.session.run(net.Q, feed_dict={net.conv_inputs: self.make_inputs(inputs), net.actions: action})
        
    def action_gradients(self, inputs, actions):
        return self.session.run(self.action_grads, feed_dict = {self.online.conv_inputs: self.make_inputs(inputs), self.online.actions: actions})
        
            
    def update_target(self):
        _runMultipleOps(self.smoothTargetUpdate, self.session)        

##########################################################################################        
        
        
class DDPG_model():
    
    def __init__(self, conf, agent, session):
        self.conf = conf
        self.agent = agent
        self.session = session
        
        self.actor = actor(conf, agent, session)
        self.critic = critic(conf, agent, session)
        
        self.session.run(tf.global_variables_initializer())
        _runMultipleOps(_netCopyOps(self.actor.target, self.actor.online), self.session)
        _runMultipleOps(_netCopyOps(self.critic.target, self.critic.online), self.session)
        
        
    #actor predicts action. critic predicts q-value of action. That is compared to the *actual* q-value of action.
    #critic gets trained with the actual q-values. online-actor predicts new actions. actor uses critic's gradients to train, too.
    def train_step(self, batch):
        oldstates, actions, rewards, newstates, terminals = batch
        target_q = self.critic.predict(newstates, self.actor.predict(newstates))
        cumrewards = [rewards[i] if terminals[i] else rewards[i]+self.conf.q_decay*target_q[i] for i in range(len(rewards))]
        predicted_q_value, _ =  self.critic.train(oldstates, actions, cumrewards)
        a_outs = self.actor.predict(oldstates, "online")
        grads = self.critic.action_gradients(oldstates, a_outs)
        self.actor.train(oldstates, grads[0])
        
        self.actor.update_target()
        self.critic.update_target()
        
    def inference(self, statesBatch):
        return self.actor.predict(statesBatch)

    def old_evaluate(self, batch):
        oldstates, actions, rewards, newstates, terminals = batch
        result = np.array([np.argmax(self.agent.discretize(*i)) for i in self.actor.predict(oldstates)])
        human = np.array([np.argmax(myAgent.discretize(*i)) for i in actions])
        return np.mean(np.array(human == result, dtype=int))
        
        
##########################################################################################


def TPSample(conf, agent, batchsize):
    return read_supervised.create_QLearnInputs_from_PTStateBatch(*trackingpoints.next_batch(conf, agent, batchsize), agent)


if __name__ == '__main__':       
    import config
    conf = config.Config()
    import read_supervised
    from server import Containers; 
    import dqn_rl_agent
    myAgent = dqn_rl_agent.Agent(conf, Containers(), True)
    trackingpoints = read_supervised.TPList(conf.LapFolderName, conf.use_second_camera, conf.msperframe, conf.steering_steps, conf.INCLUDE_ACCPLUSBREAK)


    BATCHSIZE = 32   

    tf.reset_default_graph()
    model = DDPG_model(conf, myAgent, tf.Session())
    
    #TODO: einen continous agent machen, bei dem getnetusableaction nicht argmax returned!
    
    for i in range(10000):
        trackingpoints.reset_batch()
        trainBatch = TPSample(conf, myAgent, trackingpoints.numsamples)
        print("Iteration", i, "Accuracy",model.old_evaluate(trainBatch))  
        if i % 10 == 0:
            print(model.inference(trainBatch[0][:2])) #die ersten 2 states   
        trackingpoints.reset_batch()     
        while trackingpoints.has_next(BATCHSIZE):
            trainBatch = TPSample(conf, myAgent, BATCHSIZE)
            model.train_step(trainBatch)    
            
#            

           