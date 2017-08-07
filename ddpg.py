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
import time
from utils import convolutional_layer, fc_layer, variable_summary
import tflearn
import tensorflow.contrib.slim as slim


#bases heavily on http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html


def dense(x, units, activation=tf.identity, decay=None, minmax=None):
    if minmax is None:
        minmax = float(x.shape[1].value) ** -.5
    return tf.layers.dense(x, units,activation=activation, kernel_initializer=tf.random_uniform_initializer(-minmax, minmax), kernel_regularizer=decay and tf.contrib.layers.l2_regularizer(1e-3))

    
class criticNet() :
    
    #TODO: ff_inputs, global_step, weight decay!
    def __init__(self, conf, agent, outerscope="critic", name="online", reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            self.conf = conf
            self.agent = agent
            self.name = name  
            self.num_actions = conf.num_actions
            self.h_size = 100
            self.conv_stacksize = (self.conf.history_frame_nr*2 if self.conf.use_second_camera else self.conf.history_frame_nr) if self.agent.conv_stacked else 1
            
            self.conv_inputs = tf.placeholder(tf.float32, shape=[None, self.conf.image_dims[0], self.conf.image_dims[1], self.conv_stacksize], name="conv_inputs")  
            self.actions = tf.placeholder(tf.float32, shape=[None, self.conf.num_actions], name="action_inputs")  
            
            rs_input = tf.reshape(self.conv_inputs, [-1, self.conf.image_dims[0], self.conf.image_dims[1], self.conv_stacksize]) #final dimension = number of color channels*number of stacked (history-)frames                  
            
            self.conv1 = slim.conv2d(inputs=rs_input,num_outputs=32,kernel_size=[4,6],stride=[2,3],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm)
            self.conv2 = slim.conv2d(inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm)
            self.conv3 = slim.conv2d(inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm)
            self.conv4 = slim.conv2d(inputs=self.conv3,num_outputs=self.h_size, kernel_size=[4,4],stride=[1,1],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm)          
            self.conv4_flat = tf.reshape(self.conv4, [-1, self.h_size])
            
            fc1 = tf.concat([self.conv4_flat, self.actions], 1) 
            fc2 = dense(fc1, 50, tf.nn.relu, decay=None)
            self.Q = dense(fc2, 1, decay=True, minmax=1e-4)
            
        self.ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=outerscope+"/"+self.name)      
        self.losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=outerscope+"/"+self.name)      
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outerscope+"/"+self.name)
        



class actorNet():
    def __init__(self, conf, agent, bounds, outerscope="actor", name="online", reuse=False):
#        tanh_min_bounds,tanh_max_bounds = np.array([-1, -1, -1]), np.array([1, 1, 1])
#        min_bounds, max_bounds = np.array(list(zip(*bounds)))
        
        with tf.variable_scope(name, reuse=reuse):
            self.conf = conf
            self.agent = agent
            self.name = name  
            self.num_actions = conf.num_actions
            self.h_size = 100
            self.action_bound = bounds #TODO: muss noch anders für brake & accelearation 
            self.conv_stacksize = (self.conf.history_frame_nr*2 if self.conf.use_second_camera else self.conf.history_frame_nr) if self.agent.conv_stacked else 1
            
            self.conv_inputs = tf.placeholder(tf.float32, shape=[None, self.conf.image_dims[0], self.conf.image_dims[1], self.conv_stacksize], name="conv_inputs")  

            rs_input = tf.reshape(self.conv_inputs, [-1, self.conf.image_dims[0], self.conf.image_dims[1], self.conv_stacksize]) #final dimension = number of color channels*number of stacked (history-)frames                  
            #convolutional_layer(input_tensor, input_channels, kernel_size, stride, output_channels, name, act, is_trainable, batchnorm, is_training, weightdecay=False, pool=True, trainvars=None, varSum=None, initializer=None)
           
            self.conv1 = slim.conv2d(inputs=rs_input,num_outputs=32,kernel_size=[4,6],stride=[2,3],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm)
            self.conv2 = slim.conv2d(inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm)
            self.conv3 = slim.conv2d(inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm)
            self.conv4 = slim.conv2d(inputs=self.conv3,num_outputs=self.h_size, kernel_size=[4,4],stride=[1,1],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm)          
            self.conv4_flat = tf.reshape(self.conv4, [-1, self.h_size])
            
            fc1 = dense(self.conv4_flat, 50, tf.nn.relu)
            self.fc2 = dense(fc1, self.num_actions, tf.nn.tanh, minmax = 3e-4)
            self.outs = self.fc2
#            self.scaled = (((self.outs - tanh_min_bounds)/ (tanh_max_bounds - tanh_min_bounds)) * (max_bounds - min_bounds) + min_bounds) #broadcasts the bound arrays
            self.scaled = self.outs

        self.ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=outerscope+"/"+self.name)      
        self.losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=outerscope+"/"+self.name)      
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
            self.learning_rate = 0.001 #0.0001
            bounds = [(0, 1), (0, 1), (-1, 1)]
            
            self.online = actorNet(conf, agent, bounds)
            self.target = actorNet(conf, agent, bounds, name="target")
            
            self.smoothTargetUpdate = _netCopyOps(self.online, self.target, self.conf.target_update_tau)
            
            # This gradient will be provided by the critic network
            self.action_gradient = tf.placeholder(tf.float32, [None, self.conf.num_actions], name="actiongradient")        
            
            # Combine the gradients here
            self.actor_gradients = tf.gradients(self.online.outs, self.online.trainables, -self.action_gradient)  #we want gradient AScent
            
            with tf.control_dependencies(self.online.ops):
                self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.online.trainables))

        
    def make_inputs(self, inputs):
        conv_inputs = [inputs[i][0] for i in range(len(inputs))]
        return conv_inputs
                
    def train(self, inputs, actiongrad):
        return self.session.run([self.online.outs, self.optimize], feed_dict={self.online.conv_inputs: self.make_inputs(inputs), self.action_gradient: actiongrad})
        
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
                
            self.target_Q = tf.placeholder(tf.float32, [None, 1], name="target_Q")        
                
#            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.online.Q))
            self.loss = tf.losses.mean_squared_error(self.target_Q, self.online.Q)
            
            with tf.control_dependencies(self.online.ops):
                self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)    
            
            self.action_grads = tf.gradients(self.online.Q, self.online.actions) #letztere sind ja placeholders, kommend vom actor
#            self.action_grads = tf.stop_gradient(self.action_grads)

    def make_inputs(self, inputs):
        conv_inputs = [inputs[i][0] for i in range(len(inputs))]
        return conv_inputs        
    
    def train(self, inputs, actions, predQVals):
        return self.session.run([self.online.Q, self.loss, self.optimize], feed_dict={self.online.conv_inputs: self.make_inputs(inputs), \
                                                                      self.online.actions: actions, self.target_Q: predQVals})      
        
    def TDError(self, inputs, actions, predQVals):
        return self.session.run([self.loss], feed_dict={self.online.conv_inputs: self.make_inputs(inputs), \
                                                                      self.online.actions: actions, self.target_Q: predQVals})           
        
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
        self.initNet()
        
        
    def initNet(self):
        self.actor = actor(self.conf, self.agent, self.session)
        self.critic = critic(self.conf, self.agent, self.session)
        
        self.session.run(tf.global_variables_initializer())
        _runMultipleOps(_netCopyOps(self.actor.target, self.actor.online), self.session)
        _runMultipleOps(_netCopyOps(self.critic.target, self.critic.online), self.session)
        
        
    #actor predicts action. critic predicts q-value of action. That is compared to the actual q-value of action. (TD-Error)
    #online-actor predicts new actions. actor uses critic's gradients to train, too.
    def train_step(self, batch):
        oldstates, actions, rewards, newstates, terminals = batch
        #Training the critic...
        target_Q = self.critic.predict(newstates, self.actor.predict(newstates))
        cumrewards = [rewards[i] if terminals[i] else rewards[i]+self.conf.q_decay*target_Q[i] for i in range(len(rewards))]
        predicted_q_value, TDError, _ =  self.critic.train(oldstates, actions, cumrewards)
        #Training the actor...
        a_outs = self.actor.predict(oldstates, "online")
        grads = self.critic.action_gradients(oldstates, a_outs)
        actorOuts, _ = self.actor.train(oldstates, grads[0])
#        print(np.mean(np.array([abs(actorOuts[i][0] -actions[i][0]) for i in range(len(actions))])))
        
        
        self.critic.update_target()
        self.actor.update_target()
        
        
    def inference(self, statesBatch):
        return self.actor.predict(statesBatch)

        
    def evaluate(self, batch):
        oldstates, actions, rewards, newstates, terminals = batch
#        result = np.array([np.argmax(self.agent.discretize(*i)) for i in self.actor.predict(oldstates)])
#        human = np.array([np.argmax(myAgent.discretize(*i)) for i in actions])
        actorOuts = self.actor.predict(oldstates)
        return np.mean(np.array([abs(actorOuts[i][0] -actions[i][0]) for i in range(len(actions))]))
        
        
    def eval_TDError(self, batch):
        oldstates, actions, rewards, newstates, terminals = batch
        target_Q = self.critic.predict(newstates, self.actor.predict(newstates))
        cumrewards = [rewards[i] if terminals[i] else rewards[i]+self.conf.q_decay*target_Q[i] for i in range(len(rewards))]
        loss =  self.critic.TDError(oldstates, actions, cumrewards)
        return loss[0]
        
        
##########################################################################################


def TPSample(conf, agent, batchsize):
    tmp = list(read_supervised.create_QLearnInputs_from_PTStateBatch(*trackingpoints.next_batch(conf, agent, batchsize), agent))
    tmp[1] = [[i[2]] for i in tmp[1]]
    return tmp 
    

if __name__ == '__main__':       
    import config
    conf = config.Config()
    conf.target_update_tau = 1e-3
    conf.num_actions = 1
    import read_supervised
    from server import Containers; 
    import dqn_rl_agent
    myAgent = dqn_rl_agent.Agent(conf, Containers(), True)
    trackingpoints = read_supervised.TPList(conf.LapFolderName, conf.use_second_camera, conf.msperframe, conf.steering_steps, conf.INCLUDE_ACCPLUSBREAK)

    BATCHSIZE = 32   

    tf.reset_default_graph()
    model = DDPG_model(conf, myAgent, tf.Session())
    
    #TODO: einen continous agent machen, bei dem getnetusableaction nicht argmax returned!
    
#    trainBatch = TPSample(conf, myAgent, trackingpoints.numsamples)
#    print(trainBatch[2][1])
    
    for i in range(10000):
        trackingpoints.reset_batch()
        trainBatch = TPSample(conf, myAgent, trackingpoints.numsamples)
        print("Iteration", i, "Accuracy (0 is best)",model.evaluate(trainBatch))  
        print("TD-Error vom Critic:",model.eval_TDError(trainBatch))
        if i % 10 == 0:
            print(np.array(model.inference(trainBatch[0][:2]))) #die ersten 2 states   
            print(np.array(trainBatch[1][:2]))
        trackingpoints.reset_batch()     
        while trackingpoints.has_next(BATCHSIZE):
            trainBatch = TPSample(conf, myAgent, BATCHSIZE)
            model.train_step(trainBatch)    
            
    time.sleep(99999)

           