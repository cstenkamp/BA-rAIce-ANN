import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn

from replay_buffer import ReplayBuffer

# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = True
# Use Gym Monitor
GYM_MONITOR_EN = False
# Gym environment
ENV_NAME = 'Pendulum-v0'
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64


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
        self.conf = conf
        self.agent = agent
        self.name = name  
#        self.num_actions = conf.num_actions
#        self.h_size = 100
#        self.conv_stacksize = (self.conf.history_frame_nr*2 if self.conf.use_second_camera else self.conf.history_frame_nr) if self.agent.conv_stacked else 1
                
        self.s_dim = 3
        self.a_dim = 1

        with tf.variable_scope(name, reuse=reuse):
            self.conv_inputs = tflearn.input_data(shape=[None, self.s_dim])
            self.actions = tflearn.input_data(shape=[None, self.a_dim])
            net = tflearn.fully_connected(self.conv_inputs, 400, activation='relu')
    
            # Add the action tensor in the 2nd hidden layer
            # Use two temp layers to get the corresponding weights and biases
            t1 = tflearn.fully_connected(net, 300)
            t2 = tflearn.fully_connected(self.actions, 300)
    
            net = tflearn.activation(
                tf.matmul(net, t1.W) + tf.matmul(self.actions, t2.W) + t2.b, activation='relu')
    
            # linear layer connected to 1 output representing Q(s,a)
            # Weights are init to Uniform[-3e-3, 3e-3]
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            self.Q = tflearn.fully_connected(net, 1, weights_init=w_init)            

            
        self.ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=outerscope+"/"+self.name)      
        self.losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=outerscope+"/"+self.name)      
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outerscope+"/"+self.name)
        



class actorNet():
    def __init__(self, conf, agent, bounds, outerscope="actor", name="online", reuse=False):
#        tanh_min_bounds,tanh_max_bounds = np.array([-1, -1, -1]), np.array([1, 1, 1])
#        min_bounds, max_bounds = np.array(list(zip(*bounds)))
        self.conf = conf
        self.agent = agent
        self.name = name  
#        self.num_actions = conf.num_actions
#        self.h_size = 100
#        self.action_bound = bounds #TODO: muss noch anders für brake & accelearation 
#        self.conv_stacksize = (self.conf.history_frame_nr*2 if self.conf.use_second_camera else self.conf.history_frame_nr) if self.agent.conv_stacked else 1

        self.s_dim = 3        
        self.a_dim = 1
        self.action_bound = [2]

        with tf.variable_scope(name, reuse=reuse):
            
            self.conv_inputs = tflearn.input_data(shape=[None, self.s_dim])
            net = tflearn.fully_connected(self.conv_inputs, 400, activation='relu')
            net = tflearn.fully_connected(net, 300, activation='relu')
            # Final layer weights are init to Uniform[-3e-3, 3e-3]
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            self.outs = tflearn.fully_connected(net, self.a_dim, activation='tanh', weights_init=w_init)
            # Scale output to -action_bound to action_bound
            self.scaled = tf.multiply(self.outs, self.action_bound)

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
            self.learning_rate = 0.0001 #0.0001
            bounds = [(0, 1), (0, 1), (-1, 1)]
            
            self.online = actorNet(conf, agent, bounds)
            self.target = actorNet(conf, agent, bounds, name="target")
            
            self.smoothTargetUpdate = _netCopyOps(self.online, self.target, 0.001)
            
            # This gradient will be provided by the critic network
            self.action_gradient = tf.placeholder(tf.float32, [None, 1], name="actiongradient")        
            
            # Combine the gradients here
            self.actor_gradients = tf.gradients(self.online.outs, self.online.trainables, -self.action_gradient)  #we want gradient AScent

            self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.online.trainables))
                
        
    def make_inputs(self, inputs):
#        conv_inputs = [inputs[i][0] for i in range(len(inputs))]
#        return conv_inputs
        return inputs
                
    def train(self, inputs, actiongrad):
        return self.session.run([self.actor_gradients, self.optimize], feed_dict={self.online.conv_inputs: self.make_inputs(inputs), self.action_gradient: actiongrad})
        
        
    def predict(self, inputs, which="target"):
        net = self.online if which == "online" else self.target
        return self.session.run(net.scaled, feed_dict={net.conv_inputs: self.make_inputs(inputs)})
        
    def update_target(self):
        self.session.run(self.smoothTargetUpdate)
    

        
class critic():
    def __init__(self, conf, agent, session):
        with tf.variable_scope("critic"):
            self.conf = conf
            self.agent = agent
            self.session = session
            self.learning_rate = 0.001
            
            self.online = criticNet(conf, agent)
            self.target = criticNet(conf, agent, name="target")
            
            self.smoothTargetUpdate = _netCopyOps(self.online, self.target, 0.001)      
                
            self.target_Q = tf.placeholder(tf.float32, [None, 1], name="target_Q")        
                
#            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.online.Q))
            self.loss = tf.losses.mean_squared_error(self.target_Q, self.online.Q)
            
            with tf.control_dependencies(self.online.ops):
                self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)    
            
            self.action_grads = tf.gradients(self.online.Q, self.online.actions) #letztere sind ja placeholders, kommend vom actor
#            self.action_grads = tf.stop_gradient(self.action_grads)

    def make_inputs(self, inputs):
#        conv_inputs = [inputs[i][0] for i in range(len(inputs))]
#        return conv_inputs        
        return inputs
    
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
        self.session.run(self.smoothTargetUpdate)    

##########################################################################################        
        
        
class DDPG_model():
    
    def __init__(self, conf, agent, session, delme):
        self.conf = conf
        self.agent = agent
        self.session = session
        self.initNet()
        self.s_dim, self.a_dim, self.action_bound = delme
        
        
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
        target_q = self.critic.predict(newstates, self.actor.predict(newstates))
#        cumrewards = [rewards[i] if terminals[i] else rewards[i]+self.conf.q_decay*target_Q[i] for i in range(len(rewards))]

        cumrewards = np.reshape([rewards[i] if terminals[i] else rewards[i]+0.99*target_q[i] for i in range(len(rewards))], (len(rewards),1))

        predicted_q_value, TDError, _ =  self.critic.train(oldstates, actions, cumrewards)
        #Training the actor...
        a_outs = self.actor.predict(oldstates, "online")
        grads = self.critic.action_gradients(oldstates, a_outs)
        asdf, _ = self.actor.train(oldstates, grads[0])
        
        self.critic.update_target()
        self.actor.update_target()
        
        return predicted_q_value
        
        
    def inference(self, statesBatch):
        return self.actor.predict(statesBatch, "online")

        
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


def train(sess, env, model):

    
    sess.run(tf.global_variables_initializer())


    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    for i in range(MAX_EPISODES):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(MAX_EP_STEPS):

            if RENDER_ENV:
                env.render()

            # Added exploration noise
            a = model.inference(np.reshape(s, (1, 3))) + (1. / (1. + i))

            s2, r, terminal, info = env.step(a[0])
                        
            
            replay_buffer.add(np.reshape(s, (model.s_dim,)), np.reshape(a, (model.a_dim,)), r, terminal, np.reshape(s2, (model.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)
                r_batch = [[i] for i in r_batch]
                t_batch = [[i] for i in t_batch]
                batch = s_batch, a_batch, r_batch, s2_batch, t_batch


                ep_ave_max_q += model.train_step(batch)[0]
#                # Calculate targets
#                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))
#
#                y_i = []
#                for k in range(MINIBATCH_SIZE):
#                    if t_batch[k]:
#                        y_i.append(r_batch[k])
#                    else:
#                        y_i.append(r_batch[k] + GAMMA * target_q[k])
#
#                # Update the critic given the targets
#                predicted_q_value, _ = critic.train(
#                    s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
#
#                ep_ave_max_q += np.amax(predicted_q_value)
#
#                # Update the actor policy using the sampled gradient
#                a_outs = actor.predict(s_batch)
#                grads = critic.action_gradients(s_batch, a_outs)
#                actor.train(s_batch, grads[0])
#
#                # Update target networks
#                actor.update_target_network()
#                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:

                print('| Reward: %.2i' % int(ep_reward), " | Episode", i, '| Qmax: %.4f' % (ep_ave_max_q / float(j)))

                break
            
            
            
            
            
    
           
def main(_):
    tf.reset_default_graph()
    with tf.Session() as sess:

        env = gym.make(ENV_NAME)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        
        
        model = DDPG_model(None, None, tf.Session(), [state_dim, action_dim, action_bound])
#        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
#                             ACTOR_LEARNING_RATE, TAU)
#
#        critic = CriticNetwork(sess, state_dim, action_dim,
#                               CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

        if GYM_MONITOR_EN:
            if not RENDER_ENV:
                env = wrappers.Monitor(
                    env, MONITOR_DIR, video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, MONITOR_DIR, force=True)

        train(sess, env, model)

        if GYM_MONITOR_EN:
            env.monitor.close()
            
            

if __name__ == '__main__':
    tf.app.run()