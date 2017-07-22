# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 23:12:49 2017

@author: csten_000
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import get_variables
from collections import namedtuple


Network = namedtuple('Network', ['outputs', 'vars', 'ops'])





def dense(x, units, activation=None, decay=None, minmax=None):
    if minmax is None:
        minmax = float(x.shape[1].value) ** -.5
    return tf.layers.dense(x, units, activation=activation, kernel_initializer=tf.random_uniform_initializer(-minmax, minmax),
                           kernel_regularizer=decay and tf.contrib.layers.l2_regularizer(1e-3) )


def conv(x, filters, kernelsize, padding="same", activation=tf.nn.reulu):
    return tf.layers.conv2d(inputs=x, filters=filters, kernel_size=kernelsize, padding=padding, activation=activation)


def pool(x, pool_size=[2,2], strides=2):
     return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=strides)


#soooooooooo, critic gets as input action+state, and returns a single (Q-) value

def make_critic(states, actions, name='online', reuse=False, batchnorm=False):  #TODO: USE BATCHNORM!!!!!!!
    #hier fehlen aber definitiv noch die convolutional layers
    with tf.variable_scope(name, reuse=reuse) as scope:
        net = dense(states, 100, selu, decay=True)  
        net = tf.concat([net, actions], axis=1)  # Actions enter the net
        net = dense(net, 50, selu, decay=True)  # Value estimation
        y = dense(net, 1, decay=True, minmax=3e-4)
        ops = get_variables(scope, collection=tf.GraphKeys.UPDATE_OPS)
        return Network(tf.squeeze(y), get_variables(scope), ops)


        h1 = convolutional_layer(rs_input, self.stacksize, 32, "Conv1", tf.nn.relu) #reduces to x//2*y//2
        h2 = convolutional_layer(h1, 32, 64, "Conv2", tf.nn.relu)      #reduces to x//4*y//4
        h_pool_flat =  tf.reshape(h2, [-1, math.ceil(self.config.image_dims[0]/4)*math.ceil(self.config.image_dims[1]/4)*64])
        h_fc1 = fc_layer(h_pool_flat, math.ceil(self.config.image_dims[0]/4)*math.ceil(self.config.image_dims[1]/4)*64, final_neuron_num*20, "FC1", tf.nn.relu, do_dropout=for_training)                 
        if self.config.speed_neurons:
            h_fc1 = tf.concat([h_fc1, spinputs], 1)   #its lengths is now in any case 1024+speed_neurons
        q = fc_layer(h_fc1, final_neuron_num*20+self.config.speed_neurons, final_neuron_num, "FC2", None, do_dropout=False) 














    
def selu(x): #https://arxiv.org/pdf/1706.02515.pdf
    with tf.name_scope('selu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x)) #why not scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x) - alpha) ?


def make_critic(states, actions, name='online', reuse=False):
    """Build a critic network q, the value function approximator."""
    with tf.variable_scope(name, reuse=reuse) as scope:
#         training = tf.shape(states)[0] > 1  # Training or evaluating?
#         states = tf.layers.batch_normalization(states, training=training)
        net = dense(states, 100, selu, decay=True)  # Feature extraction
#         net = tf.layers.batch_normalization(net, training=training)
        net = tf.concat([net, actions], axis=1)  # Actions enter the net
        net = dense(net, 50, selu, decay=True)  # Value estimation
        y = dense(net, 1, decay=True, minmax=3e-4)
#         ops = get_variables(scope, collection=tf.GraphKeys.UPDATE_OPS)
        return Network(tf.squeeze(y), get_variables(scope), [])
    
    
def train_critic(critic, critic_, terminals, rewards, gamma=.99):
    """Build critic network optimizer minimizing MSE."""
    with tf.variable_scope('critic'):
        # Terminal states limit the horizon -- only look at future rewards
        # if the agent did not reach a terminal state.
        targets = tf.where(terminals, rewards, rewards + gamma * critic_.y)
        mse = tf.reduce_mean(tf.squared_difference(targets, critic.y))
        tf.summary.scalar('loss', mse)
        optimizer = tf.train.AdamOptimizer(1e-3)
        with tf.control_dependencies(critic.ops):
            return optimizer.minimize(mse, tf.train.get_global_step())
        
def make_actor(states, dout, max_out, name='online'):
    """Build an actor network mu, the policy function approximator."""
    with tf.variable_scope(name) as scope:
#         training = tf.shape(states)[0] > 1  # Training or evaluating?
#         states = tf.layers.batch_normalization(states, training=training)
        net = dense(states, 100, selu)
#         net = tf.layers.batch_normalization(net, training=training)
        net = dense(net, 50, selu)
        y = dense(net, dout, tf.nn.tanh, minmax=3e-4)
        scaled = y * max_out
#         ops = get_variables(scope, collection=tf.GraphKeys.UPDATE_OPS)
        return Network(scaled, get_variables(scope), [])
    
    
def train_actor(actor, critic):
    """Build actor network optimizier performing action gradient ascent."""
    with tf.variable_scope('actor'):
        # What is `actor.y`'s influence on the critic network's output?
        value_gradient, = tf.gradients(critic.y, actor.y)  # (batchsize, dout)
        value_gradient = tf.stop_gradient(value_gradient)
        # Use `value_gradient` as initial value for the `actor.y` gradients --
        # normally this is set to 1s by TF. Results in a value per parameter.
        policy_gradients = tf.gradients(actor.y, actor.vars, -value_gradient)
        mapping = zip(policy_gradients, actor.vars)
        with tf.control_dependencies(actor.ops):
            return tf.train.AdamOptimizer(1e-4).apply_gradients(mapping)
        
        
def hard_updates(src, dst):
    """Overwrite target with online network parameters."""
    with tf.variable_scope('hardupdates'):
        return [target.assign(online)
                for online, target in zip(src.vars, dst.vars)]        
        
def soft_updates(src, dst, tau=1e-3):
    """Soft update the dst net's parameters using those of the src net."""
    with tf.variable_scope('softupdates'):
        return [target.assign(tau * online + (1 - tau) * target)
                for online, target in zip(src.vars, dst.vars)]        
        
def make_noise(n, theta=.15, sigma=.4):
    """Ornstein-Uhlenbeck noise process."""
    with tf.variable_scope('OUNoise'):
        state = tf.Variable(tf.zeros((n,)))
        noise = -theta * state + sigma * tf.random_normal((n,))
        # reset = state.assign(tf.zeros((n,)))
        return state.assign_add(noise)        