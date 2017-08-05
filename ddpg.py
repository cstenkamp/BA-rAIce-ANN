# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 22:37:40 2017

@author: csten_000
"""

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #so that TF doesn't show its warnings
import numpy as np
from tensorflow.contrib.framework import get_variables #unterschied zu tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='')?
from collections import namedtuple
#====own classes====
import read_supervised
from myprint import myprint as print
import config 
from utils import convolutional_layer, fc_layer, variable_summary


Network = namedtuple('Network', ['outputs', 'vars', 'ops', 'losses'])
                                      
class DDPG(object):
    
    ######methods for BUILDING the computation graph######
    def __init__(self, conf, is_training=True):
        #builds the computation graph, using the next few functions (this is basically the interface)
        self.conf = conf
        self.iterations = 0
        self.stacksize = self.conf.history_frame_nr*2 if self.conf.use_second_camera else self.conf.history_frame_nr
        
        self.prepareNumIters()
        
    
    self.action, init_op, self.train_op = self.make_network(action_states, states, actions, rewards, terminals, folgestates,
            training=self.training, action_bounds=env.action_bounds, steps=self.planned_steps)
        
            
            
    def make_network(self, act_states, states, actions, rewards, terminals, folgestates, training, action_bounds, steps):
        with tf.variable_scope('actor'):
            #actshape = actions.shape.as_list()[1:] ist bei mir einfach [3], würde benötigt werden für random
            #make_actor(self, states, bounds, name='online', reuse=False, batchnorm=True, is_training=True):
            actor = self.make_actor(states, action_bounds)
            actor_short = self.make_actor(act_states, action_bounds, reuse=True)
            #epsilon = (1. - tf.to_float(tf.train.get_global_step()) * (1. / tf.to_float(steps)))
            noise = 0 #noise = epsilon * tf.cond(training, lambda: self.make_noise(actshape), lambda: tf.constant(0.))
            action = actor_short.y + epsilon * noise
            action = tf.clip_by_value(action, *action_bounds)  # after noise
            actor_ = self.make_actor(folgestates, actshape, action_bounds, name='target')
        #tf.contrib.layers.summarize_tensors(actor.vars)

        # Create the online and target critic networks. This has a small
        # speciality: The online critic is created twice, once using the
        # fed states and fed actions as input and once using the fed states
        # and online actor's output as input. The latter is required to compute
        # the `policy gradient` to train the actor. The policy gradient
        # directly depends on how the online policy would currently 'act' in
        # the given state. The important part here is that those two critics
        # (in the following `critic` and `critic_short`) actually are the same
        # network, just with different inputs, but shared (!) parameters.
        with tf.variable_scope('critic'):
            #make_critic(self, inputs, actions, name='online', reuse=False, batchnorm=True, is_training=True):  
            critic = self.make_critic(states, actions)
            critic_short = self.make_critic(states, actor.y, reuse=True)
            critic_ = self.make_critic(folgestates, actor_.y, 'target')
        #tf.contrib.layers.summarize_tensors(critic.vars)

        # Create training and soft update operations.
        train_ops = [
            self.make_critic_trainer(critic, critic_, terminals, rewards, self.gamma, self.critic_learning_rate),
            self.make_actor_trainer(actor, critic_short, self.actor_learning_rate),
            self.make_soft_updates(critic, critic_, tau=self.tau),
            self.make_soft_updates(actor, actor_, tau=self.tau),
        ]

        # Sync the two network pairs initially.
        init_ops = [self.make_hard_updates(critic, critic_) + self.make_hard_updates(actor, actor_)]

        return action, init_ops, train_ops
            
############################ this part is done ################################         
        

    #TODO: settozero (me), weight-decay only for online/target nets??
    #TODO: weight sharing between critic's convlayers & actors convlayers
    #soooooooooo, critic gets as input action+state, and returns a single (Q-) value
    def make_critic(self, inputs, actions, name='online', reuse=False, batchnorm=True, is_training=True):  
        self.trainvars = {}
        with tf.variable_scope(name, reuse=reuse) as scope:
            flat_size = math.ceil(math.ceil(self.conf.image_dims[0]/4)*math.ceil(self.conf.image_dims[1]/4)/(2*2)*64) #die /2*2 wegen stride=2
            rs_input = tf.reshape(inputs, [-1, self.conf.image_dims[0], self.conf.image_dims[1], self.stacksize])
            self.keep_prob = tf.Variable(tf.constant(1.0), trainable=False) #wenn nicht gefeedet ist sie standardmäßig 1    
            #convolutional_layer(input_tensor, input_channels, kernel_size, stride, output_channels, name, act, is_trainable, batchnorm, is_training, weightdecay=False, pool=True, trainvars=None, varSum=None, initializer=None): #trainvars is call-by-reference-array, varSum is a function
            conv1 = convolutional_layer(rs_input, self.stacksize, [5,5], 2, 32, "Conv1", tf.nn.relu, True, batchnorm, is_training, 0.01, False, self.trainvars, variable_summary, "fanin") #reduces to x//2*y//2
            conv2 = convolutional_layer(conv1, 32,                [3,3], 1, 32, "Conv2", tf.nn.relu, True, batchnorm, is_training, 0.01, False, self.trainvars, variable_summary, "fanin")                #reduces to x//4*y//4
            conv2_flat =  tf.reshape(conv2, [-1, flat_size])
            #fc_layer(input_tensor, input_size, output_size, name, is_trainable, batchnorm, is_training, weightdecay=False, act=None, keep_prob=1, trainvars=None, varSum=None, initializer=None):
            fc1 = fc_layer(conv2_flat, flat_size, 200, "FC1", True, True, is_training, 0.01, tf.nn.relu, 1, self.trainvars, variable_summary, "fanin")
            fc1 = tf.concat([fc1, actions], 1) 
            if self.conf.speed_neurons:
                fc1 = tf.concat([fc1, spinputs], 1)             
            fc2 = fc_layer(fc1, 200+self.conf.num_actions+self.conf.speed_neurons, 200, "FC2", True, batchnorm, is_training, 0.01, tf.nn.relu, 1, self.trainvars, variable_summary, "fanin")
            q = fc_layer(fc2, 200, 1, "Final", True, batchnorm, is_training, 0.01, tf.nn.relu, 1, self.trainvars, variable_summary, tf.random_uniform_initializer(-0.0003, 0.0003))
            ops = scope.get_collection(tf.GraphKeys.UPDATE_OPS)
            losses = scope.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            return Network(tf.squeeze(q), get_variables(scope), ops, losses) 



    def make_actor(self, states, bounds, name='online', reuse=False, batchnorm=True, is_training=True):
        """Build an actor network mu, the policy function approximator."""
        self.trainvars = {}
        with tf.variable_scope(name, reuse=reuse) as scope:
            flat_size = math.ceil(math.ceil(self.conf.image_dims[0]/4)*math.ceil(self.conf.image_dims[1]/4)/(2*2)*64) #die /2*2 wegen stride=2
            rs_input = tf.reshape(inputs, [-1, self.conf.image_dims[0], self.conf.image_dims[1], self.stacksize])
            self.keep_prob = tf.Variable(tf.constant(1.0), trainable=False) #wenn nicht gefeedet ist sie standardmäßig 1    
            #convolutional_layer(input_tensor, input_channels, kernel_size, stride, output_channels, name, act, is_trainable, batchnorm, is_training, weightdecay=False, pool=True, trainvars=None, varSum=None, initializer=None): #trainvars is call-by-reference-array, varSum is a function
            conv1 = convolutional_layer(rs_input, self.stacksize, [5,5], 2, 32, "Conv1", tf.nn.relu, True, batchnorm, is_training, 0.01, False, self.trainvars, variable_summary, "fanin") #reduces to x//2*y//2
            conv2 = convolutional_layer(conv1, 32,                [3,3], 1, 32, "Conv2", tf.nn.relu, True, batchnorm, is_training, 0.01, False, self.trainvars, variable_summary, "fanin")                #reduces to x//4*y//4
            conv2_flat =  tf.reshape(conv2, [-1, flat_size])
            #fc_layer(input_tensor, input_size, output_size, name, is_trainable, batchnorm, is_training, weightdecay=False, act=None, keep_prob=1, trainvars=None, varSum=None, initializer=None):
            fc1 = fc_layer(conv2_flat, flat_size, 200, "FC1", True, True, is_training, 0.01, tf.nn.relu, 1, self.trainvars, variable_summary, "fanin")
            if self.conf.speed_neurons:
                fc1 = tf.concat([fc1, spinputs], 1)             
            fc2 = fc_layer(fc1, 200+self.conf.speed_neurons, 200, "FC2", True, batchnorm, is_training, 0.01, tf.nn.relu, 1, self.trainvars, variable_summary, "fanin")
            fc3 = fc_layer(fc2, 200, self.conf.num_actions, "Final", True, batchnorm, is_training, 0.01, tf.nn.tanh, 1, self.trainvars, variable_summary, tf.random_uniform_initializer(-0.0003, 0.0003))
            scaled = self.scale(fc3, bounds_in=(-1, 1), bounds_out=bounds) #TODO: das muss noch komplett anders, zumal speed & brake 0..1 ist und steer -1..1
            ops = scope.get_collection(tf.GraphKeys.UPDATE_OPS)
            losses = scope.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            return Network(scaled, get_variables(scope), ops, losses) 
        
    @staticmethod
    def scale(x, bounds_in, bounds_out):
        min_in, max_in = bounds_in
        min_out, max_out = bounds_out
        with tf.variable_scope('scaling'):
            return (((x - min_in) / (max_in - min_in)) * (max_out - min_out) + min_out)


########################## this part is done END ##############################

    @staticmethod
    def make_critic_trainer(critic, critic_, terminals, rewards, gamma=.99, learning_rate=1e-3):
        """Build critic network optimizer minimizing MSE.
        Terminal states are used as final horizon, meaning future rewards are
        only considered if the agent did not reach a terminal state.
        """
        with tf.variable_scope('training/critic'):
#            tf.summary.scalar('q/max', tf.reduce_max(critic.y)) #TODO: diese hier in summary_log-whtever-function
#            tf.summary.scalar('q/mean', tf.reduce_mean(critic.y))
            targets = tf.where(terminals, rewards, rewards + gamma * critic_.y)
            loss = tf.losses.mean_squared_error(targets, critic.y)
#            tf.summary.scalar('loss', loss)
            if len(critic.losses):
                loss += tf.add_n(critic.losses)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            with tf.control_dependencies(critic.ops):
                return optimizer.minimize(loss, tf.train.get_global_step())


    @staticmethod
    def make_actor_trainer(actor, critic, learning_rate=1e-4):
        """Build actor network optimizer performing gradient ascent."""
        with tf.variable_scope('training/actor'):
            # What is `actor.y`'s influence on the critic network's output?
            act_grad, = tf.gradients(critic.y, actor.y)  # (batchsize, dout)
            act_grad = tf.stop_gradient(act_grad)
            # Use `act_grad` as initial value for the `actor.y` gradients --
            # normally this is set to 1s by TF. Results in one value per param.
            policy_gradients = tf.gradients(actor.y, actor.vars, -act_grad)
            mapping = zip(policy_gradients, actor.vars)
            with tf.control_dependencies(actor.ops):
                optimizer = tf.train.AdamOptimizer(learning_rate)
                return optimizer.apply_gradients(mapping, tf.train.get_global_step())

    @staticmethod
    def make_hard_updates(src, dst):
        """Overwrite target with online network parameters."""
        with tf.variable_scope('hardupdates'):
            return [target.assign(online) for online, target in zip(src.vars, dst.vars)]

    @staticmethod
    def make_soft_updates(src, dst, tau=1e-3):
        """Soft update the dst net's parameters using those of the src net."""
        with tf.variable_scope('softupdates'):
            return [target.assign(tau * online + (1 - tau) * target) for online, target in zip(src.vars, dst.vars)]




    def set_placeholders(self, is_training, final_neuron_num):
        if self.conf.history_frame_nr == 1:
            inputs = tf.placeholder(tf.float32, shape=[None, self.conf.image_dims[0], self.conf.image_dims[1]], name="inputs")  #first dim is none since inference has another batchsize than training
        else:
            inputs = tf.placeholder(tf.float32, shape=[None, self.stacksize, self.conf.image_dims[0], self.conf.image_dims[1]], name="inputs")  #first dim is none since inference has another batchsize than training
            
            
        return inputs, targets, speeds
    
    
    def inference(self, inputs, spinputs, final_neuron_num, rl_not_trainables, for_training=False):

#
#
#    
#    def loss_func(self, logits, targets):
#
#        
#    
#    def rl_loss_func(self, q, q_target):
#    
#    
#    
#    def training(self, loss, init_lr, optimizer_arg):
#
#        
#    def evaluation(self, argmaxs, targets):
#
#    
#    def prepareNumIters(self):
#        self.numIterations = tf.Variable(tf.constant(0), trainable=False)
#        self.newIters = tf.placeholder(tf.int32, shape=[]) 
#        self.iterUpdate = tf.assign(self.numIterations, self.newIters)           
#    
#    ######methods for RUNNING the computation graph######
#    
#    def saveNumIters(self, session, value):
#        session.run(self.iterUpdate, feed_dict={self.newIters: value})
#        
#    def restoreNumIters(self, session):
#        return self.numIterations.eval(session=session)
#        
#    
#    
#    def train_fill_feed_dict(self, conf, dataset, batchsize = 0, decay_lr = True):
#        batchsize = conf.batch_size if batchsize == 0 else batchsize
#        _, visionvec, targets, speeds = dataset.next_batch(conf, batchsize)
#        if decay_lr:
#            lr_decay = conf.lr_decay ** max(self.iterations-conf.lrdecayafter, 0.0)
#            new_lr = max(conf.initial_lr*lr_decay, conf.minimal_lr)
#        feed_dict = {self.inputs: visionvec, self.targets: targets, self.keep_prob: conf.keep_prob, self.learning_rate: new_lr}
#        if conf.speed_neurons:
#            feed_dict[self.speed_input] = speeds
#        return feed_dict            
#
#
#    def assign_lr(self, session, lr_value):
#        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})
#     
#
#    def run_train_epoch(self, session, dataset, summarywriter = None):
#
#        return gesamtLoss
#        
#    
#    def run_eval(self, session, dataset):            
#        dataset.reset_batch()
#        feed_dict = self.train_fill_feed_dict(self.conf, dataset, dataset.numsamples) #would be terribly slow if we learned, but luckily we only evaluate. should be fine.
#        accuracy, loss = session.run([self.accuracy, self.loss], feed_dict=feed_dict)
#        return accuracy, loss, dataset.numsamples
#            
#            
#    def run_inference(self, session, visionvec, otherinputs, hframes):
#        if hframes > 1:
#            if not type(visionvec[0]).__module__ == np.__name__:
#                return False, (None, None) #dann ist das input-array leer
#        else:
#            if not type(visionvec).__module__ == np.__name__:
#                return False, (None, None) #dann ist das input-array leer
#            assert (np.array(visionvec.shape) == np.array(self.inputs.get_shape().as_list()[1:])).all()
#            
#        
#        with tf.device("/cpu:0"):
#            visionvec = np.expand_dims(visionvec, axis=0)
#            feed_dict = {self.inputs: visionvec}  
#            if self.conf.speed_neurons:
#                speed_disc = read_supervised.inflate_speed(otherinputs.SpeedSteer.velocity, self.conf.speed_neurons, self.conf.SPEED_AS_ONEHOT)
#                feed_dict[self.speed_input] = np.expand_dims(speed_disc, axis=0)
#            
#            return True, session.run([self.argmax, self.q], feed_dict=feed_dict)
#
#        
#        
#    def calculate_value(self, session, visionvec, speed, hframes):
#        with tf.device("/cpu:0"):
#            visionvec = np.expand_dims(visionvec, axis=0)
#            feed_dict = {self.inputs: visionvec}  
#            if self.conf.speed_neurons:
#               speed_disc = read_supervised.inflate_speed(speed, self.conf.speed_neurons, self.conf.SPEED_AS_ONEHOT)
#               feed_dict[self.speed_input] = np.expand_dims(speed_disc, axis=0)
#            
#            return session.run(self.q_max, feed_dict=feed_dict)
#            
#        
#       
#def run_svtraining(conf, dataset):
#  
# 
#
#
#
#        
#        
#        
#def main():
#    conf = config.Config()
#        
#    trackingpoints = read_supervised.TPList(conf.LapFolderName, conf.use_second_camera, conf.msperframe, conf.steering_steps, conf.INCLUDE_ACCPLUSBREAK)
#    print("Number of samples: %s | Tracking every %s ms with %s historyframes" % (trackingpoints.numsamples, str(conf.msperframe), str(conf.history_frame_nr)), level=6)
#    run_svtraining(conf, trackingpoints)        
#    
#                
#                
#if __name__ == '__main__':    
#    main()
#    time.sleep(5)