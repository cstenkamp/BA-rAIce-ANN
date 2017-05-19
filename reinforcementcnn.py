# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:05:44 2017

@author: nivradmin
"""

import tensorflow as tf
import numpy as np
import os
import time
import math
#====own classes====
import read_supervised
import supervisedcnn

#SUMMARYALL = 1000 TODO: do

class RL_Config(object):
    log_dir = "SummaryLogDirRL/"  
    checkpoint_dir = "RL_Learn/"
    
    
    keep_prob = 1
    max_grad_norm = 10
    initial_lr = 0.005
    lr_decay = 0.9
    
    
    def __init__(self):     
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)     
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)                 
            
        self.history_frame_nr = supervisedcnn.Config().history_frame_nr
        self.image_dims = supervisedcnn.Config().image_dims
        self.steering_steps = supervisedcnn.Config().steering_steps
        self.speed_neurons = supervisedcnn.Config().speed_neurons
                                                  
        assert os.path.exists(supervisedcnn.Config().checkpoint_dir), "I need a pre-trained model"


class CNN(object):
    
    ######methods for BUILDING the computation graph######
    def __init__(self, config, initializer, is_training=True, continuing = False):
        #builds the computation graph, using the next few functions (this is basically the interface)
        self.config = config
        self.iterations = 0
        final_neuron_num = self.config.steering_steps*4 if self.config.INCLUDE_ACCPLUSBREAK else self.config.steering_steps*3
    
        self.inputs, self.q_targets, self.speed_input = self.set_placeholders(is_training, final_neuron_num)
        
        if not continuing:
            with tf.variable_scope("cnnmodel", reuse=True, initializer=initializer):
                self.q, self.argmaxs, self.q_max, self.action = self.inference(final_neuron_num, is_training)         
            with tf.variable_scope("cnnmodel", reuse=None, initializer=initializer):
                self.rl_loss = self.loss_func(self.q, self.q_targets)
                self.rl_train_op = self.training(self.rl_loss, config.initial_lr) 
        else:
            self.q, self.argmaxs, self.q_max, self.action = self.inference(final_neuron_num, is_training)         
            self.rl_loss = self.loss_func(self.q, self.q_targets)
            self.rl_train_op = self.training(self.rl_loss, config.initial_lr) 
            

    
    def set_placeholders(self, is_training, final_neuron_num):
        if self.config.history_frame_nr == 1:
            inputs = tf.placeholder(tf.float32, shape=[None, self.config.image_dims[0], self.config.image_dims[1]], name="inputs")  #first dim is none since inference has another batchsize than training
        else:
            inputs = tf.placeholder(tf.float32, shape=[None, self.config.history_frame_nr, self.config.image_dims[0], self.config.image_dims[1]], name="inputs")  #first dim is none since inference has another batchsize than training
            
        if is_training:
            q_targets = tf.placeholder(tf.float32, shape=[None, final_neuron_num], name="targets")    
        else:
            q_targets = None
            
        if self.config.speed_neurons:
            speeds = tf.placeholder(tf.float32, shape=[None, self.config.speed_neurons], name="speed_inputs")
        else:
            speeds = None
            
        return inputs, q_targets, speeds
    
    
    def inference(self, final_neuron_num, for_training=False): 
    
    #TODO: das hier ist GENAU WIE supervisedcnn, nur dass die convolutional trainable = false haben -- dafür nen simplen parameter!!
    #und ypre heißt q, und q_max&action kamen dazu
        
        self.trainvars = {}
        
        def weight_variable(shape, name, is_trainable=True):
          return tf.get_variable(name, shape, initializer= tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.config.image_dims[0]*self.config.image_dims[1]))), trainable=is_trainable)
                
        def bias_variable(shape, name, is_trainable=True):
          #Since we're using ReLU neurons, it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons"
          return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1), trainable=is_trainable)
        
        def conv2d(x, W):
          return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        
        def max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
        def convolutional_layer(input_tensor, input_channels, output_channels, name, act):
            with tf.name_scope(name):
                W = weight_variable([5, 5, input_channels, output_channels], "W_%s" % name, is_trainable=False)
                b = bias_variable([output_channels], "b_%s" % name, is_trainable=False)
                h_act = act(conv2d(input_tensor, W) + b)
                h_pool = max_pool_2x2(h_act)
                return h_pool
        
        def fc_layer(input_tensor, input_size, output_size, name, act=None, do_dropout=False):
            with tf.name_scope(name):
                self.trainvars["W_%s" % name] = weight_variable([input_size, output_size], "W_%s" % name)
                self.trainvars["b_%s" % name] = bias_variable([output_size], "b_%s" % name)
                h_fc =  tf.matmul(input_tensor, self.trainvars["W_%s" % name]) + self.trainvars["b_%s" % name]
                if act is not None:
                    h_fc = act(h_fc)
                tf.summary.histogram("activations", h_fc)
                if do_dropout:
                    h_fc = tf.nn.dropout(h_fc, self.keep_prob) 
                return h_fc

        rs_input = tf.reshape(self.inputs, [-1, self.config.image_dims[0], self.config.image_dims[1],self.config.history_frame_nr]) #final dimension = number of color channels
                             
        self.keep_prob = tf.Variable(tf.constant(1.0), trainable=False) #wenn nicht gefeedet ist sie standardmäßig 1        
        h1 = convolutional_layer(rs_input, self.config.history_frame_nr, 32, "Conv1", tf.nn.relu) #reduces to 15*21
        h2 = convolutional_layer(h1, 32, 64, "Conv2", tf.nn.relu)      #reduces to 8*11
        h_pool_flat =  tf.reshape(h2, [-1, 8*11*64])         
        h_fc1 = fc_layer(h_pool_flat, 8*11*64, 1024, "FC1", tf.nn.relu, do_dropout=for_training)                 
        if self.config.speed_neurons:
            h_fc1 = tf.concat([h_fc1, self.speed_input], 1)   #its lengths is now in any case 1024+speed_neurons
        q = fc_layer(h_fc1, 1024+self.config.speed_neurons, final_neuron_num, "FC2", None, do_dropout=False) 
        q_max = tf.reduce_max(q, axis=1)
        action = tf.argmax(q, axis=1) #Todo: kann gut sein dass ich action nicht brauche wenn ich argm hab
        y_conv = tf.nn.softmax(q)
        argm = tf.one_hot(tf.argmax(y_conv, dimension=1), depth=final_neuron_num)
        return q, argm, q_max, action
    
    
    def loss_func(self, q, q_target):
        return tf.reduce_mean(tf.square(q_target - q))

        
    def training(self, loss, init_lr):
        
        self.learning_rate = tf.Variable(tf.constant(self.config.initial_lr), trainable=False)
        self.new_lr = tf.placeholder(tf.float32, shape=[]) #diese und die nächste zeile nur nötig falls man per extra-aufruf die lr verändern will, so wie ich das mache braucht man die nicht.
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)        

        if self.config.max_grad_norm > 0:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.config.max_grad_norm)
            
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        
        self.global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False) #wird hier: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py halt gemacht... warum hiernochmal weiß ich nicht.
        
        train_op = optimizer.minimize(loss, global_step=self.global_step)
        
        return train_op
    
        
#        
#    ######methods for RUNNING the computation graph######
#    def train_fill_feed_dict(self, config, dataset, batchsize = 0, decay_lr = True):
#        batchsize = config.batch_size if batchsize == 0 else batchsize
#        _, visionvec, targets, _, speeds = dataset.next_batch(config, batchsize)
#        if decay_lr:
#            lr_decay = config.lr_decay ** max(self.iterations-config.lrdecayafter, 0.0)
#            new_lr = max(config.initial_lr*lr_decay, config.minimal_lr)
#        feed_dict = {self.inputs: visionvec, self.targets: targets, self.keep_prob: config.keep_prob, self.learning_rate: new_lr}
#        if config.speed_neurons:
#            feed_dict[self.speed_input] = speeds
#        return feed_dict            
#
#
#    def assign_lr(self, session, lr_value):
#        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})
#     
#
#    def run_train_epoch(self, session, dataset, summarywriter = None):
#        gesamtLoss = 0
#        self.iterations += 1
#        
#        dataset.reset_batch()
#        for i in range(dataset.num_batches(self.config.batch_size)):
#            feed_dict = self.train_fill_feed_dict(self.config, dataset)
#            if self.iterations % SUMMARYALL == 0 and summarywriter is not None:
#                _, loss, summary_str = session.run([self.rl_train_op, self.rl_loss, self.summary], feed_dict=feed_dict)   
#                summarywriter.add_summary(summary_str, session.run(self.global_step))
#                summarywriter.flush()
#            else:
#                _, loss = session.run([self.rl_train_op, self.rl_loss], feed_dict=feed_dict)   
#            gesamtLoss += loss
#        
#        return gesamtLoss
#        
#    
#    def run_eval(self, session, dataset):            
#        dataset.reset_batch()
#        feed_dict = self.train_fill_feed_dict(self.config, dataset, dataset.numsamples) #would be terribly slow if we learned, but luckily we only evaluate. should be fine.
#        accuracy, loss = session.run([self.accuracy, self.rl_loss], feed_dict=feed_dict)
#        return accuracy, loss, dataset.numsamples
#            
#            
    def run_inference(self, session, visionvec, othervecs, hframes):
        if hframes > 1:
            if not type(visionvec[0]).__module__ == np.__name__:
                return False, None #dann ist das input-array leer
        else:
            if not type(visionvec).__module__ == np.__name__:
                return False, None #dann ist das input-array leer
            assert (np.array(visionvec.shape) == np.array(self.inputs.get_shape().as_list()[1:])).all()
        
        with tf.device("/cpu:0"):
            visionvec = np.expand_dims(visionvec, axis=0)
            feed_dict = {self.inputs: visionvec}  
            if self.config.speed_neurons:
                speed_disc = read_supervised.inflate_speed(othervecs[1][4], self.config.speed_neurons, self.config.SPEED_AS_ONEHOT)
                feed_dict[self.speed_input] = np.expand_dims(speed_disc, axis=0)

            return True, session.run(self.argmaxs, feed_dict=feed_dict)
#
#            
#       
#def run_CNN_training(config, dataset):
#    graph = tf.Graph()
#    with graph.as_default():    
#        #initializer = tf.constant_initializer(0.0) #bei variablescopes kann ich nen default-initializer für get_variables festlegen
#        initializer = tf.random_uniform_initializer(-0.1, 0.1)
#                                             
#        with tf.name_scope("train"):
#            with tf.variable_scope("cnnmodel", reuse=None, initializer=initializer):
#                cnn = CNN(config, is_training=True)
#        
#        init = tf.global_variables_initializer()
#        cnn.trainvars["global_step"] = cnn.global_step
#        saver = tf.train.Saver(cnn.trainvars, max_to_keep=3)
#
#        with tf.Session(graph=graph) as sess:
#            summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph) #aus dem toy-example
#            
#            sess.run(init)
#            ckpt = tf.train.get_checkpoint_state(config.checkpoint_dir) 
#            if ckpt and ckpt.model_checkpoint_path:
#                saver.restore(sess, ckpt.model_checkpoint_path)
#                stepsPerIt = dataset.numsamples//config.batch_size
#                already_run_iterations = cnn.global_step.eval()//stepsPerIt
#                cnn.iterations = already_run_iterations
#                print("Restored checkpoint with",already_run_iterations,"Iterations run already") 
#            else:
#                already_run_iterations = 0
#                
#            num_iterations = config.iterations - already_run_iterations
#            print("Running for",num_iterations,"further iterations" if already_run_iterations>0 else "iterations")
#            for _ in range(num_iterations):
#                start_time = time.time()
#
#                step = cnn.global_step.eval() 
#                train_loss = cnn.run_train_epoch(sess, dataset, summary_writer)
#                
#                savedpoint = ""
#                if cnn.iterations % CHECKPOINTALL == 0 or cnn.iterations == config.iterations:
#                    checkpoint_file = os.path.join(config.checkpoint_dir, 'model.ckpt')
#                    saver.save(sess, checkpoint_file, global_step=step)       
#                    
#                    savedpoint = "(checkpoint saved)"
#                
#                print('Iteration %3d (step %4d): loss = %.2f (%.3f sec)' % (cnn.iterations, step+1, train_loss, time.time()-start_time), savedpoint)
#                
#                
#            ev, loss, _ = cnn.run_eval(sess, dataset)
#            print("Result of evaluation:")
#            print("Loss: %.2f,  Correct inferences: %.2f%%" % (loss, ev*100))
#
##            dataset.reset_batch()
##            _, visionvec, _, _ = dataset.next_batch(config, 1)
##            visionvec = np.array(visionvec[0])
##            print(cnn.run_inference(sess,visionvec, config.history_frame_nr))
# 


