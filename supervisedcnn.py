# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:41:09 2017

@author: csten_000
"""
import tensorflow as tf
import numpy as np
import os
#====own functions====
from read_supervised import TPList


class Config(object):
    foldername = "SavedLaps/"
    history_frame_nr = 1
    batch_size = 20
    image_dims = [30,42]
    vector_len = 59
    keep_prob = 0.8
    initscale = 0.1
    iterations = 200
    num_steps = 100
    steering_steps = 11
    log_dir = "SummaryLogDir/"    



##what do we want? stride 1, SAME-padding
#class CNN(object):
#    def __init__(self, config, is_training=False): 
#        self.config = config
#        self.batch_size = config.batch_size
#        self.num_steps = config.num_steps
#        self.in_x = config.image_dims[0]
#        self.in_y = config.image_dims[1]
#
#        self.inputs = tf.placeholder(tf.float32, shape=[None,self.in_x, self.in_y], name="inputs") #TODO: eigentlich ist die letzte dimension ja config.history_frame_nr
#        self.targets = tf.placeholder(tf.float32, shape=[None, 3], name="targets") 
#        
#        def weight_variable(shape):
#          initial = tf.truncated_normal(shape, stddev=0.1)
#          return tf.Variable(initial)
#        
#        def bias_variable(shape):
#          initial = tf.constant(0.1, shape=shape)
#          return tf.Variable(initial)
#        
#        def conv2d(x, W):
#          return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#        
#        def max_pool_2x2(x):
#          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#  
#        self.inputs = tf.reshape(self.inputs,[-1, self.in_x, self.in_y, 1])
#
#        W_conv1 = weight_variable([5, 5, 1, 32])
#        b_conv1 = bias_variable([32])
#                
#        h_conv1 = tf.nn.relu(conv2d(self.inputs, W_conv1) + b_conv1)
#        h_pool1 = max_pool_2x2(h_conv1)  #reduces in this case to 15*21
#        
#        
#        W_conv2 = weight_variable([5, 5, 32, 64])
#        b_conv2 = bias_variable([64])
#        
#        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#        h_pool2 = max_pool_2x2(h_conv2) #reduces to... 8*11?
#        
#        W_fc1 = weight_variable([8 * 11 * 64, 1024])
#        b_fc1 = bias_variable([1024])
#        
#        h_pool2_flat = tf.reshape(h_pool2, [-1, 8*11*64])
#        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1 )       
#
#        if is_training:
#            h_fc1 = tf.nn.dropout(h_fc1, config.keep_prob) 
#        
#        W_fc2 = weight_variable([1024, 3])
#        b_fc2 = bias_variable([3])
#        
#        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2 #standard FF stuff here.
#                          
#        if not is_training:
#            return
#      
#        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.targets, logits=y_conv))
#        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self.targets,1)) #TODO its never gonna be equal since its continuus
#        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#    
#    def run_epoch(self, session, dataset, eval_op=None, printstuff=False):
#        epoch_size = len(dataset) // self.batch_size
#
#        for i in range(epoch_size):
#            visions, targets = sample_batch(self.config, dataset)
#            session.run(self.train_step, feed_dict={self.inputs:visions, self.targets: targets})
#            
#            
#        return self.accuracy.eval(feed_dict={self.inputs:visions, self.targets: targets})
#
#    
#
#
#def run_CNN(dataset, config):
#    with tf.Graph().as_default():
#        initializer = tf.random_uniform_initializer(-config.initscale, config.initscale)
#
#        with tf.name_scope("Train"):
#            with tf.variable_scope("Model", reuse=None, initializer=initializer):
#                m = CNN(config, is_training=True)
#
#        with tf.name_scope("Test"):
#            with tf.variable_scope("Model", reuse=True, initializer=initializer):
#                mtest = CNN(config, is_training=True)
#
#        with tf.Session() as session:
#            init = tf.global_variables_initializer()
#            init.run()
#            print("Running for",config.iterations,"iterations.")
#            for i in range(config.iterations):
#                print("Epoch: %d" % (i+1))
#                train_loss = m.run_epoch(session, dataset)
#                print("accuracy %g"%train_loss)     
#                
#                
#                
#class FFNN_lookahead_accbreak(object):
#    def __init__(self, config, is_training=False): 
#        self.config = config
#        self.batch_size = config.batch_size
#        self.num_steps = config.num_steps
#        
#        self.inputs = tf.placeholder(tf.float32, shape=[None, 59], name="inputs")  #TODO: eigentlich ist die letzte dimension ja config.history_frame_nr
#        self.targets = tf.placeholder(tf.float32, shape=[None, 4], name="targets") #two values, both of which become one-hot (0,0 = 0,1;0,1) etc
#     
#    #TODO: do.
#          
                                      
                                      
class FFNN_lookahead_steer(object):
    
    ######methods for BUILDING the computation graph######
    def __init__(self, config, is_training=True):
        #builds the computation graph, using the next few functions (this is basically the interface)
        self.config = config
        self.stepsofar = 0
    
        self.inputs, self.targets = self.set_placeholders()
        
        self.logits, self.argmaxs = self.inference(is_training)        
        self.loss = self.loss(self.logits)
        self.train_op = self.training(self.loss, 0.5) #TODO: die learning rate muss sich verkleinern können
        self.accuracy = self.evaluation(self.argmaxs, self.targets)        
        
        self.summary = tf.summary.merge_all() #aus dem toy-example
        
    
    def set_placeholders(self):
        inputs = tf.placeholder(tf.float32, shape=[None, 59], name="inputs")  #first dim is none since inference has another batchsize than training
        targets = tf.placeholder(tf.float32, shape=[None, self.config.steering_steps], name="targets")    
        return inputs, targets
    
    def inference(self, for_training=False):
        #infos über hidden_units etc ziehen wir aus der config, die bei init mitgegeben wurde
        #for_training wird benötigt, da wir, wenn wir dropout nutzen, nen neuen nicht-für-training initialisieren müssten!
        #nutzt get_var und variable scopes
        
        #bei größeren Sachen würde man hier name_scopes verwenden!
        W = tf.Variable(tf.zeros([59, self.config.steering_steps]), name="W") #TODO: diese hier rather mit get_variable?? #TODO: mir überlegen wie das stattdessen mit mehreren history-frames geht
        b = tf.Variable(tf.zeros([self.config.steering_steps]), name="b")
        y = tf.nn.softmax(tf.matmul(self.inputs, W) + b, name="y")
        argm = tf.one_hot(tf.argmax(y, dimension=1), depth=self.config.steering_steps)
        return y, argm
        
    
    def loss(self, logits):
        #calculates cross-entropy 
        cross_entropy = -tf.reduce_sum(self.targets * tf.log(logits), reduction_indices=[1])
        return tf.reduce_mean(cross_entropy)
        
        
    def training(self, loss, learning_rate):
        #returns the minimizer op
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        return train_op
        
        
    def evaluation(self, argmaxs, labels):
        #returns how many percent it got correct.
        made = tf.cast(argmaxs, tf.bool)
        real = tf.cast(labels, tf.bool)
        compare = tf.reduce_mean(tf.cast(tf.reduce_all(tf.equal(made, real),axis=1), tf.float32))
        return compare
        
        
    ######methods for RUNNING the computation graph######
    def train_fill_feed_dict(self, config, dataset, batchsize = 0):
        batchsize = config.batch_size if batchsize == 0 else batchsize
        lookaheads, _, targets = dataset.next_batch(config, batchsize)
        feed_dict = {self.inputs: lookaheads, self.targets: targets}
        return feed_dict            


    def run_train_epoch(self, session, dataset, learning_rate, summarywriter = None):
        self.stepsofar += 1
        gesamtLoss = 0
        
        dataset.reset_batch()
        for i in range(dataset.num_batches(self.config.batch_size)):
            feed_dict = self.train_fill_feed_dict(self.config, dataset)
            _, loss = session.run([self.train_op, self.loss], feed_dict=feed_dict)   
            gesamtLoss += loss
            
#        if summarywriter is not None: #aus dem toy-example
#            summary_str = session.run(self.summary, feed_dict=feed_dict)
#            summarywriter.add_summary(summary_str, self.stepsofar)
#            summarywriter.flush()
            
        return gesamtLoss
            
    def run_eval(self, session, dataset):            
        dataset.reset_batch()
        feed_dict = self.train_fill_feed_dict(self.config, dataset, dataset.numsamples) #would be terribly slow if we learned, but luckily we only evaluate. should be fine.
        accuracy, loss = session.run([self.accuracy, self.loss], feed_dict=feed_dict)
        return accuracy, loss, dataset.numsamples
            
            
    def run_inference(self, session, inputvec):
        assert np.array(inputvec.shape) == np.array(self.inputs.get_shape().as_list()[1:])
        inputvec = np.expand_dims(inputvec, axis=0)
        feed_dict = {self.inputs: inputvec}  
        return session.run(self.argmaxs, feed_dict=feed_dict)

            
       
def run_training(config, dataset):
    with tf.Graph().as_default():    
        initializer = tf.constant_initializer(0) #bei variablescopes kann ich nen default-initializer für get_variables festlegen
        
        with tf.name_scope("train"):
            with tf.variable_scope("steermodel", reuse=None, initializer=initializer):
                ffnn = FFNN_lookahead_steer(config)
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver() #der sollte ja nur einige werte machen
        
        sv = tf.train.Supervisor(logdir="./supervisortraining/")
        with sv.managed_session() as sess:
            summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph) #aus dem toy-example
            sess.run(init)
                
            for step in range(config.iterations):
                train_loss = ffnn.run_train_epoch(sess, dataset, 0.5, summary_writer)
                print(train_loss)
                
            checkpoint_file = os.path.join(config.log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=ffnn.stepsofar)                
                
            ev, loss, _ = ffnn.run_eval(sess, dataset)
            print("Loss:", loss, " Percentage of correct ones:", ev)

            dataset.reset_batch()
            lookaheads, _, _ = dataset.next_batch(config, 1)
            lookaheads = np.array(lookaheads[0])
            print(ffnn.run_inference(sess,lookaheads))

        
        
def main():
    config = Config()
    
    if tf.gfile.Exists(config.log_dir):
        tf.gfile.DeleteRecursively(config.log_dir)
    tf.gfile.MakeDirs(config.log_dir)
    
    trackingpoints = TPList(config.foldername)
    print("Number of samples:",trackingpoints.numsamples)  
    run_training(config, trackingpoints)        
    
                
                
if __name__ == '__main__':    
    main()