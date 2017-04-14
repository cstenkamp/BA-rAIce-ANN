# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:41:09 2017

@author: csten_000
"""
import tensorflow as tf
import numpy as np
import os
import time
#====own functions====
import read_supervised

SUMMARYALL = 5
CHECKPOINTALL = 5


class Config(object):
    foldername = "SavedLaps/"
    history_frame_nr = 1
    batch_size = 32
    image_dims = [30,42]
    vector_len = 59
    keep_prob = 0.8
    initscale = 0.1
    iterations = 100 #100
    steering_steps = 11
    log_dir = "SummaryLogDir/"  
    checkpoint_dir = "Checkpoint/"
    layer1_neurons = 100
    keep_prob = 0.8
    
    def __init__(self):
        assert os.path.exists(self.foldername), "No data to train on at all!"        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)         
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir) 



                   

#was fehlt am ANN:
#    -changing learning rate incorporaten
#    -den supervisor und das andere nutzen, damit er bspw nach nem crash bereits mit global_step iterationen weitermacht
#    -dass man nicht immer gleich die Konsole resetten muss bei nem neustart
#    -TensorBoard integration!!
#    -Name-Scopes für die einzelnen Convolutional layer
#    -muss ich alle trainable sachen clippen?
#    -nutzen: Tensorflow website's stuff, die 3(!) dinge von meinem TF-Project, Leons TF-Project

#Funktioniert der SummaryWriter? Was macht er?                                               
#Fragen ob er weitertrainieren soll am anfang!
                                      
class CNN(object):
    
    ######methods for BUILDING the computation graph######
    def __init__(self, config, is_training=True):
        #builds the computation graph, using the next few functions (this is basically the interface)
        self.config = config
        self.stepsofar = 0
    
        self.inputs, self.targets = self.set_placeholders()
        
        if is_training:
            self.logits, self.argmaxs= self.inference(True)         
            self.loss = self.loss_func(self.logits, self.argmaxs)
            self.train_op = self.training(self.loss, 1e-4) #TODO: die learning rate muss sich verkleinern können
            self.accuracy = self.evaluation(self.argmaxs, self.targets)       
        else:
            with tf.device("/cpu:0"): #less overhead by not trying to switch to gpu
                self.logits, self.argmaxs= self.inference(False) 
        
        self.summary = tf.summary.merge_all() #aus dem toy-example
        
    
    def set_placeholders(self):
        inputs = tf.placeholder(tf.float32, shape=[None, self.config.image_dims[0], self.config.image_dims[1]], name="inputs")  #first dim is none since inference has another batchsize than training
        targets = tf.placeholder(tf.float32, shape=[None, self.config.steering_steps*4], name="targets")    
        return inputs, targets
    
    def inference(self, for_training=False):
        
        #TODO: das hier so much anders, mit variablescopes und get_variable -> ONLINE-SOURCECODE FINDEN!
        def weight_variable(shape, name):
          initial = tf.truncated_normal(shape, stddev=0.1)
          return tf.Variable(initial, name=name)
                
        def bias_variable(shape, name):
          initial = tf.constant(0.1, shape=shape) #Since we're using ReLU neurons, it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons"
          return tf.Variable(initial, name=name)
        
        def conv2d(x, W):
          return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        
        def max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
        rs_input = tf.reshape(self.inputs, [-1, self.config.image_dims[0], self.config.image_dims[1], 1]) #final dimension = number of color channels
    
        self.W_conv1 = weight_variable([5, 5, 1, 32], "WCon1") # patchsizey, patchsizey, #inputchannels, #outputchannels
        self.b_conv1 = bias_variable([32], "bCon1")
                
        h_conv1 = tf.nn.relu(conv2d(rs_input, self.W_conv1) + self.b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)  #reduces in this case to 15*21
        
        self.W_conv2 = weight_variable([5, 5, 32, 64], "WCon2")
        self.b_conv2 = bias_variable([64], "bCon2")
        
        h_conv2 = tf.nn.relu(conv2d(h_pool1, self.W_conv2) + self.b_conv2)
        h_pool2 = max_pool_2x2(h_conv2) #reduces to... 8*11?
        
        self.W_fc1 = weight_variable([8 * 11 * 64, 1024], "WFc1")
        self.b_fc1 = bias_variable([1024], "bFc2")
        
        h_pool2_flat = tf.reshape(h_pool2, [-1, 8*11*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1 )       

        if for_training: #TODO: warum nur hier dropout?
            self.keep_prob = tf.Variable(tf.constant(1.0), trainable=False) #wenn nicht gefeedet ist sie standardmäßig 1
            h_fc1 = tf.nn.dropout(h_fc1, self.keep_prob) 
        
        self.W_fc2 = weight_variable([1024, self.config.steering_steps*4], "WFc2")
        self.b_fc2 = bias_variable([self.config.steering_steps*4], "bfc2")
                
        y_conv = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
        argm = tf.one_hot(tf.argmax(y_conv, dimension=1), depth=self.config.steering_steps*4)
        
        return y_conv, argm
    
    
    def loss_func(self, logits, argmaxs):
        #calculates cross-entropy 
#        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.targets, logits=logits))
#        return cross_entropy
        #"Note that tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the model's unnormalized model prediction and sums across all classes"
        cross_entropy = -tf.reduce_sum(self.targets * tf.log(logits), reduction_indices=[1])
        return tf.reduce_mean(cross_entropy)        
        
        
    def training(self, loss, learning_rate):
        #returns the minimizer op

# TODO: incorporate this!        
#            tvars = tf.trainable_variables()
#            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
#            self.train_op = tf.train.AdamOptimizer().minimize(self.cost)        
        
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        #https://www.tensorflow.org/api_guides/python/train#optimizers
        return train_op
        
        
    def evaluation(self, argmaxs, targets):
        #returns how many percent it got correct
        made = tf.cast(argmaxs, tf.bool)
        real = tf.cast(targets, tf.bool)
        compare = tf.reduce_mean(tf.cast(tf.reduce_all(tf.equal(made, real),axis=1), tf.float32))
        return compare
        
        
    ######methods for RUNNING the computation graph######
    def train_fill_feed_dict(self, config, dataset, batchsize = 0):
        batchsize = config.batch_size if batchsize == 0 else batchsize
        _, visionvec, targets, _ = dataset.next_batch(config, batchsize)
        feed_dict = {self.inputs: visionvec, self.targets: targets, self.keep_prob: config.keep_prob}
        return feed_dict            


    def run_train_epoch(self, session, dataset, learning_rate, summarywriter = None):
        self.stepsofar += 1
        gesamtLoss = 0
        
        dataset.reset_batch()
        for i in range(dataset.num_batches(self.config.batch_size)):
            feed_dict = self.train_fill_feed_dict(self.config, dataset)
            _, loss = session.run([self.train_op, self.loss], feed_dict=feed_dict)   
            gesamtLoss += loss
        
        if self.stepsofar % SUMMARYALL == 0 and summarywriter is not None: #aus dem toy-example
            summary_str = session.run(self.summary, feed_dict=feed_dict)
            summarywriter.add_summary(summary_str, self.stepsofar)
            summarywriter.flush()
            
        return gesamtLoss
            
    def run_eval(self, session, dataset):            
        dataset.reset_batch()
        feed_dict = self.train_fill_feed_dict(self.config, dataset, dataset.numsamples) #would be terribly slow if we learned, but luckily we only evaluate. should be fine.
        accuracy, loss = session.run([self.accuracy, self.loss], feed_dict=feed_dict)
        return accuracy, loss, dataset.numsamples
            
            
    def run_inference(self, session, visionvec):
        if not type(visionvec).__module__ == np.__name__:
            return False, None
        assert (np.array(visionvec.shape) == np.array(self.inputs.get_shape().as_list()[1:])).all()
        visionvec = np.expand_dims(visionvec, axis=0)
        feed_dict = {self.inputs: visionvec}  
        return True, session.run(self.argmaxs, feed_dict=feed_dict)

            
       
def run_CNN_training(config, dataset):
    with tf.Graph().as_default():    
        #initializer = tf.constant_initializer(0.0) #bei variablescopes kann ich nen default-initializer für get_variables festlegen
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
                                             
        with tf.name_scope("train"):
            with tf.variable_scope("cnnmodel", reuse=None, initializer=initializer):
                cnn = CNN(config, is_training=True)
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver({"WCon1": cnn.W_conv1, "bCon1": cnn.b_conv1, "WCon2": cnn.W_conv2, "bCon2": cnn.b_conv2, "WFc1": cnn.W_fc1, "bFc1": cnn.b_fc1, "WFc2": cnn.W_fc2, "bFc2": cnn.b_fc2, })
        
        sv = tf.train.Supervisor(logdir="./supervisortraining/")
        with sv.managed_session() as sess:
            summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph) #aus dem toy-example
            sess.run(init)
                
            for step in range(config.iterations):
                start_time = time.time()
                if sv.should_stop():
                    break
                train_loss = cnn.run_train_epoch(sess, dataset, 0.5, summary_writer)
                print('Step %d: loss = %.2f (%.3f sec)' % (step, train_loss, time.time()-start_time))
                
                if step % CHECKPOINTALL == 0 or step == config.iterations-1:
                    checkpoint_file = os.path.join(config.checkpoint_dir, 'model.ckpt', global_step=step)
                    saver.save(sess, checkpoint_file, global_step=cnn.stepsofar)                
                
            ev, loss, _ = cnn.run_eval(sess, dataset)
            print("Loss:", loss, " Percentage of correct ones:", ev)

#            dataset.reset_batch()
#            _, visionvec, _, _ = dataset.next_batch(config, 1)
#            visionvec = np.array(visionvec[0])
#            print(cnn.run_inference(sess,visionvec))
 




#
#
#                         
#class FFNN_lookahead_steer(object):
#    
#    ######methods for BUILDING the computation graph######
#    def __init__(self, config, is_training=True):
#        #builds the computation graph, using the next few functions (this is basically the interface)
#        self.config = config
#        self.stepsofar = 0
#    
#        self.inputs, self.targets = self.set_placeholders()
#        
#        self.logits, self.argmaxs = self.inference(is_training)        
#        
#        if is_training:
#            self.loss = self.loss_func(self.logits)
#            self.train_op = self.training(self.loss, 0.5) #TODO: die learning rate muss sich verkleinern können
#            self.accuracy = self.evaluation(self.argmaxs, self.targets)        
#            
#        self.summary = tf.summary.merge_all() #aus dem toy-example
#        
#    
#    def set_placeholders(self):
#        inputs = tf.placeholder(tf.float32, shape=[None, self.config.vector_len], name="inputs")  #first dim is none since inference has another batchsize than training
#        targets = tf.placeholder(tf.float32, shape=[None, self.config.steering_steps], name="targets")    
#        return inputs, targets
#    
#    def inference(self, for_training=False):
#        #infos über hidden_units etc ziehen wir aus der config, die bei init mitgegeben wurde
#        #for_training wird benötigt, da wir, wenn wir dropout nutzen, nen neuen nicht-für-training initialisieren müssten!
#        #nutzt get_var und variable scopes
#        
#        #bei größeren Sachen würde man hier variable_scopes/name_scopes verwenden!
#        #self.W = tf.Variable(tf.zeros([59, self.config.steering_steps]), name="W")  
#        #self.b = tf.Variable(tf.zeros([self.config.steering_steps]), name="b")
#        with tf.name_scope("Layer1"):
#            self.W1 = tf.get_variable("W1", [59, self.config.layer1_neurons]) #TODO: mir überlegen wie das stattdessen mit mehreren history-frames geht
#            self.b1 = tf.get_variable("b1", [self.config.layer1_neurons])
#        with tf.name_scope("Layer2"):
#            self.W2 = tf.get_variable("W2", [self.config.layer1_neurons, self.config.steering_steps]) 
#            self.b2 = tf.get_variable("b2", [self.config.steering_steps])
#            
#        l1 = tf.nn.softmax(tf.matmul(self.inputs, self.W1) + self.b1)
#        y = tf.nn.softmax(tf.matmul(l1, self.W2) + self.b2, name="y")
#        argm = tf.one_hot(tf.argmax(y, dimension=1), depth=self.config.steering_steps)
#        return y, argm
#        
#    
#    def loss_func(self, logits):
#        #calculates cross-entropy 
#        cross_entropy = -tf.reduce_sum(self.targets * tf.log(logits), reduction_indices=[1])
#        return tf.reduce_mean(cross_entropy)
#        
#        
#    def training(self, loss, learning_rate):
#        #returns the minimizer op
#        #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#        train_op = tf.train.AdamOptimizer().minimize(loss)
#        return train_op
#        
#        
#    def evaluation(self, argmaxs, labels):
#        #returns how many percent it got correct.
#        made = tf.cast(argmaxs, tf.bool)
#        real = tf.cast(labels, tf.bool)
#        compare = tf.reduce_mean(tf.cast(tf.reduce_all(tf.equal(made, real),axis=1), tf.float32))
#        return compare
#        
#        
#    ######methods for RUNNING the computation graph######
#    def train_fill_feed_dict(self, config, dataset, batchsize = 0):
#        batchsize = config.batch_size if batchsize == 0 else batchsize
#        lookaheads, _, _, dtargets = dataset.next_batch(config, batchsize)
#        targets = dtargets[:,22:] #nur steering
#        feed_dict = {self.inputs: lookaheads, self.targets: targets}
#        return feed_dict            
#
#
#    def run_train_epoch(self, session, dataset, learning_rate, summarywriter = None):
#        self.stepsofar += 1
#        gesamtLoss = 0
#        
#        dataset.reset_batch()
#        for i in range(dataset.num_batches(self.config.batch_size)):
#            feed_dict = self.train_fill_feed_dict(self.config, dataset)
#            _, loss = session.run([self.train_op, self.loss], feed_dict=feed_dict)   
#            gesamtLoss += loss
#            
##        if summarywriter is not None: #aus dem toy-example
##            summary_str = session.run(self.summary, feed_dict=feed_dict)
##            summarywriter.add_summary(summary_str, self.stepsofar)
##            summarywriter.flush()
#            
#        return gesamtLoss
#            
#    def run_eval(self, session, dataset):            
#        dataset.reset_batch()
#        feed_dict = self.train_fill_feed_dict(self.config, dataset, dataset.numsamples) #would be terribly slow if we learned, but luckily we only evaluate. should be fine.
#        accuracy, loss = session.run([self.accuracy, self.loss], feed_dict=feed_dict)
#        return accuracy, loss, dataset.numsamples
#            
#            
#    def run_inference(self, session, inputvec):
#        assert np.array(inputvec.shape) == np.array(self.inputs.get_shape().as_list()[1:])
#        inputvec = np.expand_dims(inputvec, axis=0)
#        feed_dict = {self.inputs: inputvec}  
#        return session.run(self.argmaxs, feed_dict=feed_dict)
#
#            
#    
#       
#def run_FFNNSteer_training(config, dataset):
#    with tf.Graph().as_default():    
#        #initializer = tf.constant_initializer(0.0) #bei variablescopes kann ich nen default-initializer für get_variables festlegen
#        initializer = tf.random_uniform_initializer(-0.1, 0.1)
#                                             
#        with tf.name_scope("train"):
#            with tf.variable_scope("steermodel", reuse=None, initializer=initializer):
#                ffnn = FFNN_lookahead_steer(config)
#        
#        init = tf.global_variables_initializer()
#        saver = tf.train.Saver({"W1": ffnn.W1, "b1": ffnn.b1, "W2": ffnn.W2, "b2": ffnn.b2}) #der sollte ja nur einige werte machen
#        
#        sv = tf.train.Supervisor(logdir="./supervisortraining/")
#        with sv.managed_session() as sess:
#            summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph) #aus dem toy-example
#            sess.run(init)
#                
#            for step in range(config.iterations):
#                if sv.should_stop():
#                    break
#                train_loss = ffnn.run_train_epoch(sess, dataset, 0.5, summary_writer)
#                print(train_loss)
#                
#            checkpoint_file = os.path.join(config.log_dir, 'model.ckpt')
#            saver.save(sess, checkpoint_file, global_step=ffnn.stepsofar)                
#                
#            ev, loss, _ = ffnn.run_eval(sess, dataset)
#            print("Loss:", loss, " Percentage of correct ones:", ev)
#
#            dataset.reset_batch()
#            lookaheads, _, _, _ = dataset.next_batch(config, 1)
#            lookaheads = np.array(lookaheads[0])
#            print(ffnn.run_inference(sess,lookaheads))

        
        
        
def main(Steer=False):
    config = Config()
    
    if tf.gfile.Exists(config.log_dir):
        tf.gfile.DeleteRecursively(config.log_dir)
    tf.gfile.MakeDirs(config.log_dir)
    
    trackingpoints = read_supervised.TPList(config.foldername)
    print("Number of samples:",trackingpoints.numsamples)  
    if Steer:
#        run_FFNNSteer_training(config, trackingpoints)            
        None
    else:
        run_CNN_training(config, trackingpoints)        
    
                
                
if __name__ == '__main__':    
    main()