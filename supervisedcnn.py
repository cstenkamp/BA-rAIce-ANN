# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:41:09 2017

@author: csten_000
"""
import tensorflow as tf
import numpy as np
import os
import time
import math
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
    iterations = 200 #100
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
        self.iterations = 0
    
        self.inputs, self.targets = self.set_placeholders()
        
        if is_training:
            self.logits, self.argmaxs = self.inference(True)         
            self.loss = self.loss_func(self.logits)
            self.train_op = self.training(self.loss, 1e-4) #TODO: die learning rate muss sich verkleinern können
            self.accuracy = self.evaluation(self.argmaxs, self.targets)       
            self.summary = tf.summary.merge_all() #für TensorBoard
        else:
            with tf.device("/cpu:0"): #less overhead by not trying to switch to gpu
                self.logits, self.argmaxs = self.inference(False) 
            
    def variable_summary(self, var, what=""):
      with tf.name_scope('summaries_'+what):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
    
    
    def set_placeholders(self):
        inputs = tf.placeholder(tf.float32, shape=[None, self.config.image_dims[0], self.config.image_dims[1]], name="inputs")  #first dim is none since inference has another batchsize than training
        targets = tf.placeholder(tf.float32, shape=[None, self.config.steering_steps*4], name="targets")    
        return inputs, targets
    
    def inference(self, for_training=False):
        self.trainvars = {}
        
        def weight_variable(shape, name):
          return tf.get_variable(name, shape, initializer= tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.config.image_dims[0]*self.config.image_dims[1]))))
                
        def bias_variable(shape, name):
          #Since we're using ReLU neurons, it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons"
          return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))
        
        def conv2d(x, W):
          return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        
        def max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
        def convolutional_layer(input_tensor, input_channels, output_channels, name, act):
            with tf.name_scope(name):
                self.trainvars["W_%s" % name] = weight_variable([5, 5, input_channels, output_channels], "W_%s" % name)
                self.variable_summary(self.trainvars["W_%s" % name])
                self.trainvars["b_%s" % name] = bias_variable([output_channels], "b_%s" % name)
                self.variable_summary(self.trainvars["b_%s" % name])
                h_act = act(conv2d(input_tensor, self.trainvars["W_%s" % name]) + self.trainvars["b_%s" % name])
                h_pool = max_pool_2x2(h_act)
                tf.summary.histogram("activations", h_pool)
                return h_pool
        
        def fc_layer(input_tensor, input_size, output_size, name, act, do_dropout):
            with tf.name_scope(name):
                self.trainvars["W_%s" % name] = weight_variable([input_size, output_size], "W_%s" % name)
                self.variable_summary(self.trainvars["W_%s" % name])
                self.trainvars["b_%s" % name] = bias_variable([output_size], "b_%s" % name)
                self.variable_summary(self.trainvars["b_%s" % name])
                h_fc =  tf.matmul(input_tensor, self.trainvars["W_%s" % name]) + self.trainvars["b_%s" % name]
                if act is not None:
                    h_fc = act(h_fc)
                tf.summary.histogram("activations", h_fc)
                if do_dropout:
                    h_fc = tf.nn.dropout(h_fc, self.keep_prob) 
                return h_fc

        rs_input = tf.reshape(self.inputs, [-1, self.config.image_dims[0], self.config.image_dims[1], 1]) #final dimension = number of color channels
      
        self.keep_prob = tf.Variable(tf.constant(1.0), trainable=False) #wenn nicht gefeedet ist sie standardmäßig 1        
        h1 = convolutional_layer(rs_input, 1, 32, "Conv1", tf.nn.relu) #reduces to 15*21
        h2 = convolutional_layer(h1, 32, 64, "Conv2", tf.nn.relu)      #reduces to 8*11
        h_pool_flat =  tf.reshape(h2, [-1, 8*11*64])
        h_fc1 = fc_layer(h_pool_flat, 8*11*64, 1024, "FC1", tf.nn.relu, do_dropout=for_training)                 
        y_pre = fc_layer(h_fc1, 1024, self.config.steering_steps*4, "FC2", None, do_dropout=False) #TODO: dropout nur beim letzten??

        y_conv = tf.nn.softmax(y_pre)
        argm = tf.one_hot(tf.argmax(y_conv, dimension=1), depth=self.config.steering_steps*4)
        
        return y_pre, argm
    
    
    def loss_func(self, logits):
        #tf.nn.softmax_cross_entropy_with_logits vergleicht 1-hot-labels mit integer-logits
        #"Note that tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the model's unnormalized model prediction and sums across all classes"
        #The raw formulation of cross-entropy (one line below) can be numerically unstable. -> use tf's function!
        #cross_entropy = -tf.reduce_sum(self.targets * tf.log(tf.nn.softmax(logits)), reduction_indices=[1])       
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.targets, logits=logits)
        return tf.reduce_mean(cross_entropy)
        
    def training(self, loss, learning_rate):
        #returns the minimizer op
        tf.summary.scalar('loss', loss) #für TensorBoard
         
# TODO: incorporate this!        
#            tvars = tf.trainable_variables()
#            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
#            self.train_op = tf.train.AdamOptimizer().minimize(self.cost)        

        optimizer = tf.train.AdamOptimizer(learning_rate)
        #TODO: AUSSUCHEN können welchen optimizer, und meinen ausgesuchten verteidigen können
        #https://www.tensorflow.org/api_guides/python/train#optimizers
        
        self.global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False) #wird hier: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py halt gemacht... warum hiernochmal weiß ich nicht.
        
        train_op = optimizer.minimize(loss, global_step=self.global_step)
        return train_op
        
        
    def evaluation(self, argmaxs, targets):
        #returns how many percent it got correct
        made = tf.cast(argmaxs, tf.bool)
        real = tf.cast(targets, tf.bool)
        compare = tf.reduce_mean(tf.cast(tf.reduce_all(tf.equal(made, real),axis=1), tf.float32))
        tf.summary.scalar('accuracy', compare)
        return compare
        
        
    ######methods for RUNNING the computation graph######
    def train_fill_feed_dict(self, config, dataset, batchsize = 0):
        batchsize = config.batch_size if batchsize == 0 else batchsize
        _, visionvec, targets, _ = dataset.next_batch(config, batchsize)
        feed_dict = {self.inputs: visionvec, self.targets: targets, self.keep_prob: config.keep_prob}
        return feed_dict            


    def run_train_epoch(self, session, dataset, learning_rate, summarywriter = None):
        gesamtLoss = 0
        self.iterations += 1
        
        dataset.reset_batch()
        for i in range(dataset.num_batches(self.config.batch_size)):
            feed_dict = self.train_fill_feed_dict(self.config, dataset)
            if self.iterations % SUMMARYALL == 0 and summarywriter is not None:
                _, loss, summary_str = session.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)   
                summarywriter.add_summary(summary_str, session.run(self.global_step))
                summarywriter.flush()
            else:
                _, loss = session.run([self.train_op, self.loss], feed_dict=feed_dict)   
            gesamtLoss += loss
        
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
        cnn.trainvars["global_step"] = cnn.global_step
        saver = tf.train.Saver(cnn.trainvars, max_to_keep=3)

#        sv = tf.train.Supervisor(logdir="./supervisortraining/")
#        with sv.managed_session() as sess:
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph) #aus dem toy-example
            

            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(config.checkpoint_dir) 
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                stepsPerIt = dataset.numsamples//config.batch_size
                already_run_iterations = cnn.global_step.eval()//stepsPerIt
                cnn.iterations = already_run_iterations
                print("Restored checkpoint with",already_run_iterations,"Iterations run already") 
            else:
                already_run_iterations = 0
                
            num_iterations = config.iterations - already_run_iterations
            print("Running for",num_iterations,"further iterations" if already_run_iterations>0 else "iterations")
            for _ in range(num_iterations):
                start_time = time.time()
#                if sv.should_stop():
#                    break
                step = cnn.global_step.eval() 
                train_loss = cnn.run_train_epoch(sess, dataset, 0.5, summary_writer)
                
                savedpoint = ""
                if cnn.iterations % CHECKPOINTALL == 0 or cnn.iterations == config.iterations:
                    checkpoint_file = os.path.join(config.checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)                
                    savedpoint = "(checkpoint saved)"
                
                print('Iteration %3d (step %4d): loss = %.2f (%.3f sec)' % (cnn.iterations, step+1, train_loss, time.time()-start_time), savedpoint)
                
                
            ev, loss, _ = cnn.run_eval(sess, dataset)
            print("Result of evaluation:")
            print("Loss: %.2f,  Correct inferences: %.2f%%" % (loss, ev*100))

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
        
    trackingpoints = read_supervised.TPList(config.foldername)
    print("Number of samples:",trackingpoints.numsamples)  
    if Steer:
#        run_FFNNSteer_training(config, trackingpoints)            
        None
    else:
        run_CNN_training(config, trackingpoints)        
    
                
                
if __name__ == '__main__':    
    main()