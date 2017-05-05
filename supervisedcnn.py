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

SUMMARYALL = 1000
CHECKPOINTALL = 5


class Config(object):
    foldername = "SavedLaps/"
    log_dir = "SummaryLogDir/"  
    checkpoint_pre_dir = "Checkpoint"
    
    history_frame_nr = 4 #incl. dem jetzigem!
    steering_steps = 11
    image_dims = [30,42]
    vector_len = 59
    msperframe = 50 #50   #ACHTUNG!!! Dieser wert wird von unity überschrieben!!!!! #TODO: dass soll mit unity abgeglichen werden!
    
    batch_size = 32
    keep_prob = 0.8
    initscale = 0.1
    max_grad_norm = 10
    
    iterations = 60      #90, 120
    initial_lr = 0.005
    lr_decay = 0.9
    lrdecayafter = iterations//2  #//3 für 90, 120
    minimal_lr = 1e-5 #mit diesen settings kommt er auf 0.01 loss, 99.7% correct inferences
    
    def __init__(self):
        assert os.path.exists(self.foldername), "No data to train on at all!"        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)         
            
        self.checkpoint_dir = self.checkpoint_pre_dir + "_hframes"+str(self.history_frame_nr)+"_msperframe"+str(self.msperframe)+"/"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir) 



                   

#was fehlt am ANN:
#    -nutzen: Tensorflow website's stuff, Leons TF-Project
                                      
class CNN(object):
    
    ######methods for BUILDING the computation graph######
    def __init__(self, config, is_training=True):
        #builds the computation graph, using the next few functions (this is basically the interface)
        self.config = config
        self.iterations = 0
    
        self.inputs, self.targets = self.set_placeholders(is_training)
        
        if is_training:
            self.logits, self.argmaxs = self.inference(True)         
            self.loss = self.loss_func(self.logits)
            self.train_op = self.training(self.loss, config.initial_lr) 
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
    
    
    def set_placeholders(self, is_training):
        if self.config.history_frame_nr == 1:
            inputs = tf.placeholder(tf.float32, shape=[None, self.config.image_dims[0], self.config.image_dims[1]], name="inputs")  #first dim is none since inference has another batchsize than training
        else:
            inputs = tf.placeholder(tf.float32, shape=[None, self.config.history_frame_nr, self.config.image_dims[0], self.config.image_dims[1]], name="inputs")  #first dim is none since inference has another batchsize than training
            
        if is_training:
            targets = tf.placeholder(tf.float32, shape=[None, self.config.steering_steps*4], name="targets")    
        else:
            targets = None
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

        #rs_input = tf.reshape(self.inputs, [-1, self.config.image_dims[0], self.config.image_dims[1],1]) #final dimension = number of color channels
        rs_input = tf.reshape(self.inputs, [-1, self.config.image_dims[0], self.config.image_dims[1],self.config.history_frame_nr]) #final dimension = number of color channels
                             
        self.keep_prob = tf.Variable(tf.constant(1.0), trainable=False) #wenn nicht gefeedet ist sie standardmäßig 1        
        h1 = convolutional_layer(rs_input, self.config.history_frame_nr, 32, "Conv1", tf.nn.relu) #reduces to 15*21
        h2 = convolutional_layer(h1, 32, 64, "Conv2", tf.nn.relu)      #reduces to 8*11
        h_pool_flat =  tf.reshape(h2, [-1, 8*11*64])
        h_fc1 = fc_layer(h_pool_flat, 8*11*64, 1024, "FC1", tf.nn.relu, do_dropout=for_training)                 
        y_pre = fc_layer(h_fc1, 1024, self.config.steering_steps*4, "FC2", None, do_dropout=False) 
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
        
    def training(self, loss, init_lr):
        #returns the minimizer op
        self.variable_summary(loss, "loss") #tf.summary.scalar('loss', loss) #für TensorBoard
        
        self.learning_rate = tf.Variable(tf.constant(self.config.initial_lr), trainable=False)
        self.new_lr = tf.placeholder(tf.float32, shape=[]) #diese und die nächste zeile nur nötig falls man per extra-aufruf die lr verändern will, so wie ich das mache braucht man die nicht.
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)        

        if self.config.max_grad_norm > 0:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.config.max_grad_norm)
            
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
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
    def train_fill_feed_dict(self, config, dataset, batchsize = 0, decay_lr = True):
        batchsize = config.batch_size if batchsize == 0 else batchsize
        _, visionvec, targets, _ = dataset.next_batch(config, batchsize)
        if decay_lr:
            lr_decay = config.lr_decay ** max(self.iterations-config.lrdecayafter, 0.0)
            new_lr = max(config.initial_lr*lr_decay, config.minimal_lr)
        feed_dict = {self.inputs: visionvec, self.targets: targets, self.keep_prob: config.keep_prob, self.learning_rate: new_lr}
        return feed_dict            


    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})
     

    def run_train_epoch(self, session, dataset, summarywriter = None):
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
        KLAPPT NICHT BEI MEHREREN HISTORYFRAMES!!
        if not type(visionvec).__module__ == np.__name__:
            return False, None #dann ist das input-array leer
        assert (np.array(visionvec.shape) == np.array(self.inputs.get_shape().as_list()[1:])).all()
        visionvec = np.expand_dims(visionvec, axis=0)
        feed_dict = {self.inputs: visionvec}  
        return True, session.run(self.argmaxs, feed_dict=feed_dict)

            
       
def run_CNN_training(config, dataset):
    graph = tf.Graph()
    with graph.as_default():    
        #initializer = tf.constant_initializer(0.0) #bei variablescopes kann ich nen default-initializer für get_variables festlegen
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
                                             
        with tf.name_scope("train"):
            with tf.variable_scope("cnnmodel", reuse=None, initializer=initializer):
                cnn = CNN(config, is_training=True)
        
        init = tf.global_variables_initializer()
        cnn.trainvars["global_step"] = cnn.global_step
        saver = tf.train.Saver(cnn.trainvars, max_to_keep=3)

        with tf.Session(graph=graph) as sess:
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

                step = cnn.global_step.eval() 
                train_loss = cnn.run_train_epoch(sess, dataset, summary_writer)
                
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
 



        
        
        
def main(Steer=False):
    config = Config()
        
    trackingpoints = read_supervised.TPList(config.foldername, config.msperframe)
    print("Number of samples: %s | Tracking every %s ms with %s historyframes" % (trackingpoints.numsamples, str(config.msperframe), str(config.history_frame_nr)))
    run_CNN_training(config, trackingpoints)        
    
                
                
if __name__ == '__main__':    
    main()
    time.sleep(5)