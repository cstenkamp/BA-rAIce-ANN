# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:41:09 2017

@author: csten_000
"""
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #so that TF doesn't show its warnings
import time
import math
#====own classes====
import read_supervised
from myprint import myprint as print
import config 
from utils import convolutional_layer, fc_layer, variable_summary

SUMMARYALL = 1000

                                      
class CNN(object):
    
    #learning on gpu and application on cpu: https://stackoverflow.com/questions/44255362/tensorflow-simultaneous-prediction-on-gpu-and-cpu
    
    ######methods for BUILDING the computation graph######
    
    #since standard DQN consists of online and target-network, the latter of those will ONLY make inferences! If I want to run learning
    #and inference in parallel, I can simply have the online-net in mode "rl_train", and the target-net in "inference"
    def __init__(self, config, mode="sv_train", rl_not_trainables=[]):  #modes are "sv_train", "rl_train", "inference"
        #builds the computation graph, using the next few functions (this is basically the interface)
        self.config = config
        self.mode = mode
        final_neuron_num = self.config.steering_steps*4 if self.config.INCLUDE_ACCPLUSBREAK else self.config.steering_steps*3
        self.stacksize = self.config.history_frame_nr*2 if self.config.use_second_camera else self.config.history_frame_nr

        self.iterations = 0
        self.prepareNumIters()
        self.global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False) #wird hier: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py halt gemacht... warum hiernochmal weiß ich nicht.            
        if mode == "inference":
            device = "/gpu:0" if (config.has_gpu() and (hasattr(config, "learnMode") and config.learnMode == "between")) else "/cpu:0"
            with tf.device(device): #less overhead by not trying to switch to gpu
                self.inputs, self.targets, self.speed_input = self.set_placeholders(mode, final_neuron_num)
                self.q, self.onehot, self.q_max, self.action = self.inference(self.inputs, self.speed_input, final_neuron_num, rl_not_trainables, False) 
        else:
            device = "/gpu:0" if config.has_gpu() else "/cpu:0"
            with tf.device(device):
                self.inputs, self.targets, self.speed_input = self.set_placeholders(mode, final_neuron_num)
                self.q, self.onehot, self.q_max, self.action = self.inference(self.inputs, self.speed_input, final_neuron_num, rl_not_trainables, True)         
                if mode == "rl_train":
                    self.loss = self.rl_loss_func(self.q, self.targets)
                    self.train_op = self.training(self.loss, config.initial_lr, optimizer_arg = tf.train.RMSPropOptimizer)     
                elif mode == "sv_train":
                    self.loss = self.loss_func(self.q, self.targets)
                    self.train_op = self.training(self.loss, config.initial_lr, optimizer_arg = tf.train.AdamOptimizer) 
                    self.accuracy = self.evaluation(self.onehot, self.targets)    
                self.summary = tf.summary.merge_all() #für TensorBoard    
        
    
    def set_placeholders(self, mode, final_neuron_num):
        if self.config.history_frame_nr == 1:
            inputs = tf.placeholder(tf.float32, shape=[None, self.config.image_dims[0], self.config.image_dims[1]], name="inputs")  #first dim is none since inference has another batchsize than training
        else:
            inputs = tf.placeholder(tf.float32, shape=[None, self.stacksize, self.config.image_dims[0], self.config.image_dims[1]], name="inputs")  #first dim is none since inference has another batchsize than training
            
        speeds = tf.placeholder(tf.float32, shape=[None, self.config.speed_neurons], name="speed_inputs") if self.config.speed_neurons else None
        targets = None if mode=="inference" else tf.placeholder(tf.float32, shape=[None, final_neuron_num], name="targets")    
            
        return inputs, targets, speeds
    
    
    def inference(self, inputs, spinputs, final_neuron_num, rl_not_trainables, for_training=False):
        self.trainvars = {}

        def trainable(name):
            return True if self.mode=="sv_train" else not (name in rl_not_trainables)

        def settozero(q):
            q = tf.squeeze(q)
            if not self.config.INCLUDE_ACCPLUSBREAK: #dann nimmste nur das argmax von den mittleren neurons
                q = tf.slice(q,tf.shape(q)//3,tf.shape(q)//3)
                q = tf.concat([tf.multiply(tf.ones(tf.shape(q)),-50), q, tf.multiply(tf.ones(tf.shape(q)),-50)], axis=0)
            else:
                q = tf.slice(q,tf.shape(q)//2,(tf.shape(q)//4)*3)
                q = tf.concat([tf.multiply(tf.ones(tf.shape(q)*2), -50), q, tf.multiply(tf.ones(tf.shape(q)),-50)], axis=0)                   
            q = tf.expand_dims(q, 0)            
            return q

        rs_input = tf.reshape(inputs, [-1, self.config.image_dims[0], self.config.image_dims[1], self.stacksize]) #final dimension = number of color channels*number of stacked (history-)frames                  
        self.keep_prob = tf.Variable(tf.constant(1.0), trainable=False) #wenn nicht gefeedet ist sie standardmäßig 1        
        flat_size = math.ceil(self.config.image_dims[0]/4)*math.ceil(self.config.image_dims[1]/4)*64 #die /(2*2) ist wegen dem einen stride=2 
        ini = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.config.image_dims[0]*self.config.image_dims[1])))
        #convolutional_layer(input_tensor, input_channels, kernel_size, stride, output_channels, name, act, is_trainable, batchnorm, is_training, weightdecay=False, pool=True, trainvars=None, varSum=None, initializer=None)
        conv1 = convolutional_layer(rs_input, self.stacksize, [5,5], 1, 32, "Conv1", tf.nn.relu, trainable("Conv1"), False, for_training, False, True, self.trainvars, variable_summary, initializer=ini) #reduces to x//2*y//2
        conv2 = convolutional_layer(conv1, 32, [5,5], 1, 64, "Conv2", tf.nn.relu, trainable("Conv2"), False, for_training, False, True, self.trainvars, variable_summary, initializer=ini)                #reduces to x//4*y//4
        conv2_flat =  tf.reshape(conv2, [-1, flat_size])                                                                #x//4*y//4+speed_neurons
        #fc_layer(input_tensor, input_size, output_size, name, is_trainable, batchnorm, is_training, weightdecay=False, act=None, keep_prob=1, trainvars=None, varSum=None, initializer=None):
        fc1 = fc_layer(conv2_flat, flat_size, final_neuron_num*20, "FC1", trainable("FC1"), False, for_training, False, tf.nn.relu, 1 if for_training else self.keep_prob, self.trainvars, variable_summary, initializer=ini)                 
        if self.config.speed_neurons:
            fc1 = tf.concat([fc1, spinputs], 1)         #beim letztem layer btw kein dropout
        q = fc_layer(fc1, final_neuron_num*20+self.config.speed_neurons, final_neuron_num, "FC2", trainable("FC2"), False, for_training, False, None, 1, self.trainvars, variable_summary, initializer=ini) 

        q = tf.cond(tf.reduce_sum(spinputs) < 1, lambda: settozero(q), lambda: q)   #[10.3, 23.1, ...] #wenn du stehst, brauchste dich nicht mehr für die ohne gas zu interessieren
        y_conv = tf.nn.softmax(q)                                                   #[ 0.1,  0.2, ...]
        onehot = tf.one_hot(tf.argmax(y_conv, dimension=1), depth=final_neuron_num) #[   0,    1, ...]
        q_max = tf.reduce_max(q, axis=1)                                            #23.1
        action = tf.argmax(q, axis=1)                                               #2
        
        return q, onehot, q_max, action
    
    
    def loss_func(self, logits, targets):
        #tf.nn.softmax_cross_entropy_with_logits vergleicht 1-hot-labels mit integer-logits
        #"Note that tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the model's unnormalized model prediction and sums across all classes"
        #The raw formulation of cross-entropy (one line below) can be numerically unstable. -> use tf's function!
        #cross_entropy = -tf.reduce_sum(self.targets * tf.log(tf.nn.softmax(logits)), reduction_indices=[1])       
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        return tf.reduce_mean(cross_entropy)
        
    
    def rl_loss_func(self, q, q_target):
        return tf.reduce_mean(tf.square(q_target - q))
    
    
    
    def training(self, loss, init_lr, optimizer_arg):
        #returns the minimizer op
        variable_summary(loss, "loss") #tf.summary.scalar('loss', loss) #für TensorBoard
        
        self.learning_rate = tf.Variable(tf.constant(self.config.initial_lr), trainable=False)
        self.new_lr = tf.placeholder(tf.float32, shape=[]) #diese und die nächste zeile nur nötig falls man per extra-aufruf die lr verändern will, so wie ich das mache braucht man die nicht.
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)        

        if self.config.max_grad_norm > 0:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.config.max_grad_norm)
            
        optimizer = optimizer_arg(self.learning_rate)
        
        if (optimizer_arg == tf.train.RMSPropOptimizer):
            arguments = {"learning_rate": self.config.initial_lr}
            try:
                arguments["decay"] = self.config.lr_decay
            except:
                pass
            try: 
                arguments["momentum"] = self.config.rms_momentum
            except:
                pass
            try: 
                arguments["epsilon"] = self.config.min_sq_grad
            except:
                pass
            optimizer = optimizer_arg(**arguments)
        #TODO: AUSSUCHEN können welchen optimizer, und meinen ausgesuchten verteidigen können
        #https://www.tensorflow.org/api_guides/python/train#optimizers
        
        
        train_op = optimizer.minimize(loss, global_step=self.global_step)
        
        return train_op
    
        
    def evaluation(self, onehots, targets):
        #returns how many percent it got correct
        made = tf.cast(onehots, tf.bool)
        real = tf.cast(targets, tf.bool)
        compare = tf.reduce_mean(tf.cast(tf.reduce_all(tf.equal(made, real),axis=1), tf.float32))
        tf.summary.scalar('accuracy', compare)
        return compare
        
    
    def prepareNumIters(self):
        self.numIterations = tf.Variable(tf.constant(0), trainable=False)
        self.newIters = tf.placeholder(tf.int32, shape=[]) 
        self.iterUpdate = tf.assign(self.numIterations, self.newIters)           
    
    ######methods for RUNNING the computation graph######
    
    def saveNumIters(self, session, value):
        session.run(self.iterUpdate, feed_dict={self.newIters: value})
        
    def restoreNumIters(self, session):
        return self.numIterations.eval(session=session)
        
    
    
    def train_fill_feed_dict(self, config, dataset, batchsize = 0, decay_lr = True): 
        batchsize = config.batch_size if batchsize == 0 else batchsize
        _, visionvec, targets, speeds = dataset.next_batch(config, batchsize)
        if decay_lr:
            lr_decay = config.lr_decay ** max(self.iterations-config.lrdecayafter, 0.0)
            new_lr = max(config.initial_lr*lr_decay, config.minimal_lr)
        feed_dict = {self.inputs: visionvec, self.targets: targets, self.keep_prob: config.keep_prob, self.learning_rate: new_lr}
        if config.speed_neurons:
            feed_dict[self.speed_input] = speeds
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
            
    def calculate_value(self, session, visionvec, speed):
        visionvec = np.expand_dims(visionvec, axis=0)
        feed_dict = {self.inputs: visionvec}  
        if self.config.speed_neurons: #das config.speed_neurons muss anders... puhhhh....
           speed_disc = read_supervised.inflate_speed(speed, self.config.speed_neurons, self.config.SPEED_AS_ONEHOT)
           feed_dict[self.speed_input] = np.expand_dims(speed_disc, axis=0)
        
        return session.run(self.q_max, feed_dict=feed_dict)
            

##############################################################################################################################
    
    
    def rl_fill_feeddict(self, conv_inputs, other_inputs):
        if len(conv_inputs.shape) <= 3:
            feed_dict = {self.inputs: np.expand_dims(conv_inputs, axis=0)}  #expand_dims weil hier quasi batchsize=1 ist
            if self.config.speed_neurons:  #das config.speed_neurons muss anders... puhhhh.... #ES HEIßZT NICHT MEHR SPEED_INPUT SONDERN EINFACH OTHER-IPNUTS UND CONV-INPUTS
                feed_dict[self.speed_input] = np.expand_dims(other_inputs, axis=0)         
        else:
            feed_dict = {self.inputs: conv_inputs}
            if self.config.speed_neurons:
                feed_dict[self.speed_input] = other_inputs
        return feed_dict
        
        
    def run_inference(self, session, conv_inputs, other_inputs):        
        assert type(conv_inputs[0]).__module__ == np.__name__
        assert (np.array(conv_inputs.shape) == np.array(self.inputs.get_shape().as_list()[1:])).all()
        feed_dict = self.rl_fill_feeddict(conv_inputs, other_inputs)
        return session.run([self.onehot, self.q], feed_dict=feed_dict)

        
    
    def rl_learn_forward(self, session, conv_inputs, other_inputs, following_conv_inputs, following_other_inputs):
        qs = session.run(self.q, feed_dict = self.rl_fill_feeddict(conv_inputs, other_inputs)) 
        max_qs = session.run(self.q_max, feed_dict = self.rl_fill_feeddict(following_conv_inputs, following_other_inputs))
        return qs, max_qs


    def rl_learn_step(self, session, conv_inputs, other_inputs, qs):
        feed_dict = self.rl_fill_feeddict(conv_inputs, other_inputs)
        feed_dict[self.targets] = qs
        session.run(self.train_op, feed_dict=feed_dict)    
    
    
       
def run_svtraining(config, dataset):
    graph = tf.Graph()
    with graph.as_default():    
        #initializer = tf.constant_initializer(0.0) #bei variablescopes kann ich nen default-initializer für get_variables festlegen
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
                                             
        with tf.name_scope("train"):
            with tf.variable_scope("cnnmodel", reuse=None, initializer=initializer):
                cnn = CNN(config, mode="sv_train")
        
        init = tf.global_variables_initializer()
        cnn.trainvars["global_step"] = cnn.global_step #TODO: try to remove this and see if it still works, cause it should
        saver = tf.train.Saver(cnn.trainvars, max_to_keep=2)

        with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph) #aus dem toy-example
            
            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(config.checkpoint_dir) 
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                stepsPerIt = dataset.numsamples//config.batch_size
                already_run_iterations = cnn.global_step.eval()//stepsPerIt
                cnn.iterations = already_run_iterations
                print("Restored checkpoint with",already_run_iterations,"Iterations run already", level=8) 
            else:
                already_run_iterations = 0
                
            num_iterations = config.iterations - already_run_iterations
            print("Running for",num_iterations,"further iterations" if already_run_iterations>0 else "iterations", level=8)
            for _ in range(num_iterations):
                start_time = time.time()

                step = cnn.global_step.eval() 
                train_loss = cnn.run_train_epoch(sess, dataset, summary_writer)
                
                savedpoint = ""
                if cnn.iterations % config.checkpointall == 0 or cnn.iterations == config.iterations:
                    checkpoint_file = os.path.join(config.checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)       
                    savedpoint = "(checkpoint saved)"
                
                print('Iteration %3d (step %4d): loss = %.2f (%.3f sec)' % (cnn.iterations, step+1, train_loss, time.time()-start_time), savedpoint, level=8)
                
                
            ev, loss, _ = cnn.run_eval(sess, dataset)
            print("Result of evaluation:", level=8)
            print("Loss: %.2f,  Correct inferences: %.2f%%" % (loss, ev*100), level=8)

#            dataset.reset_batch()
#            _, visionvec, _, _ = dataset.next_batch(config, 1)
#            visionvec = np.array(visionvec[0])
#            print(cnn.run_inference(sess,visionvec, self.stacksize))
 



        
        
        
def main():
    conf = config.Config()
        
    trackingpoints = read_supervised.TPList(conf.LapFolderName, conf.use_second_camera, conf.msperframe, conf.steering_steps, conf.INCLUDE_ACCPLUSBREAK)
    print("Number of samples: %s | Tracking every %s ms with %s historyframes" % (trackingpoints.numsamples, str(conf.msperframe), str(conf.history_frame_nr)), level=6)
    run_svtraining(conf, trackingpoints)        
    
                
                
if __name__ == '__main__':    
    main()
    time.sleep(5)