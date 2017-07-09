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

SUMMARYALL = 1000

class Config(object):
    foldername = "SavedLaps/"
    log_dir = "SummaryLogDir/"  
    checkpoint_pre_dir = "Checkpoint"
    
    history_frame_nr = 4 #incl. dem jetzigem!
    speed_neurons = 30 #wenn null nutzt er sie nicht
    SPEED_AS_ONEHOT = False
    steering_steps = 7
    INCLUDE_ACCPLUSBREAK = False
    
    reset_if_wrongdirection = True
    
    image_dims = [30,45] 
    msperframe = 200 #50   #ACHTUNG!!! Dieser wert wird von unity überschrieben!!!!! #TODO: dass soll mit unity abgeglichen werden!
    
    batch_size = 32
    keep_prob = 0.8
    initscale = 0.1
    max_grad_norm = 10
    
    iterations = 90     #90, 120
    initial_lr = 0.005
    lr_decay = 0.9
    lrdecayafter = iterations//2  #//3 für 90, 120
    minimal_lr = 1e-6 #mit diesen settings kommt er auf 0.01 loss, 99.7% correct inferences
    checkpointall = 10
    
    def __init__(self):
        assert os.path.exists(self.foldername), "No data to train on at all!"        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)         
            
        self.checkpoint_dir = self.checkpoint_pre_dir + "_hframes"+str(self.history_frame_nr)+"_msperframe"+str(self.msperframe)+"/"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir) 


class RL_Config(Config):
    log_dir = "SummaryLogDirRL/"  
    checkpoint_dir = "RL_Learn/"
    savememorypath = "./" #will be a pickle-file
    saveMemoryAllMins = 150
    
    keep_prob = 1
    max_grad_norm = 10
    initial_lr = 0.001
    #lr_decay = 1
    
    startepsilon = 0.2
    epsilondecrease = 0.0001
    minepsilon = 0.005
    batchsize = 32
    q_decay = 0.99
    checkpointall = 500
    copy_target_all = 100
    
    replaystartsize = 0
    memorysize = 30000
    useprecisebuthugememory = True
    learnAllXInferences = False
    CheckLearnInfRateAll = False #je höher dieser wert ist desto seltener checkt er die learnAllXInferences-ratio, desto seltener freezed er.
    
    #re-uses history_frame_nr, image_dims, steering_steps, speed_neurons, INCLUDE_ACCPLUSBREAK, SPEED_AS_ONEHOT
    
    def __init__(self):     
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)     
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)                 
                                    
        assert os.path.exists(Config().checkpoint_dir), "I need a pre-trained model"



    
class DQN_Config(RL_Config):
#    batch_size = 32                 #minibatch size
#    memorysize = 1000000            #replay memory size
#    history_frame_nr = 4            #agent history length
#    copy_target_all = 10000         #target network update frequency (C)
#    q_decay = 0.99                  #discount factor
#    #action repeat & update frequency & noop-max
#    initial_lr = 0.00025            #learning rate used by RMSProp
#    lr_decay = 1                    #as the lr seems to stay equal, no decay
#    rms_momentum = 0.95             #gradient momentum (=squared gradient momentum)
#    min_sq_grad = 0.1               #min squared gradient 
#    startepsilon = 1                #initial exploration
#    minepsilon = 0.1                #final exploration
#    finalepsilonframe = 1000000     #final exploration frame
#    replaystartsize = 50000         #replay start size
#    train_for = 50000000            #number of iterations to train for 
#    ForEveryInf, ComesALearn = 4, 1 #update frequency & how often it checks it


    batch_size = 32             #minibatch size
    memorysize = 800          #replay memory size
    history_frame_nr = 4        #agent history length
    copy_target_all = 100       #target network update frequency (C)
    q_decay = 0.99              #discount factor
    #action repeat & update frequency & noop-max
    initial_lr = 0.00025        #learning rate used by RMSProp
    lr_decay = 1                #as the lr seems to stay equal, no decay
    rms_momentum = 0.95         #gradient momentum (=squared gradient momentum)
    min_sq_grad = 0.1           #min squared gradient 
    startepsilon = 0.1            #initial exploration
    minepsilon = 0.1            #final exploration
    finalepsilonframe = 1000   #final exploration frame
    replaystartsize = 33       #replay start size
    train_for = 2700           #number of iterations to train for 
    ForEveryInf, ComesALearn = 30, 10
    
    def __init__(self):
        super().__init__()
    

                                      
class CNN(object):
    
    ######methods for BUILDING the computation graph######
    def __init__(self, config, is_reinforcement, is_training=True, rl_not_trainables=[]):
        #builds the computation graph, using the next few functions (this is basically the interface)
        self.config = config
        self.is_reinforcement = is_reinforcement
        self.iterations = 0
        final_neuron_num = self.config.steering_steps*4 if self.config.INCLUDE_ACCPLUSBREAK else self.config.steering_steps*3
        
        self.prepareNumIters()
        self.inputs, self.targets, self.speed_input = self.set_placeholders(is_training, final_neuron_num)
        
        if is_training:
            self.q, self.argmax, self.q_max, self.action = self.inference(self.inputs, self.speed_input, final_neuron_num, rl_not_trainables, True)         
            if is_reinforcement:
                self.loss = self.rl_loss_func(self.q, self.targets)
                self.train_op = self.training(self.loss, config.initial_lr, optimizer_arg = tf.train.RMSPropOptimizer)     
            else:
                self.loss = self.loss_func(self.q, self.targets)
                self.train_op = self.training(self.loss, config.initial_lr, optimizer_arg = tf.train.AdamOptimizer) 
                self.accuracy = self.evaluation(self.argmax, self.targets)    
            self.summary = tf.summary.merge_all() #für TensorBoard
        else:
            with tf.device("/cpu:0"): #less overhead by not trying to switch to gpu
                self.q, self.argmax, self.q_max, self.action = self.inference(self.inputs, self.speed_input, final_neuron_num, rl_not_trainables, False) 
            
            
    def variable_summary(self, var, what=""):
      with tf.name_scope('summaries_'+what):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
    
    
    def set_placeholders(self, is_training, final_neuron_num):
        if self.config.history_frame_nr == 1:
            inputs = tf.placeholder(tf.float32, shape=[None, self.config.image_dims[0], self.config.image_dims[1]], name="inputs")  #first dim is none since inference has another batchsize than training
        else:
            inputs = tf.placeholder(tf.float32, shape=[None, self.config.history_frame_nr, self.config.image_dims[0], self.config.image_dims[1]], name="inputs")  #first dim is none since inference has another batchsize than training
            
        if is_training:
            targets = tf.placeholder(tf.float32, shape=[None, final_neuron_num], name="targets")    
        else:
            targets = None
            
        if self.config.speed_neurons:
            speeds = tf.placeholder(tf.float32, shape=[None, self.config.speed_neurons], name="speed_inputs")
        else:
            speeds = None
            
        return inputs, targets, speeds
    
    
    def inference(self, inputs, spinputs, final_neuron_num, rl_not_trainables, for_training=False):
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
                is_trainable = True if not self.is_reinforcement else not (name in rl_not_trainables)
                if is_trainable:
                    self.trainvars["W_%s" % name] = weight_variable([5, 5, input_channels, output_channels], "W_%s" % name, is_trainable=True)
                    self.variable_summary(self.trainvars["W_%s" % name])
                    self.trainvars["b_%s" % name] = bias_variable([output_channels], "b_%s" % name, is_trainable=True)
                    self.variable_summary(self.trainvars["b_%s" % name])
                    h_act = act(conv2d(input_tensor, self.trainvars["W_%s" % name]) + self.trainvars["b_%s" % name])
                else:
                    W = weight_variable([5, 5, input_channels, output_channels], "W_%s" % name, is_trainable=False)
                    b = bias_variable([output_channels], "b_%s" % name, is_trainable=False)
                    h_act = act(conv2d(input_tensor, W) + b)                
                h_pool = max_pool_2x2(h_act)
                tf.summary.histogram("activations", h_pool)
                return h_pool
        
        def fc_layer(input_tensor, input_size, output_size, name, act=None, do_dropout=False):
            with tf.name_scope(name):
                is_trainable = True if not self.is_reinforcement else not (name in rl_not_trainables)
                self.trainvars["W_%s" % name] = weight_variable([input_size, output_size], "W_%s" % name, is_trainable=is_trainable)
                self.variable_summary(self.trainvars["W_%s" % name])
                self.trainvars["b_%s" % name] = bias_variable([output_size], "b_%s" % name, is_trainable=is_trainable)
                self.variable_summary(self.trainvars["b_%s" % name])
                h_fc =  tf.matmul(input_tensor, self.trainvars["W_%s" % name]) + self.trainvars["b_%s" % name]
                if act is not None:
                    h_fc = act(h_fc)
                tf.summary.histogram("activations", h_fc)
                if do_dropout:
                    h_fc = tf.nn.dropout(h_fc, self.keep_prob) 
                return h_fc

        rs_input = tf.reshape(inputs, [-1, self.config.image_dims[0], self.config.image_dims[1],self.config.history_frame_nr]) #final dimension = number of color channels
                             
        self.keep_prob = tf.Variable(tf.constant(1.0), trainable=False) #wenn nicht gefeedet ist sie standardmäßig 1        
        h1 = convolutional_layer(rs_input, self.config.history_frame_nr, 32, "Conv1", tf.nn.relu) #reduces to x//2*y//2
        h2 = convolutional_layer(h1, 32, 64, "Conv2", tf.nn.relu)      #reduces to x//4*y//4
        h_pool_flat =  tf.reshape(h2, [-1, math.ceil(self.config.image_dims[0]/4)*math.ceil(self.config.image_dims[1]/4)*64])
        h_fc1 = fc_layer(h_pool_flat, math.ceil(self.config.image_dims[0]/4)*math.ceil(self.config.image_dims[1]/4)*64, final_neuron_num*20, "FC1", tf.nn.relu, do_dropout=for_training)                 
        if self.config.speed_neurons:
            h_fc1 = tf.concat([h_fc1, spinputs], 1)   #its lengths is now in any case 1024+speed_neurons
        q = fc_layer(h_fc1, final_neuron_num*20+self.config.speed_neurons, final_neuron_num, "FC2", None, do_dropout=False) 


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

        q = tf.cond(tf.reduce_sum(spinputs) < 1, lambda: settozero(q), lambda: q)#wenn du stehst, brauchste dich nicht mehr für die ohne gas zu interessieren

        q_max = tf.reduce_max(q, axis=1)
        action = tf.argmax(q, axis=1) #Todo: kann gut sein dass ich action nicht brauche wenn ich argm hab
        y_conv = tf.nn.softmax(q)
        argm = tf.one_hot(tf.argmax(y_conv, dimension=1), depth=final_neuron_num)
        
        return q, argm, q_max, action
    
    
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
        self.variable_summary(loss, "loss") #tf.summary.scalar('loss', loss) #für TensorBoard
        
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
            
            
    def run_inference(self, session, visionvec, otherinputs, hframes):
        if hframes > 1:
            if not type(visionvec[0]).__module__ == np.__name__:
                return False, (None, None) #dann ist das input-array leer
        else:
            if not type(visionvec).__module__ == np.__name__:
                return False, (None, None) #dann ist das input-array leer
            assert (np.array(visionvec.shape) == np.array(self.inputs.get_shape().as_list()[1:])).all()
            
        
        with tf.device("/cpu:0"):
            visionvec = np.expand_dims(visionvec, axis=0)
            feed_dict = {self.inputs: visionvec}  
            if self.config.speed_neurons:
                speed_disc = read_supervised.inflate_speed(otherinputs.SpeedSteer.velocity, self.config.speed_neurons, self.config.SPEED_AS_ONEHOT)
                feed_dict[self.speed_input] = np.expand_dims(speed_disc, axis=0)
            
            return True, session.run([self.argmax, self.q], feed_dict=feed_dict)
        
        
        
    def calculate_value(self, session, visionvec, speed, hframes):
        with tf.device("/cpu:0"):
            visionvec = np.expand_dims(visionvec, axis=0)
            feed_dict = {self.inputs: visionvec}  
            if self.config.speed_neurons:
               speed_disc = read_supervised.inflate_speed(speed, self.config.speed_neurons, self.config.SPEED_AS_ONEHOT)
               feed_dict[self.speed_input] = np.expand_dims(speed_disc, axis=0)
            
            return session.run(self.q_max, feed_dict=feed_dict)
            
        
       
def run_svtraining(config, dataset):
    graph = tf.Graph()
    with graph.as_default():    
        #initializer = tf.constant_initializer(0.0) #bei variablescopes kann ich nen default-initializer für get_variables festlegen
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
                                             
        with tf.name_scope("train"):
            with tf.variable_scope("cnnmodel", reuse=None, initializer=initializer):
                cnn = CNN(config, is_reinforcement=False, is_training=True)
        
        init = tf.global_variables_initializer()
        cnn.trainvars["global_step"] = cnn.global_step #TODO: try to remove this and see if it still works, cause it should
        saver = tf.train.Saver(cnn.trainvars, max_to_keep=2)

        with tf.Session(graph=graph) as sess:
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
#            print(cnn.run_inference(sess,visionvec, config.history_frame_nr))
 



        
        
        
def main():
    config = Config()
        
    trackingpoints = read_supervised.TPList(config.foldername, config.msperframe, config.steering_steps, config.INCLUDE_ACCPLUSBREAK)
    print("Number of samples: %s | Tracking every %s ms with %s historyframes" % (trackingpoints.numsamples, str(config.msperframe), str(config.history_frame_nr)), level=6)
    run_svtraining(config, trackingpoints)        
    
                
                
if __name__ == '__main__':    
    main()
    time.sleep(5)