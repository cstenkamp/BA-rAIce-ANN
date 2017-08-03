# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:41:09 2017

@author: csten_000
"""
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #so that TF doesn't show its warnings
import math
#====own classes====
from myprint import myprint as print
from utils import convolutional_layer, fc_layer, variable_summary


###############################################################################
###############################################################################
                                      
class CNN(object): #learning on gpu and application on cpu: https://stackoverflow.com/questions/44255362/tensorflow-simultaneous-prediction-on-gpu-and-cpu
    ######methods for BUILDING the computation graph######
    
    #since standard DQN consists of online and target-network, the latter of those will ONLY make inferences! If I want to run learning
    #and inference in parallel, I can simply have the online-net in mode "rl_train", and the target-net in "inference"
    def __init__(self, config, agent, mode="sv_train", rl_not_trainables=[]):  #modes are "sv_train", "rl_train", "inference"
        #builds the computation graph, using the next few functions (this is basically the interface)
        self.config = config
        self.mode = mode
        self.agent = agent
        final_neuron_num = self.config.steering_steps*4 if self.config.INCLUDE_ACCPLUSBREAK else self.config.steering_steps*3
        self.conv_stacksize = (self.config.history_frame_nr*2 if self.config.use_second_camera else self.config.history_frame_nr) if self.agent.conv_stacked else 1
        self.ff_stacksize = self.config.history_frame_nr if self.agent.ff_stacked else 1
        self.stood_frames_ago = 0 #das wird benutzt damit er, wenn er einmal stand, sich merken kann ob erst kurz her ist (für settozero)

        self.sv_iterations = 0
        self.sv_global_step = tf.Variable(0, dtype=tf.int32, name='sv_global_step', trainable=False)
        self._prepareNumIters()
        self.global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False) #https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py 
            
        ####
        self.conv_inputs, self.ff_inputs, self.targets, self.stands_inputs = self._set_placeholders(mode, final_neuron_num)
        self.q, self.onehot, self.q_max, self.action = self._inference(self.conv_inputs, self.ff_inputs, final_neuron_num, rl_not_trainables, True)   
        self.q_targets = tf.placeholder(tf.float32, shape=[None, final_neuron_num], name="q_targets")
        self.loss = tf.reduce_mean(tf.square(self.q_targets - self.q))
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.00001)
        self.train_op = self.trainer.minimize(self.loss)
        self.accuracy = self._evaluation(self.onehot, self.targets)  #TODO: DELETE THIS LINE!!!!


#        if mode == "inference":
#            device = "/gpu:0" if (config.has_gpu() and (hasattr(config, "learnMode") and config.learnMode == "between")) else "/cpu:0"
#            with tf.device(device): #less overhead by not trying to switch to gpu
#                self.conv_inputs, self.ff_inputs, self.targets, self.stands_inputs = self._set_placeholders(mode, final_neuron_num)
#                self.q, self.onehot, self.q_max, self.action = self._inference(self.conv_inputs, self.ff_inputs, final_neuron_num, rl_not_trainables, False, self.stands_inputs) 
#                self.accuracy = self._evaluation(self.onehot, self.targets)  #TODO: DELETE THIS LINE!!!!
#        else:
#            device = "/gpu:0" if config.has_gpu() else "/cpu:0"
#            with tf.device(device):
#                self.conv_inputs, self.ff_inputs, self.targets, self.stands_inputs = self._set_placeholders(mode, final_neuron_num)
#                self.q, self.onehot, self.q_max, self.action = self._inference(self.conv_inputs, self.ff_inputs, final_neuron_num, rl_not_trainables, True)         
#                if mode == "rl_train":
#                    self.q_targets = tf.placeholder(tf.float32, shape=[None, final_neuron_num], name="q_targets")
#                    self.loss = self._rl_loss_func(self.q, self.q_targets)
#                    self.train_op = self._training(self.loss, config.initial_lr, self.global_step, optimizer_arg = tf.train.RMSPropOptimizer)     
#                    self.accuracy = self._evaluation(self.onehot, self.targets)  #TODO: DELETE THIS LINE!!!!
#                elif mode == "sv_train":
#                    self.loss = self._loss_func(self.q, self.targets)
#                    self.train_op = self._training(self.loss, config.initial_lr, self.sv_global_step, optimizer_arg = tf.train.AdamOptimizer) 
#                    self.accuracy = self._evaluation(self.onehot, self.targets)    
#        self.summary = tf.summary.merge_all() #für TensorBoard    
        
    
    def _set_placeholders(self, mode, final_neuron_num):
        conv_inputs = tf.placeholder(tf.float32, shape=[None, self.conv_stacksize, self.config.image_dims[0], self.config.image_dims[1]], name="conv_inputs")  if self.agent.usesConv else None
        ff_inputs = tf.placeholder(tf.float32, shape=[None, self.ff_stacksize*self.agent.ff_inputsize], name="ff_inputs") if self.agent.ff_inputsize else None
        #targets = None if mode=="inference" else tf.placeholder(tf.float32, shape=[None, final_neuron_num], name="sv_targets")    
        targets = tf.placeholder(tf.float32, shape=[None, final_neuron_num], name="sv_targets")    #TODO: DELETE THIS LINE, UND PACKE DAFÜR DIE VORHERIGE WIEDER HIER HIN!!!
        stands_inputs = tf.placeholder(tf.float32, shape=[None], name="standing_inputs") #necessary for settozero
        return conv_inputs, ff_inputs, targets, stands_inputs
    
    
    def _inference(self, conv_inputs, ff_inputs, final_neuron_num, rl_not_trainables, for_training=False, stands_inputs=False): #stands_inputs existiert nur bei inference, sonst ists eh immer 0
        assert(conv_inputs is not None or ff_inputs is not None)
        self.trainvars = {}
        flat_size = 0

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
        
        ini = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self.config.image_dims[0]*self.config.image_dims[1])))
        self.keep_prob = tf.Variable(tf.constant(1.0), trainable=False) #wenn nicht gefeedet ist sie standardmäßig 1        
            
        if conv_inputs is not None:
            rs_input = tf.reshape(conv_inputs, [-1, self.config.image_dims[0], self.config.image_dims[1], self.conv_stacksize]) #final dimension = number of color channels*number of stacked (history-)frames                  
            flat_size = math.ceil(self.config.image_dims[0]/4)*math.ceil(self.config.image_dims[1]/4)*64 #die /(2*2) ist wegen dem einen stride=2 
            #convolutional_layer(input_tensor, input_channels, kernel_size, stride, output_channels, name, act, is_trainable, batchnorm, is_training, weightdecay=False, pool=True, trainvars=None, varSum=None, initializer=None)
            conv1 = convolutional_layer(rs_input, self.conv_stacksize, [5,5], 1, 32, "Conv1", tf.nn.relu, trainable("Conv1"), False, for_training, False, True, self.trainvars, variable_summary, initializer=ini) #reduces to x//2*y//2
            conv2 = convolutional_layer(conv1, 32, [5,5], 1, 64, "Conv2", tf.nn.relu, trainable("Conv2"), False, for_training, False, True, self.trainvars, variable_summary, initializer=ini)                #reduces to x//4*y//4
            conv2_flat =  tf.reshape(conv2, [-1, flat_size])    #x//4*y//4+speed_neurons
            if ff_inputs is not None:
                fc0 = tf.concat([conv2_flat, ff_inputs], 1) 
        else:
            fc0 = fc_layer(ff_inputs, self.ff_stacksize*self.agent.ff_inputsize, self.agent.ff_inputsize, "FC0", trainable("FC0"), False, for_training, False, tf.nn.relu, 1 if for_training else self.keep_prob, self.trainvars, variable_summary, initializer=ini)   
        flat_size += self.agent.ff_inputsize    
        
        #fc_layer(input_tensor, input_size, output_size, name, is_trainable, batchnorm, is_training, weightdecay=False, act=None, keep_prob=1, trainvars=None, varSum=None, initializer=None):
        fc1 = fc_layer(fc0, flat_size, final_neuron_num*20, "FC1", trainable("FC1"), False, for_training, False, tf.nn.relu, 1 if for_training else self.keep_prob, self.trainvars, variable_summary, initializer=ini)                 
        q = fc_layer(fc1, final_neuron_num*20, final_neuron_num, "FC2", trainable("FC2"), False, for_training, False, None, 1, self.trainvars, variable_summary, initializer=ini) 

#        if self.mode == "inference":
#            q = tf.cond(tf.reduce_sum(stands_inputs) > 0, lambda: settozero(q), lambda: q) #[10.3, 23.1, ...] #wenn du stehst, brauchste dich nicht mehr für die ohne gas zu interessieren
        y_conv = tf.nn.softmax(q)                                                          #[ 0.1,  0.2, ...]
        onehot = tf.one_hot(tf.argmax(y_conv, dimension=1), depth=final_neuron_num)        #[   0,    1, ...]
        q_max = tf.reduce_max(q, axis=1)                                                   #23.1
        action = tf.argmax(q, axis=1)                                                      #2
        
        return q, onehot, q_max, action
    
    
    def _loss_func(self, logits, targets):
        #tf.nn.softmax_cross_entropy_with_logits vergleicht 1-hot-labels mit integer-logits
        #"Note that tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the model's unnormalized model prediction and sums across all classes"
        #The raw formulation of cross-entropy (one line below) can be numerically unstable. -> use tf's function!
        #cross_entropy = -tf.reduce_sum(self.targets * tf.log(tf.nn.softmax(logits)), reduction_indices=[1])       
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        return tf.reduce_mean(cross_entropy)
        
    
    def _rl_loss_func(self, q, q_target):
        return tf.reduce_mean(tf.square(q_target - q))
    
    
    
    def _training(self, loss, init_lr, stepcounter, optimizer_arg):
        #returns the minimizer op
        variable_summary(loss, "loss") #tf.summary.scalar('loss', loss) #für TensorBoard
        
        self.learning_rate = tf.Variable(tf.constant(self.config.initial_lr), trainable=False)
        self.new_lr = tf.placeholder(tf.float32, shape=[]) #diese und die nächste zeile nur nötig falls man per extra-aufruf die lr verändern will, so wie ich das mache braucht man die nicht.
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)        

#        if self.config.max_grad_norm > 0:
#            tvars = tf.trainable_variables()
#            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.config.max_grad_norm)
          
        optimizer = optimizer_arg(0.000005)  
        #optimizer = optimizer_arg(0.0000005) #klappt, though slow
        
#        if (optimizer_arg == tf.train.RMSPropOptimizer):
#            kwargs = {"learning_rate": self.config.initial_lr}
#            try:
#                kwargs["decay"] = self.config.lr_decay
#            except:
#                pass
#            try: 
#                kwargs["momentum"] = self.config.rms_momentum
#            except:
#                pass
#            try: 
#                kwargs["epsilon"] = self.config.min_sq_grad
#            except:
#                pass
#            optimizer = optimizer_arg(**kwargs)
        #TODO: AUSSUCHEN können welchen optimizer, und meinen ausgesuchten verteidigen können
        #https://www.tensorflow.org/api_guides/python/train#optimizers
        
        train_op = optimizer.minimize(loss, global_step=stepcounter)
        
        return train_op
    
        
    def _evaluation(self, onehots, targets):
        #returns how many percent it got correct
        made = tf.cast(onehots, tf.bool)
        real = tf.cast(targets, tf.bool)
        compare = tf.reduce_mean(tf.cast(tf.reduce_all(tf.equal(made, real),axis=1), tf.float32))
        tf.summary.scalar('accuracy', compare)
        return compare
        
    
    def _prepareNumIters(self):
        self.numIterations = tf.Variable(tf.constant(0), trainable=False)
        self.newIters = tf.placeholder(tf.int32, shape=[]) 
        self.iterUpdate = tf.assign(self.numIterations, self.newIters)           

###############################################################################
    ######methods for RUNNING the computation graph######
###############################################################################
    
    def saveNumIters(self, session, value):
        session.run(self.iterUpdate, feed_dict={self.newIters: value})
        
    def restoreNumIters(self, session):
        return self.numIterations.eval(session=session)
        
    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})
        
    
    def sv_fill_feed_dict(self, config, conv_inputs, other_inputs, targets, decay_lr = True, dropout = True): 
        feed_dict = {self.targets: targets}
#        feed_dict[self.keep_prob] = config.keep_prob if dropout else None
#        if decay_lr:
#            lr_decay = config.lr_decay ** max(self.sv_iterations-config.lrdecayafter, 0.0)
#            new_lr = max(config.initial_lr*lr_decay, config.minimal_lr)
#            feed_dict[self.learning_rate] = new_lr
        if self.agent.usesConv:
            feed_dict[self.conv_inputs] = conv_inputs
        if self.agent.ff_inputsize:
            feed_dict[self.ff_inputs] = other_inputs
        return feed_dict            


    def run_sv_train_step(self, session, agent, stateBatch, summarywriter = None):
        with self.agent.graph.as_default():
            conv_inputs, other_inputs, _ = self.EnvStateBatch_to_AgentStateBatch(agent, stateBatch)
            targets = self.EnvStateBatch_to_AgentActionBatch(agent, stateBatch)
            feed_dict = self.sv_fill_feed_dict(self.config, conv_inputs, other_inputs, targets)
            if self.config.summaryall and self.sv_iterations % self.config.summaryall == 0 and summarywriter is not None:
                _, loss, summary_str = session.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)   
                summarywriter.add_summary(summary_str, session.run(self.global_step))
                summarywriter.flush()     
            else:
                _, loss = session.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss


    def run_sv_eval(self, session, agent, stateBatch):    
        conv_inputs, other_inputs, _ = self.EnvStateBatch_to_AgentStateBatch(agent, stateBatch)
        targets = self.EnvStateBatch_to_AgentActionBatch(agent, stateBatch)
        feed_dict = self.sv_fill_feed_dict(self.config, conv_inputs, other_inputs, targets)        
#        accuracy, loss = session.run([self.accuracy, self.loss], feed_dict=feed_dict)
#        return accuracy, loss, len(stateBatch)
        accuracy = session.run(self.accuracy, feed_dict=feed_dict)
        return accuracy, accuracy, accuracy
           
    
##################### RL-stuff ################################################
    
    
    def rl_fill_feeddict(self, conv_inputs, other_inputs, stands_inputs=False):
        
        def is_inference(conv_inputs, other_inputs):
            return len(conv_inputs.shape) <= 3 if conv_inputs is not None else len(other_inputs.shape) <= 2
        
        feed_dict = {}
        if is_inference(conv_inputs, other_inputs):   
            self.stood_frames_ago = 0 if stands_inputs else self.stood_frames_ago + 1
            if self.stood_frames_ago < 10:
                stands_inputs = True
            conv_inputs = np.expand_dims(conv_inputs, axis=0) #expand_dims weil hier quasi batchsize=1
            other_inputs= np.expand_dims(other_inputs, axis=0) 
            stands_inputs = np.expand_dims(stands_inputs, axis=0)
        else:
            stands_inputs = [stands_inputs]*other_inputs.shape[0]
        if self.agent.usesConv:
            feed_dict[self.conv_inputs] = conv_inputs
        if self.agent.ff_inputsize:
            feed_dict[self.ff_inputs] = other_inputs 
        feed_dict[self.stands_inputs] = stands_inputs #ist halt 0 wenn false, was richtig ist
        return feed_dict
        
        
    def run_inference(self, session, conv_inputs, other_inputs, stands_inputs=False):        
        if conv_inputs is not None:
            assert type(conv_inputs[0]).__module__ == np.__name__
            assert (np.array(conv_inputs.shape) == np.array(self.conv_inputs.get_shape().as_list()[1:])).all()
        feed_dict = self.rl_fill_feeddict(conv_inputs, other_inputs, stands_inputs)
        return session.run([self.onehot, self.q], feed_dict=feed_dict)

    
    def calculate_value(self, session, conv_inputs, other_inputs, stands_inputs=False):
        feed_dict = self.rl_fill_feeddict(conv_inputs, other_inputs, stands_inputs)
        return session.run(self.q_max, feed_dict=feed_dict)
    
        
    def rl_learn_forward(self, session, conv_inputs, other_inputs, following_conv_inputs, following_other_inputs):
        qs = session.run(self.q, feed_dict = self.rl_fill_feeddict(conv_inputs, other_inputs)) 
        max_qs = session.run(self.q_max, feed_dict = self.rl_fill_feeddict(following_conv_inputs, following_other_inputs))
        return qs, max_qs


    def rl_learn_step(self, session, conv_inputs, other_inputs, qs):
        feed_dict = self.rl_fill_feeddict(conv_inputs, other_inputs)
        feed_dict[self.q_targets] = qs
        session.run(self.train_op, feed_dict=feed_dict)    
    
    
############################## helper functions ###############################
    
    #takes as input a batch of ENV-STATES, and returns batch of AGENT-STATES
    def EnvStateBatch_to_AgentStateBatch(self, agent, stateBatch):
        presentStates = list(zip(*stateBatch))
        conv_inputs, other_inputs, _ = list(zip(*[agent.getAgentState(*presentState) for presentState in presentStates]))
        other_inputs = [agent.makeNetUsableOtherInputs(i) for i in other_inputs]
        return conv_inputs, other_inputs, False
        
    #takes as input batch of ENV-STATES, and return batch of AGENT-ACTIONS
    def EnvStateBatch_to_AgentActionBatch(self, agent, stateBatch):
        presentStates = list(zip(*stateBatch))
        targets = [agent.makeNetUsableAction(agent.getAction(*presentState)) for presentState in presentStates]
        return targets
       