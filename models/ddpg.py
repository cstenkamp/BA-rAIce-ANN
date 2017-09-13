import tensorflow as tf
import numpy as np
import time
import tensorflow.contrib.slim as slim
from myprint import myprint as print
from utils import netCopyOps
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #so that TF doesn't show its warnings
from tensorflow.contrib.framework import get_variables
from utils import variable_summary, dense
flatten = lambda l: [item for sublist in l for item in sublist]


#batchnorm doesnt really work, and if, only with huge minibatches https://www.reddit.com/r/MachineLearning/comments/671455/d_batch_normalization_in_reinforcement_learning/

      
        
class conv_actorNet():
     def __init__(self, conf, agent, outerscope="actor", name="online", batchnorm = "ffffftt"):   #tffffft    
        tanh_min_bounds,tanh_max_bounds = np.array([-1]), np.array([1])
        min_bounds, max_bounds = np.array(list(zip(*conf.action_bounds))) 
        self.name = name
        self.conf = conf
        self.agent = agent
        decay = False #"For Q we included L2 weight decay", not for müh
        conv_stacksize = self.conf.conv_stacksize if self.agent.conv_stacked else 1        
        ff_stacksize = self.conf.ff_stacksize if self.agent.ff_stacked else 1

        with tf.variable_scope(name):
            self.conv_inputs = tf.placeholder(tf.float32, shape=[None, self.conf.image_dims[0], self.conf.image_dims[1], conv_stacksize], name="conv_inputs")  
            self.ff_inputs =   tf.placeholder(tf.float32, shape=[None, ff_stacksize*self.agent.ff_inputsize], name="ff_inputs")  
            self.stands_input = tf.placeholder(tf.bool, name="stands_input") #necessary for settozero            
            self.phase = tf.placeholder(tf.bool, name='phase') #for batchnorm, true heißt is_training

            rs_input = tf.reshape(self.conv_inputs, [-1, self.conf.image_dims[0], self.conf.image_dims[1], conv_stacksize]) #final dimension = number of color channels*number of stacked (history-)frames                  
            if batchnorm[0]=="t":
                rs_input = tf.contrib.layers.batch_norm(rs_input, is_training=self.phase, updates_collections=None, epsilon=1e-7) 
            self.conv1 = slim.conv2d(inputs=rs_input,num_outputs=32,kernel_size=[4,6],stride=[2,3],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[1]=="t" else None, activation_fn=tf.nn.relu)
            variable_summary(self.conv1, "conv1")
            self.conv2 = slim.conv2d(inputs=self.conv1,num_outputs=32,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[2]=="t" else None, activation_fn=tf.nn.relu)
            variable_summary(self.conv2, "conv2")
            self.conv3 = slim.conv2d(inputs=self.conv2,num_outputs=32,kernel_size=[3,3],stride=[2,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[3]=="t" else None, activation_fn=tf.nn.relu)
            if np.prod(np.array(self.conv3.get_shape()[1:])) != 2*2*32:
                self.conv3 = slim.conv2d(inputs=self.conv3,num_outputs=32,kernel_size=[3,3],stride=[4,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[3]=="t" else None, activation_fn=tf.nn.relu)
                
            self.conv3_flat = tf.reshape(self.conv3, [-1, 2*2*32])
            variable_summary(self.conv3_flat, "conv3_flat")
            if batchnorm[4]=="t":
                self.conv3_flat = tf.contrib.layers.batch_norm(self.conv3_flat, updates_collections=None, is_training=self.phase, epsilon=1e-7) #"in all layers prior to the action input" 
            self.fc1 = dense(self.conv3_flat, 200, tf.nn.relu, decay=decay)
            if batchnorm[5]=="t":
                self.fc1 = tf.contrib.layers.batch_norm(self.fc1, updates_collections=None, is_training=self.phase, epsilon=1e-7)
            variable_summary(self.fc1, "fc1")
            self.fc2 = dense(self.fc1, 200, tf.nn.relu, decay=decay)
            if batchnorm[6]=="t":
                self.fc2 = tf.contrib.layers.batch_norm(self.fc2, updates_collections=None, is_training=self.phase, epsilon=1e-7)
            variable_summary(self.fc2, "fc2")
            self.outs = dense(self.fc2, conf.num_actions, tf.nn.tanh, decay=decay, minmax = 3e-4)
            variable_summary(self.outs, "outs")
            self.outs = apply_constraints(self.conf, self.outs, self.stands_input)
            self.scaled_out = (((self.outs - tanh_min_bounds)/ (tanh_max_bounds - tanh_min_bounds)) * (max_bounds - min_bounds) + min_bounds) #broadcasts the bound arrays
            

        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outerscope+"/"+self.name)
        self.ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=outerscope+"/"+self.name)  
        self.summaryOps = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=outerscope+"/"+self.name))     


        
class lowdim_actorNet():
     def __init__(self, conf, agent, outerscope="actor", name="online", batchnorm = "ftt"):      
        tanh_min_bounds,tanh_max_bounds = np.array([-1]), np.array([1])
        min_bounds, max_bounds = np.array(list(zip(*conf.action_bounds))) 
        self.name = name
        self.conf = conf
        self.agent = agent
        decay = False #"For Q we included L2 weight decay", not for müh     
        ff_stacksize = self.conf.ff_stacksize if self.agent.ff_stacked else 1

        with tf.variable_scope(name):
            
            self.phase = tf.placeholder(tf.bool, name='phase') #for batchnorm, true heißt is_training
            self.ff_inputs =   tf.placeholder(tf.float32, shape=[None, ff_stacksize*self.agent.ff_inputsize], name="ff_inputs")  
            self.stands_input = tf.placeholder(tf.bool, name="stands_input") #necessary for settozero            
            if batchnorm[0]=="t":
                self.fc1 = dense(tf.contrib.layers.batch_norm(self.ff_inputs, updates_collections=None, is_training=self.phase), 400, tf.nn.relu, decay=decay) 
            else:
                self.fc1 = dense(self.ff_inputs, 400, tf.nn.relu, decay=decay) 
            if batchnorm[1]=="t":
                self.fc1 = tf.contrib.layers.batch_norm(self.fc1, updates_collections=None, is_training=self.phase, epsilon=1e-7)
            variable_summary(self.fc1, "fc1AfterBN")
            self.fc2 = dense(self.fc1, 300, tf.nn.relu, decay=decay)
            if batchnorm[2]=="t":
                self.fc2 = tf.contrib.layers.batch_norm(self.fc2, updates_collections=None, is_training=self.phase, epsilon=1e-7)
            variable_summary(self.fc2, "fc1AfterBN")
            self.outs = dense(self.fc2, conf.num_actions, tf.nn.tanh, decay=decay, minmax = 3e-4)
            variable_summary(self.outs, "outs")
            self.outs = apply_constraints(self.conf, self.outs, self.stands_input)
            self.scaled_out = (((self.outs - tanh_min_bounds)/ (tanh_max_bounds - tanh_min_bounds)) * (max_bounds - min_bounds) + min_bounds) #broadcasts the bound arrays
            
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outerscope+"/"+self.name)
        self.ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=outerscope+"/"+self.name)     
        self.summaryOps = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=outerscope+"/"+self.name)) 


#set brake-value to zero if car stands and care for not braking & accelerating simultaneously... however both apply ONLY in inference
def apply_constraints(conf, outs, stands_input):
    if not conf.INCLUDE_ACCPLUSBREAK: #if both throttle and brake are > 0.5, set brake to zero
        brakeallzero =  tf.stack([outs[:,0], -tf.random_uniform([tf.shape(outs)[0]], maxval=1), outs[:,2]],axis=1) if conf.brake_index == 1 \
                   else tf.stack([outs[:,0], outs[:,1], -tf.random_uniform([tf.shape(outs)[0]], maxval=1)],axis=1) if conf.brake_index == 2 \
                   else tf.stack([-tf.random_uniform([tf.shape(outs)[0]], maxval=1), outs[:,1], outs[:,2]],axis=1)
        throttallzero=  tf.stack([outs[:,0], -tf.random_uniform([tf.shape(outs)[0]], maxval=1), outs[:,2]],axis=1) if conf.throttle_index == 1 \
                   else tf.stack([outs[:,0], outs[:,1], -tf.random_uniform([tf.shape(outs)[0]], maxval=1)],axis=1) if conf.throttle_index == 2 \
                   else tf.stack([-tf.random_uniform([tf.shape(outs)[0]], maxval=1), outs[:,1], outs[:,2]],axis=1)                                     
        setone = tf.where(tf.random_normal([tf.shape(outs)[0]]) > 0, brakeallzero, throttallzero)
        applywhere = tf.logical_and(tf.cast((outs[:,conf.throttle_index] > 0.5), tf.bool), tf.cast((outs[:,conf.brake_index] > 0.5), tf.bool))
        outs = tf.cond(tf.equal(tf.shape(outs)[0], tf.constant(1)), lambda:tf.where(applywhere, setone, outs), lambda: outs)  
    if conf.use_settozero:
        brakeallzero =  tf.stack([outs[:,0], -tf.ones([tf.shape(outs)[0]]), outs[:,2]],axis=1) if conf.brake_index == 1 \
                   else tf.stack([outs[:,0], outs[:,1], -tf.ones([tf.shape(outs)[0]])],axis=1) if conf.brake_index == 2 \
                   else tf.stack([-tf.ones([tf.shape(outs)[0]]), outs[:,1], outs[:,2]],axis=1)
        outs = tf.cond(stands_input,lambda: brakeallzero, lambda: outs)
        throttlebig =  tf.stack([outs[:,0], tf.random_uniform([tf.shape(outs)[0]], maxval=1), outs[:,2]],axis=1) if conf.throttle_index == 1 \
                  else tf.stack([outs[:,0], outs[:,1], tf.random_uniform([tf.shape(outs)[0]], maxval=1)],axis=1) if conf.throttle_index == 2 \
                  else tf.stack([tf.random_uniform([tf.shape(outs)[0]], maxval=1), outs[:,1], outs[:,2]],axis=1)
        applywhere = tf.cast((outs[:,conf.throttle_index] < -0.8), tf.bool)
        outs = tf.cond(stands_input, lambda: tf.where(applywhere, throttlebig, outs), lambda: outs)
    return outs


##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
        
        
        
class conv_criticNet():
     def __init__(self, conf, agent, outerscope="critic", name="online", batchnorm="fffffff"): #tftftff   
        self.conf = conf
        self.agent = agent
        self.name = name  
        conv_stacksize = self.conf.conv_stacksize if self.agent.conv_stacked else 1        
        ff_stacksize = self.conf.ff_stacksize if self.agent.ff_stacked else 1
        decayrate = 1e-2        

        with tf.variable_scope(name):
            self.conv_inputs = tf.placeholder(tf.float32, shape=[None, self.conf.image_dims[0], self.conf.image_dims[1], conv_stacksize], name="conv_inputs")  
            self.ff_inputs =   tf.placeholder(tf.float32, shape=[None, ff_stacksize*self.agent.ff_inputsize], name="ff_inputs")  
            self.phase = tf.placeholder(tf.bool, name='phase') #for batchnorm, true heißt is_training
            self.actions = tf.placeholder(tf.float32, shape=[None, self.conf.num_actions], name="action_inputs")  
                        
            rs_input = tf.reshape(self.conv_inputs, [-1, self.conf.image_dims[0], self.conf.image_dims[1], conv_stacksize]) #final dimension = number of color channels*number of stacked (history-)frames                  

            if batchnorm[0]=="t":
                rs_input = tf.contrib.layers.batch_norm(rs_input, updates_collections=None, is_training=self.phase, epsilon=1e-7)  
            self.conv1 = slim.conv2d(inputs=rs_input,num_outputs=32,kernel_size=[4,6],stride=[2,3],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[1]=="t" else None, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(decayrate))
            variable_summary(self.conv1, "conv1")
            self.conv2 = slim.conv2d(inputs=self.conv1,num_outputs=32,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[2]=="t" else None, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(decayrate))
            variable_summary(self.conv2, "conv2")
            self.conv3 = slim.conv2d(inputs=self.conv2,num_outputs=32,kernel_size=[3,3],stride=[2,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[3]=="t" else None, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(decayrate))
            variable_summary(self.conv3, "conv3")
            if np.prod(np.array(self.conv3.get_shape()[1:])) != 2*2*32:
                self.conv3 = slim.conv2d(inputs=self.conv3,num_outputs=32,kernel_size=[3,3],stride=[4,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[3]=="t" else None, activation_fn=tf.nn.relu)            
            self.conv3_flat = tf.reshape(self.conv3, [-1, 2*2*32])
            if batchnorm[4]=="t":
                self.conv3_flat = tf.contrib.layers.batch_norm(self.conv3_flat, updates_collections=None, is_training=self.phase, epsilon=1e-7) #"in all layers prior to the action input"
            self.conv3_flat = tf.concat([self.conv3_flat, self.actions], 1) 
            variable_summary(self.conv3_flat, "conv3_flat")
            self.fc1 = dense(self.conv3_flat, 200, tf.nn.relu, decay=True)
            if batchnorm[5]=="t":
                self.fc1 = tf.contrib.layers.batch_norm(self.fc1, updates_collections=None, is_training=self.phase, epsilon=1e-7)
            variable_summary(self.fc1, "fc1")
            self.fc2 = dense(self.fc1, 200, tf.nn.relu, decay=True)
            if batchnorm[6]=="t":
                self.fc2 = tf.contrib.layers.batch_norm(self.fc2, updates_collections=None, is_training=self.phase, epsilon=1e-7)
            variable_summary(self.fc2, "fc2")
            self.Q = dense(self.fc2, 1, decay=True, minmax=3e-4)
            variable_summary(self.Q, "Q")
            
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outerscope+"/"+self.name)
        self.ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=outerscope+"/"+self.name)
        self.summaryOps = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=outerscope+"/"+self.name))
        
       
        
        
class lowdim_criticNet():
     def __init__(self, conf, agent, outerscope="critic", name="online", batchnorm="ffff"):       
        self.conf = conf
        self.agent = agent
        self.name = name   
        ff_stacksize = self.conf.ff_stacksize if self.agent.ff_stacked else 1

        with tf.variable_scope(name):
            
            self.ff_inputs =  tf.placeholder(tf.float32, shape=[None, ff_stacksize*self.agent.ff_inputsize], name="ff_inputs")  
            self.phase = tf.placeholder(tf.bool, name='phase') #for batchnorm, true heißt is_training
            self.actions = tf.placeholder(tf.float32, shape=[None, self.conf.num_actions], name="action_inputs")  
            variable_summary(self.ff_inputs, "inputs")
            if batchnorm[0]=="t":
                self.fc1 = dense(tf.contrib.layers.batch_norm(self.ff_inputs, updates_collections=None, is_training=self.phase, epsilon=1e-7), 400, tf.nn.relu, decay=True)
            else:
                self.fc1 = dense(self.ff_inputs, 400, tf.nn.relu, decay=True)
            if batchnorm[1]=="t":
                self.fc1 = tf.contrib.layers.batch_norm(self.fc1, updates_collections=None, is_training=self.phase, epsilon=1e-7)
            self.fc1 =  tf.concat([self.fc1, self.actions], 1)   
            variable_summary(self.fc1, "fc1AfterBNWithActions")
            self.fc2 = dense(self.fc1, 300, tf.nn.relu, decay=True)
            if batchnorm[2]=="t":
                self.fc2 = tf.contrib.layers.batch_norm(self.fc2, updates_collections=None, is_training=self.phase, epsilon=1e-7)
            variable_summary(self.fc2, "fc2AfterBN")
            self.fc3 = dense(self.fc2, 20, decay=True)
#            self.fc3 = tf.clip_by_value(self.fc3, -20, 20) #TODO sollte die performance irgendwann stagnieren, hieran liegts^^
            if batchnorm[3]=="t":
                self.fc3 = tf.contrib.layers.batch_norm(self.fc3, updates_collections=None, is_training=self.phase, epsilon=1e-7)
            variable_summary(self.fc3, "fc3AfterBN")
            self.Q = dense(self.fc3, 1, decay=True, minmax=3e-4)
            variable_summary(self.Q, "Q")
            
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outerscope+"/"+self.name)
        self.ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=outerscope+"/"+self.name)      
        self.summaryOps = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=outerscope+"/"+self.name))
       
                
        
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
        

        
class Actor(object):
    def __init__(self, conf, agent, session, batchnormstring="", isPretrain=False):
        self.conf = conf
        self.agent = agent
        self.session = session
        self.isPretrain = isPretrain
        self.stood_frames_ago = 0
        kwargs = {"batchnorm": batchnormstring} if len(batchnormstring) > 0 else {}
        
        with tf.variable_scope("actor"):
            with tf.variable_scope("target"): #damit der saver das mit saved            
                self.step_tf = tf.Variable(tf.constant(0), dtype=tf.int32, name='step_tf', trainable=False)
                self.run_inferences_tf = tf.Variable(tf.constant(0), dtype=tf.int32, name='run_inferences_tf', trainable=False)
                self.pretrain_episode_tf = tf.Variable(tf.constant(0), dtype=tf.int32, name='pretrain_episode_tf', trainable=False)
                self.pretrain_step_tf = tf.Variable(tf.constant(0), dtype=tf.int32, name='pretrain_step_tf', trainable=False)
                
            if self.agent.usesConv:
                self.online = conv_actorNet(conf, agent, **kwargs)
                self.target = conv_actorNet(conf, agent, name="target", **kwargs)
            else:
                self.online = lowdim_actorNet(conf, agent, **kwargs)
                self.target = lowdim_actorNet(conf, agent, name="target", **kwargs)
            self.smoothTargetUpdate = netCopyOps(self.online, self.target, self.conf.target_update_tau)
            # provided by the critic network
            self.action_gradient = tf.placeholder(tf.float32, [None, self.conf.num_actions], name="actiongradient")
            self.actor_gradients = tf.gradients(self.online.scaled_out, self.online.trainables, -self.action_gradient)
            if self.isPretrain:
                self.optimize = tf.train.AdamOptimizer(self.conf.actor_lr).apply_gradients(zip(self.actor_gradients, self.online.trainables), global_step=self.pretrain_step_tf)
            else:
                self.optimize = tf.train.AdamOptimizer(self.conf.actor_lr).apply_gradients(zip(self.actor_gradients, self.online.trainables), global_step=self.step_tf)
            
        self.saver = tf.train.Saver(var_list=get_variables("actor/target"))

    def train(self, inputs, a_gradient, doSummary=False):
        returns = [self.optimize]
        if doSummary:
            returns.append(self.online.summaryOps)
        else:
            returns.append(returns[0])
        return self.session.run(returns, feed_dict=self._make_inputs(inputs, self.online, {self.action_gradient: a_gradient}))

    def predict(self, inputs, useOnline=True, is_training=True, is_inference=False):
        carstands = inputs[0][2] if len(inputs) == 1 and len(inputs[0]) > 2 else False
        net = self.online if useOnline else self.target
        return self.session.run(net.scaled_out, feed_dict=self._make_inputs(inputs, net, is_training=is_training, carstands=carstands, is_inference=is_inference))
         

    def update_target_network(self):
        self.session.run(self.smoothTargetUpdate)
        
    def _make_inputs(self, inputs, net, others={}, is_training=True, carstands=False, is_inference=False):
        if not is_training and is_inference:   
            self.stood_frames_ago = 0 if carstands else self.stood_frames_ago + 1
            if self.stood_frames_ago < 4: #wenn du vor einigen frames stands, gib jetzt auch gas
                carstands = True
        conv_inputs = [inputs[i][0] for i in range(len(inputs))]
        ff_inputs = [inputs[i][1] for i in range(len(inputs))]
        others[net.stands_input] = carstands 
        feed_conv = {net.conv_inputs: conv_inputs} if self.agent.usesConv else {}
        feed_ff = {net.ff_inputs: ff_inputs} if self.agent.ff_inputsize > 0  else {}
        return {**feed_conv, **feed_ff, **others, net.phase: is_training}

        
        
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
        
        
        

class Critic(object):
    def __init__(self, conf, agent, session, batchnormstring="", isPretrain=False, name="critic"):
        self.conf = conf
        self.agent = agent
        self.session = session
        self.isPretrain = isPretrain
        kwargs = {"batchnorm": batchnormstring, "outerscope": name} if len(batchnormstring) > 0 else {"outerscope": name}

        with tf.variable_scope(name):
            
            with tf.variable_scope("target"): #damit der saver das mit saved
                self.step_tf = tf.Variable(tf.constant(0), dtype=tf.int32, name='step_tf', trainable=False)
                self.pretrain_step_tf = tf.Variable(tf.constant(0), dtype=tf.int32, name='pretrain_step_tf', trainable=False)
                
            if self.agent.usesConv:
                self.online = conv_criticNet(conf, agent, **kwargs)
                self.target = conv_criticNet(conf, agent, name="target", **kwargs)
            else:
                self.online = lowdim_criticNet(conf, agent, **kwargs)
                self.target = lowdim_criticNet(conf, agent, name="target", **kwargs)            
                
            self.smoothTargetUpdate = netCopyOps(self.online, self.target, self.conf.target_update_tau)
            self.target_Q = tf.placeholder(tf.float32, [None, 1], name="target_Q")
            self.loss = tf.losses.mean_squared_error(self.target_Q, self.online.Q)
            if self.isPretrain:
                self.optimize = tf.train.AdamOptimizer(self.conf.critic_lr).minimize(self.loss, global_step = self.pretrain_step_tf)
            else:
                self.optimize = tf.train.AdamOptimizer(self.conf.critic_lr).minimize(self.loss, global_step = self.step_tf)
            self.action_grads = tf.gradients(self.online.Q, self.online.actions)
            
        self.saver = tf.train.Saver(var_list=get_variables(name+"/target"))
        

    def train(self, inputs, action, target_Q, doSummary=False):
        returns = [self.online.Q, self.optimize, self.loss] 
        if doSummary:
            returns.append(self.online.summaryOps)
        else:
            returns.append(returns[0])
        return self.session.run(returns, feed_dict=self._make_inputs(inputs, self.online, {self.online.actions: action, self.target_Q: target_Q}))

    def predict(self, inputs, action, useOnline=True):
        net = self.online if useOnline else self.target
        return self.session.run(net.Q, feed_dict=self._make_inputs(inputs, net, {net.actions: action}))

    def action_gradients(self, inputs, actions):
        return self.session.run(self.action_grads, feed_dict=self._make_inputs(inputs, self.online, {self.online.actions: actions}))

    def update_target_network(self):
        self.session.run(self.smoothTargetUpdate)
        
    def _make_inputs(self, inputs, net, others={}, is_training=True):
        conv_inputs = [inputs[i][0] for i in range(len(inputs))]
        ff_inputs = [inputs[i][1] for i in range(len(inputs))]
        feed_conv = {net.conv_inputs: conv_inputs} if self.agent.usesConv else {}
        feed_ff = {net.ff_inputs: ff_inputs} if self.agent.ff_inputsize > 0  else {}
        return {**feed_conv, **feed_ff, **others, net.phase: is_training}
        


##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

        
        
class DDPG_model():
    
    def __init__(self,  conf, agent, session, isPretrain=False, actorbatchnorm="", criticbatchnorm=""):
        self.conf = conf
        self.agent = agent
        self.session = session
        self.isPretrain = isPretrain
        self.actor = Actor(self.conf, self.agent, self.session, actorbatchnorm, isPretrain)
        self.critic = Critic(self.conf, self.agent, self.session, criticbatchnorm, isPretrain) 
        self.run_inf = 0
        self.pretrain_ep = 0
        self.boardstep = 0
        if conf.summarize_tensorboard_allstep:
            self.writer = self._initTensorboard()
            
        
    def initNet(self, load=False):
        self.session.run(tf.global_variables_initializer())
        
        if load == "preTrain":
            self._load(from_pretrain=True)     
        elif load == "noPreTrain":
            self._load(from_pretrain=False)   
        elif load != False: #versuche RLLearn, wenn das nicht geht pretrain
            if not self._load(from_pretrain=False):
                self._load(from_pretrain=True)                
                
        self.session.run(netCopyOps(self.actor.target, self.actor.online))
        self.session.run(netCopyOps(self.critic.target, self.critic.online))   
        
    
    def save(self):
        folder = self.conf.pretrain_checkpoint_dir if self.isPretrain else self.conf.checkpoint_dir
        critic_file = os.path.join(self.agent.folder(os.path.join(folder,"critic")), 'model.ckpt')
        self.critic.saver.save(self.session, critic_file, global_step=self.critic.pretrain_step_tf if self.isPretrain else self.critic.step_tf)
        actor_file = os.path.join(self.agent.folder(os.path.join(folder,"actor")), 'model.ckpt')        
        self.session.run(self.actor.run_inferences_tf.assign(self.run_inf))
        self.session.run(self.actor.pretrain_episode_tf.assign(self.pretrain_ep))
        self.actor.saver.save(self.session, actor_file, global_step=self.actor.pretrain_step_tf if self.isPretrain else self.actor.step_tf)
        print("Saved Model.", level=6) 
        
    
    def _load(self, from_pretrain=False):
        folder = self.conf.pretrain_checkpoint_dir if from_pretrain else self.conf.checkpoint_dir
        critic_ckpt = tf.train.get_checkpoint_state(self.agent.folder(os.path.join(folder,"critic")))
        actor_ckpt = tf.train.get_checkpoint_state(self.agent.folder(os.path.join(folder,"actor")))
        if critic_ckpt and actor_ckpt and critic_ckpt.model_checkpoint_path and actor_ckpt.model_checkpoint_path:
            self.critic.saver.restore(self.session, critic_ckpt.model_checkpoint_path)
            self.actor.saver.restore(self.session, actor_ckpt.model_checkpoint_path)
            self.run_inf = self.actor.run_inferences_tf.eval(self.session)
            self.pretrain_ep = self.actor.pretrain_episode_tf.eval(self.session)
        else:
            print("Couldn't load", ("from pretrain" if from_pretrain else "from RL-train"), level=10)
            return False
        print("Loaded",("from pretrain" if from_pretrain else "from RL-train"), level=10)
        print("Pretrain-Step:",self.actor.pretrain_step_tf.eval(self.session), "Pretrain-Episode:",self.pretrain_ep,"Main-Step:",self.step(), "Run'n Iterations:", self.run_inf, level=10)
        return True
        
    
    def step(self): 
        return self.actor.step_tf.eval(self.session)
    def inc_episode(self): 
        if self.isPretrain:
            self.pretrain_ep += 1
    def pretrain_episode(self):
        return self.pretrain_ep
    def run_inferences(self):
        return self.run_inf
    
   
    #expects a whole s,a,r,s,t - tuple, needs however only s & a
    def getAccuracy(self, batch, likeDDPG=True): #dummy for consistency to DDDQN
        oldstates, actions, _, _, _ = batch
        predict = self.actor.predict(oldstates, useOnline=False, is_training=False)
        print("throt",np.mean(np.array([abs(np.linalg.norm(predict[i][0]-actions[i][0])) for i in range(len(actions))])))
        print("brake",np.mean(np.array([abs(np.linalg.norm(predict[i][1]-actions[i][1])) for i in range(len(actions))])))
        print("steer",np.mean(np.array([abs(np.linalg.norm(predict[i][2]-actions[i][2])) for i in range(len(actions))])))
        return np.mean(np.array([abs(np.linalg.norm(predict[i]-actions[i])) for i in range(len(actions))]))

    
    #expects only a state 
    def inference(self, oldstates):
        assert not self.isPretrain, "Please reload this network as a non-pretrain-one!"
        self.run_inf += 1
        action = self.actor.predict(oldstates, useOnline=False, is_training=False, is_inference=True)
        value =  self.critic.predict(oldstates, action, useOnline=False)
        return action, value
        
    
    #expects only a state 
    def statevalue(self, oldstates):                                                  
        action = self.actor.predict(oldstates, useOnline=False, is_training=False)  
        return self.critic.predict(oldstates, action, useOnline=False)[0]
    
    #expects state and action
    def qvalue(self, oldstates, actions):                                            
        return self.critic.predict(oldstates, actions, useOnline=False)[0]    
    
    #expects state and action
    def getstatecountfeaturevec(self, oldstates, actions):
        lastlay = np.array(self.session.run(self.critic.target.fc3, feed_dict=self.critic._make_inputs(oldstates, self.critic.target, {self.critic.target.actions: actions})))
        lastlay = np.round(np.concatenate([lastlay, np.array(actions)*20], axis=1))
        return lastlay
    
    
    #expects a whole s,a,r,s,t - tuple
    def q_train_step(self, batch, decay_lr=False): 
        assert not decay_lr, "This function is not implemented in the DDPG, as it doesnt make too much sense"
        self.boardstep += 1
        doSummary = self.boardstep % self.conf.summarize_tensorboard_allstep == 0 if self.conf.summarize_tensorboard_allstep else False
        oldstates, actions, rewards, newstates, terminals = batch
        #Training the critic...
#        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#        with tf.control_dependencies(update_ops): #because of batchnorm, see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
        act = self.actor.predict(newstates, useOnline=False)
        target_q = self.critic.predict(newstates, act, useOnline=False)
        cumrewards = np.reshape([rewards[i] if terminals[i] else rewards[i]+self.conf.q_decay*target_q[i] for i in range(len(rewards))], (len(rewards),1))
        target_Q, _, loss, sumCritic = self.critic.train(oldstates, actions, cumrewards, doSummary=doSummary)
        #training the actor...        
        a_outs = self.actor.predict(oldstates)
        grads = self.critic.action_gradients(oldstates, a_outs)
        _, sumActor = self.actor.train(oldstates, grads[0], doSummary=doSummary)
        #updating the targetnets...
        self.actor.update_target_network()
        self.critic.update_target_network()
        if doSummary:
            self.writer.add_summary(sumCritic, self.critic.step_tf.eval(self.session))
            self.writer.add_summary(sumActor, self.actor.step_tf.eval(self.session))
        return np.max(target_Q)
        
               
    def _initTensorboard(self):
        folder = self.agent.folder(self.conf.pretrain_log_dir) if self.isPretrain else self.agent.folder(self.conf.log_dir)
        writer = tf.summary.FileWriter(folder,self.session.graph)
        return writer
