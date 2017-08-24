import tensorflow as tf
import numpy as np
import time
import tensorflow.contrib.slim as slim
from myprint import myprint as print
from utils import netCopyOps
import os
from tensorflow.contrib.framework import get_variables

#batchnorm doesnt really work, and if, only with huge minibatches https://www.reddit.com/r/MachineLearning/comments/671455/d_batch_normalization_in_reinforcement_learning/

       
        
def dense(x, units, activation=tf.identity, decay=None, minmax=None):
    if minmax is None:
        minmax = float(x.shape[1].value) ** -.5
    return tf.layers.dense(x, units,activation=activation, kernel_initializer=tf.random_uniform_initializer(-minmax, minmax), kernel_regularizer=decay and tf.contrib.layers.l2_regularizer(1e-2))

    
        
class conv_actorNet():
     def __init__(self, conf, agent, outerscope="actor", name="online", batchnorm = "fffffff"):   #tffffft    
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
            self.phase = tf.placeholder(tf.bool, name='phase') #for batchnorm, true heißt is_training

            rs_input = tf.reshape(self.conv_inputs, [-1, self.conf.image_dims[0], self.conf.image_dims[1], conv_stacksize]) #final dimension = number of color channels*number of stacked (history-)frames                  
            if batchnorm[0]=="t":
                rs_input = tf.contrib.layers.batch_norm(rs_input, is_training=self.phase, updates_collections=None, epsilon=1e-7) 
            self.conv1 = slim.conv2d(inputs=rs_input,num_outputs=32,kernel_size=[4,6],stride=[2,3],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[1]=="t" else None, activation_fn=tf.nn.relu)
            self.conv2 = slim.conv2d(inputs=self.conv1,num_outputs=32,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[2]=="t" else None, activation_fn=tf.nn.relu)
            self.conv3 = slim.conv2d(inputs=self.conv2,num_outputs=32,kernel_size=[3,3],stride=[2,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[3]=="t" else None, activation_fn=tf.nn.relu)
            if np.prod(np.array(self.conv3.get_shape()[1:])) != 2*2*32:
                self.conv3 = slim.conv2d(inputs=self.conv3,num_outputs=32,kernel_size=[3,3],stride=[4,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[3]=="t" else None, activation_fn=tf.nn.relu)
                
            self.conv3_flat = tf.reshape(self.conv3, [-1, 2*2*32])
            if batchnorm[4]=="t":
                self.conv3_flat = tf.contrib.layers.batch_norm(self.conv3_flat, updates_collections=None, is_training=self.phase, epsilon=1e-7) #"in all layers prior to the action input" 
            self.fc1 = dense(self.conv3_flat, 200, tf.nn.relu, decay=decay)
            if batchnorm[5]=="t":
                self.fc1 = tf.contrib.layers.batch_norm(self.fc1, updates_collections=None, is_training=self.phase, epsilon=1e-7)
            self.fc2 = dense(self.fc1, 200, tf.nn.relu, decay=decay)
            if batchnorm[6]=="t":
                self.fc2 = tf.contrib.layers.batch_norm(self.fc2, updates_collections=None, is_training=self.phase, epsilon=1e-7)
            self.outs = dense(self.fc2, conf.num_actions, tf.nn.tanh, decay=decay, minmax = 3e-4)
            self.scaled_out = (((self.outs - tanh_min_bounds)/ (tanh_max_bounds - tanh_min_bounds)) * (max_bounds - min_bounds) + min_bounds) #broadcasts the bound arrays
            

        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outerscope+"/"+self.name)
        self.ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=outerscope+"/"+self.name)      
        
        

        
class lowdim_actorNet():
     def __init__(self, conf, agent, outerscope="actor", name="online", batchnorm = "fff"):       
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
            if batchnorm[0]=="t":
                self.ff_inputs = tf.contrib.layers.batch_norm(self.ff_inputs, updates_collections=None, is_training=self.phase)
            self.fc1 = dense(self.ff_inputs, 400, tf.nn.relu, decay=decay)
            if batchnorm[1]=="t":
                self.fc1 = tf.contrib.layers.batch_norm(self.fc1, updates_collections=None, is_training=self.phase, epsilon=1e-7)
            self.fc2 = dense(self.fc1, 300, tf.nn.relu, decay=decay)
            if batchnorm[2]=="t":
                self.fc2 = tf.contrib.layers.batch_norm(self.fc2, updates_collections=None, is_training=self.phase, epsilon=1e-7)
            self.outs = dense(self.fc2, conf.num_actions, tf.nn.tanh, decay=decay, minmax = 3e-4)
            self.scaled_out = (((self.outs - tanh_min_bounds)/ (tanh_max_bounds - tanh_min_bounds)) * (max_bounds - min_bounds) + min_bounds) #broadcasts the bound arrays
            
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outerscope+"/"+self.name)
        self.ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=outerscope+"/"+self.name)      


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
            self.conv2 = slim.conv2d(inputs=self.conv1,num_outputs=32,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[2]=="t" else None, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(decayrate))
            self.conv3 = slim.conv2d(inputs=self.conv2,num_outputs=32,kernel_size=[3,3],stride=[2,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[3]=="t" else None, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(decayrate))
            if np.prod(np.array(self.conv3.get_shape()[1:])) != 2*2*32:
                self.conv3 = slim.conv2d(inputs=self.conv3,num_outputs=32,kernel_size=[3,3],stride=[4,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[3]=="t" else None, activation_fn=tf.nn.relu)            
            self.conv3_flat = tf.reshape(self.conv3, [-1, 2*2*32])
            if batchnorm[4]=="t":
                self.conv3_flat = tf.contrib.layers.batch_norm(self.conv3_flat, updates_collections=None, is_training=self.phase, epsilon=1e-7) #"in all layers prior to the action input"
            self.conv3_flat = tf.concat([self.conv3_flat, self.actions], 1) 
            self.fc1 = dense(self.conv3_flat, 200, tf.nn.relu, decay=True)
            if batchnorm[5]=="t":
                self.fc1 = tf.contrib.layers.batch_norm(self.fc1, updates_collections=None, is_training=self.phase, epsilon=1e-7)
            self.fc2 = dense(self.fc1, 200, tf.nn.relu, decay=True)
            if batchnorm[6]=="t":
                self.fc2 = tf.contrib.layers.batch_norm(self.fc2, updates_collections=None, is_training=self.phase, epsilon=1e-7)
            self.Q = dense(self.fc2, 1, decay=True, minmax=3e-4)
            
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outerscope+"/"+self.name)
        self.ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=outerscope+"/"+self.name)
        
       
        
        
class lowdim_criticNet():
     def __init__(self, conf, agent, outerscope="critic", name="online", batchnorm="fff"):       
        self.conf = conf
        self.agent = agent
        self.name = name   
        ff_stacksize = self.conf.ff_stacksize if self.agent.ff_stacked else 1

        with tf.variable_scope(name):
            
            self.ff_inputs =   tf.placeholder(tf.float32, shape=[None, ff_stacksize*self.agent.ff_inputsize], name="ff_inputs")  
            self.phase = tf.placeholder(tf.bool, name='phase') #for batchnorm, true heißt is_training
            self.actions = tf.placeholder(tf.float32, shape=[None, self.conf.num_actions], name="action_inputs")  
            if batchnorm[0]=="t":
                self.ff_inputs = tf.contrib.layers.batch_norm(self.ff_inputs, updates_collections=None, is_training=self.phase, epsilon=1e-7)
            self.fc1 = dense(self.ff_inputs, 400, tf.nn.relu, decay=True)
            if batchnorm[1]=="t":
                self.fc1 = tf.contrib.layers.batch_norm(self.fc1, updates_collections=None, is_training=self.phase, epsilon=1e-7)
            self.fc1 =  tf.concat([self.fc1, self.actions], 1)   
            self.fc2 = dense(self.fc1, 300, tf.nn.relu, decay=True)
            if batchnorm[2]=="t":
                self.fc2 = tf.contrib.layers.batch_norm(self.fc2, updates_collections=None, is_training=self.phase, epsilon=1e-7)
            self.Q = dense(self.fc2, 1, decay=True, minmax=3e-4)
            
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outerscope+"/"+self.name)
        self.ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=outerscope+"/"+self.name)      
       
                
        
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
        

        
class Actor(object):
    def __init__(self, conf, agent, session, batchnormstring="", isPretrain=False):
        self.conf = conf
        self.agent = agent
        self.session = session
        self.isPretrain = isPretrain
        kwargs = {"batchnorm": batchnormstring} if len(batchnormstring) > 0 else {}
        
        with tf.variable_scope("actor"):
            with tf.variable_scope("target"): #damit der saver das mit saved            
                self.step_tf = tf.Variable(tf.constant(0), dtype=tf.int32, name='step_tf', trainable=False)
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
            self.optimize = tf.train.AdamOptimizer(self.conf.actor_lr).apply_gradients(zip(self.actor_gradients, self.online.trainables), global_step=self.step_tf)
            
        self.saver = tf.train.Saver(var_list=get_variables("actor/target"))

    def train(self, inputs, a_gradient):
        self.session.run(self.optimize, feed_dict=self._make_inputs(inputs, self.online, {self.action_gradient: a_gradient}))

    def predict(self, inputs, useOnline=True, is_training=True):
        net = self.online if useOnline else self.target
        return self.session.run(net.scaled_out, feed_dict=self._make_inputs(inputs, net, is_training=is_training))

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
        
        
        

class Critic(object):
    def __init__(self, conf, agent, session, batchnormstring="", isPretrain=False):
        self.conf = conf
        self.agent = agent
        self.session = session
        self.isPretrain = isPretrain
        kwargs = {"batchnorm": batchnormstring} if len(batchnormstring) > 0 else {}

        with tf.variable_scope("critic"):
            
            with tf.variable_scope("target"): #damit der saver das mit saved
                self.step_tf = tf.Variable(tf.constant(0), dtype=tf.int32, name='step_tf', trainable=False)
            
            if self.agent.usesConv:
                self.online = conv_criticNet(conf, agent, **kwargs)
                self.target = conv_criticNet(conf, agent, name="target", **kwargs)
            else:
                self.online = lowdim_criticNet(conf, agent, **kwargs)
                self.target = lowdim_criticNet(conf, agent, name="target", **kwargs)            
                
            self.smoothTargetUpdate = netCopyOps(self.online, self.target, self.conf.target_update_tau)
            self.target_Q = tf.placeholder(tf.float32, [None, 1], name="target_Q")
            self.loss = tf.losses.mean_squared_error(self.target_Q, self.online.Q)
            self.optimize = tf.train.AdamOptimizer(self.conf.critic_lr).minimize(self.loss, global_step = self.step_tf)
            self.action_grads = tf.gradients(self.online.Q, self.online.actions)
            
        self.saver = tf.train.Saver(var_list=get_variables("critic/target"))


    def train(self, inputs, action, target_Q):
        return self.session.run([self.online.Q, self.optimize, self.loss], feed_dict=self._make_inputs(inputs, self.online, {self.online.actions: action, self.target_Q: target_Q}))

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
        self.critic.saver.save(self.session, critic_file, global_step=self.critic.step_tf)
        actor_file = os.path.join(self.agent.folder(os.path.join(folder,"actor")), 'model.ckpt')
        self.actor.saver.save(self.session, actor_file, global_step=self.actor.step_tf)
        print("Saved Model.", level=6) 
        
    
    def _load(self, from_pretrain=False):
        folder = self.conf.pretrain_checkpoint_dir if from_pretrain else self.conf.checkpoint_dir
        critic_ckpt = tf.train.get_checkpoint_state(self.agent.folder(os.path.join(folder,"critic")))
        actor_ckpt = tf.train.get_checkpoint_state(self.agent.folder(os.path.join(folder,"actor")))
        if critic_ckpt and actor_ckpt and critic_ckpt.model_checkpoint_path and actor_ckpt.model_checkpoint_path:
            self.critic.saver.restore(self.session, critic_ckpt.model_checkpoint_path)
            self.actor.saver.restore(self.session, actor_ckpt.model_checkpoint_path)
        else:
            print("Couldn't load", ("from pretrain" if from_pretrain else "from RL-train"), level=10)
            return False
        print("Loaded",("from pretrain" if from_pretrain else "from RL-train"), level=10)
        print("Step:",self.actor.step_tf.eval(self.session))
        return True
        
    def step(self): #TODO DO
        return 0
    def inc_episode(self): #TODO DO
        return 0
    def pretrain_episode(self):
        return 0
    def run_inferences(self):
        return 0
    
   
    #expects a whole s,a,r,s,t - tuple, needs however only s & a
    def getAccuracy(self, batch, likeDDPG=True): #dummy for consistency to DDDQN
        oldstates, actions, _, _, _ = batch
        predict = self.actor.predict(oldstates, useOnline=False, is_training=False)
        return np.mean(np.array([abs(predict[i][0] -actions[i][0]) for i in range(len(actions))]))

    
    #expects only a state 
    def inference(self, oldstates):
        assert not self.isPretrain, "Please reload this network as a non-pretrain-one!"
        action = self.actor.predict(oldstates, useOnline=False, is_training=False)
        value =  self.critic.predict(oldstates, action, useOnline=False)
        return action, value
        
    
    #expects only a state 
    def statevalue(self, oldstates):                                                  
        action = self.actor.predict(oldstates, useOnline=False, is_training=False)  
        return self.critic.predict(oldstates, action, useOnline=False)
    
    
    
    #expects a whole s,a,r,s,t - tuple
    def q_train_step(self, batch, decay_lr = False): #TODO DO
        oldstates, actions, rewards, newstates, terminals = batch
        #Training the critic...
#        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#        with tf.control_dependencies(update_ops): #because of batchnorm, see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
        act = self.actor.predict(newstates, useOnline=False)
        target_q = self.critic.predict(newstates, act, useOnline=False)
        cumrewards = np.reshape([rewards[i] if terminals[i] else rewards[i]+self.conf.q_decay*target_q[i] for i in range(len(rewards))], (len(rewards),1))
        target_Q, _, loss = self.critic.train(oldstates, actions, cumrewards)
        #training the actor...        
        a_outs = self.actor.predict(oldstates)
        grads = self.critic.action_gradients(oldstates, a_outs)
        self.actor.train(oldstates, grads[0])
        #updating the targetnets...
        self.actor.update_target_network()
        self.critic.update_target_network()
        return np.max(target_Q)
        
               
        


        
    
    
    


#def TPSample(conf, agent, batchsize, trackingpoints):
#    import read_supervised
#    tmp = list(read_supervised.create_QLearnInputs_from_PTStateBatch(*trackingpoints.next_batch(conf, agent, batchsize), agent))
#    tmp[1] = [[i[2]] for i in tmp[1]]
#    return tmp         
#        
#    
#    
#def learn(conf, myAgent, batchsize, trackingpoints, iterations, actornorm="", criticnorm=""):
#    tf.reset_default_graph()
#    
#    model = DDPG_model(conf, myAgent, tf.Session(), actornorm, criticnorm)
#
#    print("ACTORNORM", actornorm, "  CRITICNORM", criticnorm)
#    
#    for i in range(iterations):
#        trackingpoints.reset_batch()
#        trainBatch = TPSample(conf, myAgent, trackingpoints.numsamples, trackingpoints)
#        print("Iteration", i, "Accuracy (0 is best)",model.evaluate(trainBatch))  
#        if i % 5 == 0:
#            print(np.array(model.inference(trainBatch[0][:6]))) #die ersten 2 states   
#            print(np.array(trainBatch[1][:6]))
#        trackingpoints.reset_batch()     
#        while trackingpoints.has_next(batchsize):
#            trainBatch = TPSample(conf, myAgent, batchsize, trackingpoints)
#            model.q_train_step(trainBatch)    
#                
#        
#
#
#def main():
#    import config
#    conf = config.Config()
#    conf.num_actions = 1    
#    conf.action_bounds = [(-1, 1)]
#    import read_supervised
#    from server import Containers
#    import ddpg_rl_agent
#    myAgent = ddpg_rl_agent.Agent(conf, Containers(), True)
#    
#    trackingpoints = read_supervised.TPList(conf.LapFolderName, conf.use_second_camera, conf.msperframe, conf.steering_steps, conf.INCLUDE_ACCPLUSBREAK)
#    BATCHSIZE = 128
#    
#    conf.actor_lr = 0.000001
#    conf.critic_lr = 0.0001
#    
##    for i in range(64,128):     
##        actornorm = str(bin(i))[2:].replace("0","f").replace("1","t") 
##        for j in range(64,128):         
##            criticnorm = str(bin(j))[2:].replace("0","f").replace("1","t")
#    
#    learn("tffffft", "tftftff", conf, myAgent, BATCHSIZE, trackingpoints, 200)    
#
#            
#            
#            
#    time.sleep(99999)    
#
#    
#    
#if __name__ == '__main__':
#    main()
#            