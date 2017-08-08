""" 
Implementation of DDPG - Deep Deterministic Policy Gradient
Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf
The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow
Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import time
import tensorflow.contrib.slim as slim
from myprint import printtofile as print



def _netCopyOps(fromNet, toNet, tau = 1):
    toCopy = fromNet.trainables
    toPast = toNet.trainables
    op_holder = []
    for idx,var in enumerate(toCopy[:]):
        if tau == 1:
            op_holder.append(toPast[idx].assign(var.value()))
        else:
            op_holder.append(toPast[idx].assign((var.value()*tau) + ((1-tau)*toPast[idx].value())))
    return op_holder
       
        
def dense(x, units, activation=tf.identity, decay=None, minmax=None):
    if minmax is None:
        minmax = float(x.shape[1].value) ** -.5
    return tf.layers.dense(x, units,activation=activation, kernel_initializer=tf.random_uniform_initializer(-minmax, minmax), kernel_regularizer=decay and tf.contrib.layers.l2_regularizer(1e-2))

        
        
class actorNet():
     def __init__(self, normstring, conf, agent, outerscope="actor", name="online"):       
        tanh_min_bounds,tanh_max_bounds = np.array([-1]), np.array([1])
        min_bounds, max_bounds = np.array(list(zip(*conf.action_bounds))) 
        self.name = name
        self.conf = conf
        self.agent = agent
        h_size = 100
        batchnorm = normstring
        decay = True #"For Q we included L2 weight decay", not for müh
        conv_stacksize = (self.conf.history_frame_nr*2 if self.conf.use_second_camera else self.conf.history_frame_nr) if self.agent.conv_stacked else 1        
        

        with tf.variable_scope(name):
            
            self.conv_inputs = tf.placeholder(tf.float32, shape=[None, self.conf.image_dims[0], self.conf.image_dims[1], conv_stacksize], name="conv_inputs")  

            rs_input = tf.reshape(self.conv_inputs, [-1, self.conf.image_dims[0], self.conf.image_dims[1], conv_stacksize]) #final dimension = number of color channels*number of stacked (history-)frames                  
            #convolutional_layer(input_tensor, input_channels, kernel_size, stride, output_channels, name, act, is_trainable, batchnorm, is_training, weightdecay=False, pool=True, trainvars=None, varSum=None, initializer=None)
            if batchnorm[0]=="t":
                rs_input = tf.layers.batch_normalization(rs_input, training=True, epsilon=1e-7, momentum=.95) 
            self.conv1 = slim.conv2d(inputs=rs_input,num_outputs=32,kernel_size=[4,6],stride=[2,3],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[1]=="t" else None, activation_fn=tf.nn.relu)
            self.conv2 = slim.conv2d(inputs=self.conv1,num_outputs=32,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[2]=="t" else None, activation_fn=tf.nn.relu)
            self.conv3 = slim.conv2d(inputs=self.conv2,num_outputs=32,kernel_size=[3,3],stride=[2,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[3]=="t" else None, activation_fn=tf.nn.relu)
            self.conv3_flat = tf.reshape(self.conv3, [-1, 2*2*32])
            if batchnorm[4]=="t":
                self.conv3_flat = tf.layers.batch_normalization(self.conv3_flat, training=True, epsilon=1e-7, momentum=.95) #"in all layers prior to the action input" 
            self.fc1 = dense(self.conv3_flat, 200, tf.nn.relu, decay=decay)
            if batchnorm[5]=="t":
                self.fc1 = tf.layers.batch_normalization(self.fc1, training=True, epsilon=1e-7, momentum=.95)
            self.fc2 = dense(self.fc1, 200, tf.nn.relu, decay=decay)
            if batchnorm[6]=="t":
                self.fc2 = tf.layers.batch_normalization(self.fc2, training=True, epsilon=1e-7, momentum=.95)
            self.fc3 = dense(self.fc2, conf.num_actions, tf.nn.tanh, decay=decay, minmax = 3e-4)
            self.outs = self.fc3
            self.scaled_out = self.outs# (((self.outs - tanh_min_bounds)/ (tanh_max_bounds - tanh_min_bounds)) * (max_bounds - min_bounds) + min_bounds) #broadcasts the bound arrays
            

        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outerscope+"/"+self.name)
        self.ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=outerscope+"/"+self.name)      
        
        
        
        
        
class Actor(object):
    def __init__(self, normstring, conf, agent, session):
        self.conf = conf
        self.agent = agent
        self.session = session
        
        with tf.variable_scope("actor"):
            
            self.online = actorNet(normstring, conf, agent)
            self.target = actorNet(normstring, conf, agent, name="target")
        
            self.smoothTargetUpdate = _netCopyOps(self.online, self.target, self.conf.target_update_tau)
    
            # This gradient will be provided by the critic network
            self.action_gradient = tf.placeholder(tf.float32, [None, self.conf.num_actions], name="actiongradient")
    
            # Combine the gradients here
            self.actor_gradients = tf.gradients(self.online.scaled_out, self.online.trainables, -self.action_gradient)
    
            # Optimization Op
            self.optimize = tf.train.AdamOptimizer(self.conf.actor_lr).apply_gradients(zip(self.actor_gradients, self.online.trainables))
    


    def train(self, inputs, a_gradient):
        self.session.run(self.optimize, feed_dict={self.online.conv_inputs: self.make_inputs(inputs),self.action_gradient: a_gradient})

    def predict(self, inputs, which="online"):
        net = self.online if which == "online" else self.target
        return self.session.run(net.scaled_out, feed_dict={net.conv_inputs: self.make_inputs(inputs)})

    def update_target_network(self):
        self.session.run(self.smoothTargetUpdate)
        
    def make_inputs(self, inputs):
        conv_inputs = [inputs[i][0] for i in range(len(inputs))]
        return conv_inputs        

        
        
        
        
        
class criticNet():
     def __init__(self, normstring, conf, agent, outerscope="critic", name="online"):       
        self.conf = conf
        self.agent = agent
        self.name = name  
        h_size = 100
        conv_stacksize = (self.conf.history_frame_nr*2 if self.conf.use_second_camera else self.conf.history_frame_nr) if self.agent.conv_stacked else 1
        decayrate = 1e-2        
        batchnorm = normstring

        with tf.variable_scope(name):
            
            self.conv_inputs = tf.placeholder(tf.float32, shape=[None, self.conf.image_dims[0], self.conf.image_dims[1], conv_stacksize], name="conv_inputs")  
            self.actions = tf.placeholder(tf.float32, shape=[None, self.conf.num_actions], name="action_inputs")  
                        
            rs_input = tf.reshape(self.conv_inputs, [-1, self.conf.image_dims[0], self.conf.image_dims[1], conv_stacksize]) #final dimension = number of color channels*number of stacked (history-)frames                  

            if batchnorm[0]=="t":
                rs_input = tf.layers.batch_normalization(rs_input, training=True, epsilon=1e-7, momentum=.95)  
            self.conv1 = slim.conv2d(inputs=rs_input,num_outputs=32,kernel_size=[4,6],stride=[2,3],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[1]=="t" else None, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(decayrate))
            self.conv2 = slim.conv2d(inputs=self.conv1,num_outputs=32,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[2]=="t" else None, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(decayrate))
            self.conv3 = slim.conv2d(inputs=self.conv2,num_outputs=32,kernel_size=[3,3],stride=[2,2],padding='VALID', biases_initializer=None, normalizer_fn=slim.batch_norm if batchnorm[3]=="t" else None, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(decayrate))
            self.conv3_flat = tf.reshape(self.conv3, [-1, 2*2*32])
            if batchnorm[4]=="t":
                self.conv3_flat = tf.layers.batch_normalization(self.conv3_flat, training=True, epsilon=1e-7, momentum=.95) #"in all layers prior to the action input"
            self.conv3_flat = tf.concat([self.conv3_flat, self.actions], 1) 
            self.fc1 = dense(self.conv3_flat, 200, tf.nn.relu, decay=True)
            if batchnorm[5]=="t":
                self.fc1 = tf.layers.batch_normalization(self.fc1, training=True, epsilon=1e-7, momentum=.95)
            self.fc2 = dense(self.fc1, 200, tf.nn.relu, decay=True)
            if batchnorm[6]=="t":
                self.fc2 = tf.layers.batch_normalization(self.fc2, training=True, epsilon=1e-7, momentum=.95)
            self.Q = dense(self.fc2, conf.num_actions, decay=True, minmax=3e-4)
            
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outerscope+"/"+self.name)
        self.ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=outerscope+"/"+self.name)      
        
       
        
        
        
        

class Critic(object):
    def __init__(self, normstring, conf, agent, session):
        self.conf = conf
        self.agent = agent
        self.session = session
        
        with tf.variable_scope("critic"):
            
            self.online = criticNet(normstring, conf, agent)
            self.target = criticNet(normstring, conf, agent, name="target")
                
            self.smoothTargetUpdate = _netCopyOps(self.online, self.target, self.conf.target_update_tau)
    
            self.target_Q = tf.placeholder(tf.float32, [None, 1], name="target_Q")
    
            self.loss = tf.losses.mean_squared_error(self.target_Q, self.online.Q)
            self.optimize = tf.train.AdamOptimizer(self.conf.critic_lr).minimize(self.loss)
    
            self.action_grads = tf.gradients(self.online.Q, self.online.actions)


    def train(self, inputs, action, target_Q):
        return self.session.run([self.online.Q, self.optimize, self.loss], feed_dict={self.online.conv_inputs: self.make_inputs(inputs), self.online.actions: action,self.target_Q: target_Q})

    def predict(self, inputs, action, which="online"):
        net = self.online if which == "online" else self.target
        return self.session.run(net.Q, feed_dict={net.conv_inputs: self.make_inputs(inputs), net.actions: action})

    def action_gradients(self, inputs, actions):
        return self.session.run(self.action_grads, feed_dict={self.online.conv_inputs: self.make_inputs(inputs), self.online.actions: actions})

    def update_target_network(self):
        self.session.run(self.smoothTargetUpdate)
        
    def make_inputs(self, inputs):
        conv_inputs = [inputs[i][0] for i in range(len(inputs))]
        return conv_inputs
        
    
        
class DDPG_model():
    
    def __init__(self, actornormstring, criticnormstring,  conf, agent, session):
        self.conf = conf
        self.agent = agent
        self.session = session
        self.initNet(actornormstring, criticnormstring)
        
        
    def initNet(self, actornormstring, criticnormstring):
        
        self.actor = Actor(actornormstring, self.conf, self.agent, self.session)
        self.critic = Critic(criticnormstring, self.conf, self.agent, self.session)     
        
        self.session.run(tf.global_variables_initializer())
        self.session.run(_netCopyOps(self.actor.target, self.actor.online))
        self.session.run(_netCopyOps(self.critic.target, self.critic.online))      
        
    
        
    #actor predicts action. critic predicts q-value of action. That is compared to the actual q-value of action. (TD-Error)
    #online-actor predicts new actions. actor uses critic's gradients to train, too.
    def train_step(self, batch):
        oldstates, actions, rewards, newstates, terminals = batch
        #Training the critic...
        target_q = self.critic.predict(newstates, self.actor.predict(newstates, "target"), "target")
        cumrewards = np.reshape([rewards[i] if terminals[i] else rewards[i]+0.99*target_q[i] for i in range(len(rewards))], (len(rewards),1))
        target_Q, _, loss = self.critic.train(oldstates, actions, cumrewards)
        #training the actor...        
        a_outs = self.actor.predict(oldstates)
        grads = self.critic.action_gradients(oldstates, a_outs)
        self.actor.train(oldstates, grads[0])
        #updating the targetnets...
        self.actor.update_target_network()
        self.critic.update_target_network()
        return np.max(target_Q)
        
    def inference(self, oldstates):
        return self.actor.predict(oldstates, "online") #ist halt schneller wenn online.

    def evaluate(self, batch):
        oldstates, actions, rewards, newstates, terminals = batch
#        result = np.array([np.argmax(self.agent.discretize(*i)) for i in self.actor.predict(oldstates)])
#        human = np.array([np.argmax(myAgent.discretize(*i)) for i in actions])
        actorOuts = self.inference(oldstates)
        return np.mean(np.array([abs(actorOuts[i][0] -actions[i][0]) for i in range(len(actions))]))

        


def TPSample(conf, agent, batchsize, trackingpoints):
    import read_supervised
    tmp = list(read_supervised.create_QLearnInputs_from_PTStateBatch(*trackingpoints.next_batch(conf, agent, batchsize), agent))
    tmp[1] = [[i[2]] for i in tmp[1]]
    return tmp         
        
    
        


def main():
    import config
    conf = config.Config()
    conf.num_actions = 1    
    conf.action_bounds = [(-1, 1)]
    import read_supervised
    from server import Containers
    import dqn_rl_agent
    myAgent = dqn_rl_agent.Agent(conf, Containers(), True)
    
    trackingpoints = read_supervised.TPList(conf.LapFolderName, conf.use_second_camera, conf.msperframe, conf.steering_steps, conf.INCLUDE_ACCPLUSBREAK)
    BATCHSIZE = 16
    
    for i in range(64,128):
        
        actornorm = str(bin(i))[2:].replace("0","f").replace("1","t")
    
        for j in range(64,128):
            
            criticnorm = str(bin(j))[2:].replace("0","f").replace("1","t")
        
            tf.reset_default_graph()
        
            model = DDPG_model(actornorm, criticnorm, conf, myAgent, tf.Session())
        
            print("ACTORNORM", actornorm, "  CRITICNORM", criticnorm)
            
            for i in range(51):
                trackingpoints.reset_batch()
                trainBatch = TPSample(conf, myAgent, trackingpoints.numsamples, trackingpoints)
                print("Iteration", i, "Accuracy (0 is best)",model.evaluate(trainBatch))  
                if i % 5 == 0:
                    print(np.array(model.inference(trainBatch[0][:6]))) #die ersten 2 states   
                    print(np.array(trainBatch[1][:6]))
                trackingpoints.reset_batch()     
                while trackingpoints.has_next(BATCHSIZE):
                    trainBatch = TPSample(conf, myAgent, BATCHSIZE, trackingpoints)
                    model.train_step(trainBatch)    
                
            
            
            
    time.sleep(99999)    

    
    
if __name__ == '__main__':
    main()
            