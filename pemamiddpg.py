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
import gym
from gym import wrappers
import tflearn

from replay_buffer import ReplayBuffer

# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = True
# Use Gym Monitor
GYM_MONITOR_EN = False
# Gym environment
ENV_NAME = 'Pendulum-v0'
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64

# ===========================
#   Actor and Critic DNNs
# ===========================

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
       
        
        
        
class actorNet():
     def __init__(self, conf, agent, outerscope="actor", name="online"):       
        tanh_min_bounds,tanh_max_bounds = np.array([-1]), np.array([1])
        min_bounds, max_bounds = np.array(list(zip(*conf.action_bounds))) 
        self.name = name
        self.conf = conf
        self.agent = agent
        num_actions = conf.num_actions
        conv_stacksize = (self.conf.history_frame_nr*2 if self.conf.use_second_camera else self.conf.history_frame_nr) if self.agent.conv_stacked else 1        
        
        s_dim = 3 #DELETEME

        with tf.variable_scope(name):
            self.inputs = tflearn.input_data(shape=[None, s_dim])
            net = tflearn.fully_connected(self.inputs, 400, activation='relu')
            net = tflearn.fully_connected(net, 300, activation='relu')
            # Final layer weights are init to Uniform[-3e-3, 3e-3]
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            self.outs = tflearn.fully_connected(net, num_actions, activation='tanh', weights_init=w_init)
            # Scale output to -action_bound to action_bound
            self.scaled_out = (((self.outs - tanh_min_bounds)/ (tanh_max_bounds - tanh_min_bounds)) * (max_bounds - min_bounds) + min_bounds) #broadcasts the bound arrays
            

        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outerscope+"/"+self.name)
        self.ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=outerscope+"/"+self.name)      
        
        
        
        
        
class Actor(object):
    def __init__(self, conf, agent, session):
        self.conf = conf
        self.agent = agent
        self.session = session
        
        with tf.variable_scope("actor"):
            
            self.online = actorNet(conf, agent)
            self.target = actorNet(conf, agent, name="target")
        
            self.smoothTargetUpdate = _netCopyOps(self.online, self.target, self.conf.target_update_tau)
    
            # This gradient will be provided by the critic network
            self.action_gradient = tf.placeholder(tf.float32, [None, self.conf.num_actions], name="actiongradient")
    
            # Combine the gradients here
            self.actor_gradients = tf.gradients(self.online.scaled_out, self.online.trainables, -self.action_gradient)
    
            # Optimization Op
            self.optimize = tf.train.AdamOptimizer(self.conf.actor_lr).apply_gradients(zip(self.actor_gradients, self.online.trainables))
    


    def train(self, inputs, a_gradient):
        self.session.run(self.optimize, feed_dict={self.online.inputs: inputs,self.action_gradient: a_gradient})

    def predict(self, inputs, which="online"):
        net = self.online if which == "online" else self.target
        return self.session.run(net.scaled_out, feed_dict={net.inputs: inputs})

    def update_target_network(self):
        self.session.run(self.smoothTargetUpdate)

        
        
        
        
        
class criticNet():
     def __init__(self, conf, agent, outerscope="critic", name="online"):       
        self.conf = conf
        self.agent = agent
        self.name = name  
        num_actions = conf.num_actions
        conv_stacksize = (self.conf.history_frame_nr*2 if self.conf.use_second_camera else self.conf.history_frame_nr) if self.agent.conv_stacked else 1
        
        #deleteme
        s_dim = 3

        with tf.variable_scope(name):
            self.conv_inputs = tflearn.input_data(shape=[None, s_dim])
            self.actions = tflearn.input_data(shape=[None, num_actions])
            net = tflearn.fully_connected(self.conv_inputs, 400, activation='relu')
            t1 = tflearn.fully_connected(net, 300)
            t2 = tflearn.fully_connected(self.actions, 300)
            net = tflearn.activation(tf.matmul(net, t1.W) + tf.matmul(self.actions, t2.W) + t2.b, activation='relu')
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            self.Q = tflearn.fully_connected(net, 1, weights_init=w_init)
            
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outerscope+"/"+self.name)
        self.ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=outerscope+"/"+self.name)      
        
       
        
        
        
        

class Critic(object):
    def __init__(self, conf, agent, session):
        self.conf = conf
        self.agent = agent
        self.session = session
        
        with tf.variable_scope("critic"):
            
            self.online = criticNet(conf, agent)
            self.target = criticNet(conf, agent, name="target")
                
            self.smoothTargetUpdate = _netCopyOps(self.online, self.target, self.conf.target_update_tau)
    
            self.target_Q = tf.placeholder(tf.float32, [None, 1], name="target_Q")
    
            self.loss = tf.losses.mean_squared_error(self.target_Q, self.online.Q)
            self.optimize = tf.train.AdamOptimizer(self.conf.critic_lr).minimize(self.loss)
    
            self.action_grads = tf.gradients(self.online.Q, self.online.actions)


    def train(self, inputs, action, target_Q):
        return self.session.run([self.online.Q, self.optimize, self.loss], feed_dict={self.online.conv_inputs: inputs,self.online.actions: action,self.target_Q: target_Q})

    def predict(self, inputs, action, which="online"):
        net = self.online if which == "online" else self.target
        return self.session.run(net.Q, feed_dict={net.conv_inputs: inputs, net.actions: action})

    def action_gradients(self, inputs, actions):
        return self.session.run(self.action_grads, feed_dict={self.online.conv_inputs: inputs,self.online.actions: actions})

    def update_target_network(self):
        self.session.run(self.smoothTargetUpdate)

        
    
        
class DDPG_model():
    
    def __init__(self, conf, agent, session):
        self.conf = conf
        self.agent = agent
        self.session = session
        self.initNet()
        
        
    def initNet(self):
        
        self.actor = Actor(self.conf, self.agent, self.session)
        self.critic = Critic(self.conf, self.agent, self.session)     
        
        self.session.run(tf.global_variables_initializer())
        self.session.run(_netCopyOps(self.actor.target, self.actor.online))
        self.session.run(_netCopyOps(self.critic.target, self.critic.online))      
        
    
        
    #actor predicts action. critic predicts q-value of action. That is compared to the actual q-value of action. (TD-Error)
    #online-actor predicts new actions. actor uses critic's gradients to train, too.
    def train_step(self, batch):
        oldstates, actions, rewards, terminals, newstates = batch

        #Training the critic...
        target_q = self.critic.predict(newstates, self.actor.predict(newstates, "target"), "target")
        cumrewards = np.reshape([rewards[i] if terminals[i] else rewards[i]+0.99*target_q[i] for i in range(len(rewards))], (len(rewards),1))
        target_Q, _, loss = self.critic.train(oldstates, actions, cumrewards)

        #training the actor...        
        a_outs = self.actor.predict(oldstates)
        grads = self.critic.action_gradients(oldstates, a_outs)
        self.actor.train(oldstates, grads[0])

        self.actor.update_target_network()
        self.critic.update_target_network()
        
        return np.amax(target_Q)
        
    def inference(self, oldstates):
        return self.actor.predict(oldstates, "online") #ist halt schneller wenn online.
        

        
        
        
def train(sess, env, model, s_dim, a_dim):
    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    for i in range(MAX_EPISODES):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(MAX_EP_STEPS):

            if RENDER_ENV:
                env.render()

            # Added exploration noise
            
            a = model.inference(np.reshape(s, (1, 3))) + (1. / (1. + i))
            
            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(s, (s_dim,)), np.reshape(a, (a_dim,)), r, terminal, np.reshape(s2, (s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                batch = replay_buffer.sample_batch(MINIBATCH_SIZE)
                
                    
                ep_ave_max_q += model.train_step(batch)
                

            s = s2
            ep_reward += r

            if terminal:
                print('| Reward: %.2i' % int(ep_reward), " | Episode", i, '| Qmax: %.4f' % (ep_ave_max_q / float(j)))
                break


def main(_):
    import config
    conf = config.Config()
    conf.target_update_tau = 1e-3
    conf.num_actions = 1    
    conf.action_bounds = [(-2, 2)]
    import read_supervised
    from server import Containers
    import dqn_rl_agent
    myAgent = dqn_rl_agent.Agent(conf, Containers(), True)
    
    tf.reset_default_graph()
    
    with tf.Session() as sess:

        env = gym.make(ENV_NAME)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        model = DDPG_model(conf, myAgent, tf.Session())

       
        
        if GYM_MONITOR_EN:
            if not RENDER_ENV:
                env = wrappers.Monitor(
                    env, MONITOR_DIR, video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, MONITOR_DIR, force=True)

        train(sess, env, model, state_dim, action_dim)

        if GYM_MONITOR_EN:
            env.monitor.close()

if __name__ == '__main__':
    tf.app.run()