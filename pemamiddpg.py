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

    
def _runMultipleOps(op_holder,sess):
    for op in op_holder:
        sess.run(op)      
        
        
        
        
        
        
class actorNet():
     def __init__(self, conf, agent, s_dim, a_dim, action_bound, outerscope="actor", name="online", reuse=False):       
        self.name = name
        with tf.variable_scope(name):
            self.inputs = tflearn.input_data(shape=[None, s_dim])
            net = tflearn.fully_connected(self.inputs, 400, activation='relu')
            net = tflearn.fully_connected(net, 300, activation='relu')
            # Final layer weights are init to Uniform[-3e-3, 3e-3]
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            self.out = tflearn.fully_connected(net, a_dim, activation='tanh', weights_init=w_init)
            # Scale output to -action_bound to action_bound
            self.scaled_out = tf.multiply(self.out, action_bound)
            
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outerscope+"/"+self.name)
        
        
        
        
        
class Actor(object):
    def __init__(self, session, state_dim, action_dim, action_bound, learning_rate, tau):
        with tf.variable_scope("actor"):
        
            self.session = session
            self.s_dim = state_dim
            self.a_dim = action_dim
            self.action_bound = action_bound
            self.learning_rate = learning_rate
            self.tau = tau
    
            self.online = actorNet(None, None, self.s_dim, self.a_dim, self.action_bound)
            self.target = actorNet(None, None, self.s_dim, self.a_dim, self.action_bound, name="target")
        
            self.smoothTargetUpdate = _netCopyOps(self.online, self.target, self.tau)
    
            # This gradient will be provided by the critic network
            self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim], name="actiongradient")
    
            # Combine the gradients here
            self.actor_gradients = tf.gradients(self.online.scaled_out, self.online.trainables, -self.action_gradient)
    
            # Optimization Op
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.online.trainables))
    


    def train(self, inputs, a_gradient):
        self.session.run(self.optimize, feed_dict={self.online.inputs: inputs,self.action_gradient: a_gradient})

    def predict(self, inputs, which="online"):
        net = self.online if which == "online" else self.target
        return self.session.run(net.scaled_out, feed_dict={net.inputs: inputs})

    def update_target_network(self):
        self.session.run(self.smoothTargetUpdate)

        
        
        
        
        
class criticNet():
     def __init__(self, conf, agent, s_dim, a_dim, action_bound=None, outerscope="critic", name="online", reuse=False):       
        self.name = name
        with tf.variable_scope(name):
            self.inputs = tflearn.input_data(shape=[None, s_dim])
            self.action = tflearn.input_data(shape=[None, a_dim])
            net = tflearn.fully_connected(self.inputs, 400, activation='relu')
            t1 = tflearn.fully_connected(net, 300)
            t2 = tflearn.fully_connected(self.action, 300)
            net = tflearn.activation(tf.matmul(net, t1.W) + tf.matmul(self.action, t2.W) + t2.b, activation='relu')
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            self.Q = tflearn.fully_connected(net, 1, weights_init=w_init)
            
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outerscope+"/"+self.name)
        
       
        
        
        
        

class Critic(object):
    def __init__(self, session, state_dim, action_dim, learning_rate, tau):
        
        with tf.variable_scope("critic"):
            self.session = session
            self.s_dim = state_dim
            self.a_dim = action_dim
            self.learning_rate = learning_rate
            self.tau = tau
                    
            self.online = criticNet(None, None, self.s_dim, self.a_dim)
            self.target = criticNet(None, None, self.s_dim, self.a_dim, name="target")
                
            self.smoothTargetUpdate = _netCopyOps(self.online, self.target, self.tau)
    
            self.target_Q = tf.placeholder(tf.float32, [None, 1], name="target_Q")
    
            self.loss = tf.losses.mean_squared_error(self.target_Q, self.online.Q)
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
    
            self.action_grads = tf.gradients(self.online.Q, self.online.action)


    def train(self, inputs, action, target_Q):
        return self.session.run([self.online.Q, self.optimize, self.loss], feed_dict={self.online.inputs: inputs,self.online.action: action,self.target_Q: target_Q})

    def predict(self, inputs, action, which="online"):
        net = self.online if which == "online" else self.target
        return self.session.run(net.Q, feed_dict={net.inputs: inputs, net.action: action})

    def action_gradients(self, inputs, actions):
        return self.session.run(self.action_grads, feed_dict={self.online.inputs: inputs,self.online.action: actions})

    def update_target_network(self):
        self.session.run(self.smoothTargetUpdate)

        
    
        
class DDPG_model():
    
    def __init__(self, conf, agent, session, state_dim, action_dim, action_bound):
        self.conf = conf
        self.agent = agent
        self.session = session
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.initNet()
        
        
    def initNet(self):
        
        self.actor = Actor(self.session, self.s_dim, self.a_dim, self.action_bound, ACTOR_LEARNING_RATE, TAU)
        self.critic = Critic(self.session, self.s_dim, self.a_dim, CRITIC_LEARNING_RATE, TAU)     
        
        
        self.session.run(tf.global_variables_initializer())
        _runMultipleOps(_netCopyOps(self.actor.target, self.actor.online), self.session)
        _runMultipleOps(_netCopyOps(self.critic.target, self.critic.online), self.session)        
        
    
        
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

        
        self.actor.update_target_network()
        self.critic.update_target_network()
        
        return np.amax(target_Q)
        

def train(sess, env, model):
    
    

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
            a = model.actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))

            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(s, (model.s_dim,)), np.reshape(a, (model.a_dim,)), r,
                              terminal, np.reshape(s2, (model.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                batch = replay_buffer.sample_batch(MINIBATCH_SIZE)
                
                    
                ep_ave_max_q += model.train_step(batch)
                    
                # Calculate targets
              

            s = s2
            ep_reward += r

            if terminal:


                print('| Reward: %.2i' % int(ep_reward), " | Episode", i, '| Qmax: %.4f' % (ep_ave_max_q / float(j)))

                break


def main(_):
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

        model = DDPG_model(None, None, tf.Session(), state_dim, action_dim, action_bound)

       
        
        if GYM_MONITOR_EN:
            if not RENDER_ENV:
                env = wrappers.Monitor(
                    env, MONITOR_DIR, video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, MONITOR_DIR, force=True)

        train(sess, env, model)

        if GYM_MONITOR_EN:
            env.monitor.close()

if __name__ == '__main__':
    tf.app.run()