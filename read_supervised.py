import xml.etree.ElementTree as ET
import os
import tensorflow as tf
import numpy as np
from server import cutoutandreturnvectors

#
FOLDERNAME = "SavedLaps/"

class Config(object):
    history_frame_nr = 1
    batch_size = 32
    num_steps = 20 #Uhm, kp warum dieser wert das sein sollte
    image_dims = [30,42]
    keep_prob = 0.8
    initscale = 0.1
    iterations = 10
    
    
    
#this is supposed to resemble the TrackingPoint-Class from the recorder from Unity
class TrackingPoint(object):
    def __init__(self, time, throttlePedalValue, brakePedalValue, steeringValue, progress, vectors):
        self.time = time
        self.throttlePedalValue = throttlePedalValue
        self.brakePedalValue = brakePedalValue
        self.steeringValue = steeringValue
        self.progress = progress
        self.vectors = vectors
        
    def make_vecs(self):
       if self.vectors != "":
           self.visionvec, self.AllOneDs = cutoutandreturnvectors(self.vectors)
           self.vectors = ""
    
    
def read_all_xmls(foldername):
    assert os.path.isdir(foldername) 
    all_trackingpoints = []
    for file in os.listdir(foldername):
        if file.endswith(".svlap"):
            all_trackingpoints.extend(read_xml(os.path.join(foldername, file)))
    for currpoint in all_trackingpoints:
        currpoint.make_vecs();            
    return all_trackingpoints        


            
def read_xml(FileName):
    this_trackingpoints = []
    tree = ET.parse(FileName)
    root = tree.getroot()
    assert root.tag=="ArrayOfTrackingPoint", "that is not the kind of XML I thought it would be."
    for currpoint in root:
        inputdict = {}
        for item in currpoint:
            inputdict[item.tag] = item.text
        tp = TrackingPoint(**inputdict) #ein dictionary mit kwargs, IM SO PYTHON!!
        this_trackingpoints.append(tp)
    return this_trackingpoints


#TODO: sample according to information gain, what DQN didn't do yet.
#TODO: uhm, das st jezt simples ziehen mit zurücklegen, every time... ne richtige next_batch funktion, bei der jedes mal vorkommt wäre sinnvoller, oder?
def sample_batch(config, dataset):
    indices = np.random.choice(len(dataset), config.batch_size)
    visions = []
    targets = []
    for i in indices:
        vision = [dataset[(i-j) % len(dataset)].visionvec for j in range(config.history_frame_nr,-1,-1)]
        target = [dataset[i].throttlePedalValue, dataset[i].brakePedalValue, dataset[i].steeringValue]
        if config.history_frame_nr == 1: 
            vision = vision[0]
        visions.append(vision)
        targets.append(target)
    return visions, targets



#what do we want? stride 1, SAME-padding
class CNN(object):
    def __init__(self, config, is_training=False): 
        self.config = config
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.in_x = config.image_dims[0]
        self.in_y = config.image_dims[1]

        self.inputs = tf.placeholder(tf.float32, shape=[None,self.in_x, self.in_y], name="inputs") #TODO: eigentlich ist die letzte dimension ja config.history_frame_nr
        self.targets = tf.placeholder(tf.float32, shape=[None, 3], name="targets") 
        
        def weight_variable(shape):
          initial = tf.truncated_normal(shape, stddev=0.1)
          return tf.Variable(initial)
        
        def bias_variable(shape):
          initial = tf.constant(0.1, shape=shape)
          return tf.Variable(initial)
        
        def conv2d(x, W):
          return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        
        def max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
        self.inputs = tf.reshape(self.inputs,[-1, self.in_x, self.in_y, 1])

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
                
        h_conv1 = tf.nn.relu(conv2d(self.inputs, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)  #reduces in this case to 15*21
        
        
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2) #reduces to... 8*11?
        
        W_fc1 = weight_variable([8 * 11 * 64, 1024])
        b_fc1 = bias_variable([1024])
        
        h_pool2_flat = tf.reshape(h_pool2, [-1, 8*11*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1 )       

        if is_training:
            h_fc1 = tf.nn.dropout(h_fc1, config.keep_prob) 
        
        W_fc2 = weight_variable([1024, 3])
        b_fc2 = bias_variable([3])
        
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2 #standard FF stuff here.
                          
        if not is_training:
            return
      
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.targets, logits=y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self.targets,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
    def run_epoch(self, session, dataset, eval_op=None, printstuff=False):
        epoch_size = len(dataset) // self.batch_size

        for i in range(epoch_size):
            visions, targets = sample_batch(self.config, dataset)
            session.run(self.train_step, feed_dict={self.inputs:visions, self.targets: targets})
            
            
        return self.accuracy.eval(feed_dict={self.inputs:visions, self.targets: targets})




def run_CNN(dataset, config):
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.initscale, config.initscale)

        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = CNN(config, is_training=True)

        with tf.Session() as session:
            init = tf.global_variables_initializer()
            init.run()
            print("Running for",config.iterations,"iterations.")
            for i in range(config.iterations):
                print("Epoch: %d" % (i+1))
                train_loss = m.run_epoch(session, dataset)
                print("accuracy %g"%train_loss)     






if __name__ == '__main__':    
    config = Config()
    all_trackingpoints = read_all_xmls(FOLDERNAME)
    print("Number of samples:",len(all_trackingpoints))
    run_CNN(all_trackingpoints,config)
    #visions, targets = sample_batch(config, all_trackingpoints)
    
    
    #sooo jetzt hab ich hier eine liste an trackingpoints. was ich tun muss ist jetzt dem vectors per ANN die brake, steering, throttlevalues zuzuweisen.
    