# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 19:26:53 2017

@author: nivradmin
"""

import tensorflow as tf


                                      
                                      
class FFNN(object):
    
    ######methods for BUILDING the computation graph######
    def __init__(self):
        #setzt num_steps undso... 
        #erzeugt die leeren (dimension=none, für inference!!!) placeholder für input und ggf. output...
        #kriegt auch info darüber was er speichern soll???
        
        self.targets = tf.placeholder(tf.float32, shape=[None,3], name="targets")    
        
        self.targets2 = tf.placeholder(tf.float32, shape=[None,3], name="targets2")    
        
        self.targets = tf.cast(self.targets, tf.bool)
        self.targets2 = tf.cast(self.targets2, tf.bool)
        
        correct = tf.equal(self.targets, self.targets2)
        self.compare = tf.reduce_all(correct,axis=1)
        
        
        self.compare = tf.cast(self.compare, tf.float32) #gucke wo das hier gleich der lenght ist
        self.compare = tf.reduce_mean(self.compare)
                              
        #self.compare = tf.shape(self.compare)[1]
        #self.compare = tf.reduce_sum(self.compare, axis=1)
        


    def run_train_epoch(self, session):
        for i in range(1):
            feed_dict = {self.targets: [[0,0,1],[0,1,0],[0,0,0]], self.targets2: [[0,0,0],[0,1,0],[0,0,0]]}
            t1, t2, compare = session.run([self.targets, self.targets2, self.compare], feed_dict=feed_dict)   
            print(t1,"\n")
            print(t2,"\n")
            print(compare)
            
            
            
      
       
if __name__ == '__main__':    
    
    with tf.Graph().as_default():    
        initializer = tf.constant_initializer(0) #bei variablescopes kann ich nen default-initializer für get_variables festlegen
        
        with tf.name_scope("train"):
            with tf.variable_scope("steermodel", reuse=None, initializer=initializer):
                ffnn = FFNN()
        

        init = tf.global_variables_initializer()
#        saver = tf.train.Saver() #der sollte ja nur einige werte machen
        
      
        with tf.Session() as sess:
#            summary_writer = tf.summary.FileWriter("filewriterlogdir", sess.graph)
            sess.run(init)
                
            for step in range(1):
                train_loss = ffnn.run_train_epoch(sess)
                
            