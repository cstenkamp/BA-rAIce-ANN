#!/usr/bin/env python

import numpy as np
import socket
import threading
import time
import logging
import hashlib
import tensorflow as tf
#====own functions====
import supervisedcnn 
import read_supervised


logging.basicConfig(level=logging.ERROR,
                    format='(%(threadName)-10s) %(message)s',)


TCP_IP = 'localhost'
TCP_RECEIVER_PORT = 6435
TCP_SENDER_PORT = 6436
ANNCHECKALL = 100

current_milli_time = lambda: int(round(time.time() * 1000))


class MySocket:

    def __init__(self, sock=None):
        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock
        self.sock.settimeout(10.0)
            
    def connect(self, host, port):
        self.sock.connect((host, port))

    def bind(self, host, port):
        self.sock.bind((host, port))
        
    def listen(self,queueup):
        self.sock.listen(queueup)
        
    def close(self):
        self.sock.close()
        
        
        
    def mysend(self, msg):
#        length = str(len(msg))
#        while len(length) < 5:
#            length = "0"+length
#        msg = length+msg  #TODO Eigentlich doch da, oder?
        
        msg = msg.encode()
        totalsent = 0
        while totalsent < len(msg):
            sent = self.sock.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            totalsent = totalsent + sent

            
    #I think the speed would increase manyfold if I don't close the
    #connection right away but keep the sockets communicating
            
    def myreceive(self):
        chunks = []
        bytes_recd = 0
        try:
            what = self.sock.recv(5).decode('ascii')
        except socket.timeout:
            print("Socket timed out, please tell me you continue")
            return False
        if not what:
            return False
        if what[0] not in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:
            msglen = 1000;
        else:
            msglen = int(what)
        while bytes_recd < msglen:
            chunk = self.sock.recv(min(msglen - bytes_recd, 2048))
            if chunk == b'':
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)
        final = b''.join(chunks)
        final = final.decode()
        return final

  
        
###############################################################################

# alle X sekunden kommt ein leser und liest den InputVal aus (immer der neueste, mit timestamp!!), und updated damit 
# den OutputValContainer, falls der nicht schon einen neueren inputval-timestamp hat.
class InputValContainer(object):   
    def __init__(self):
        self.lock = threading.Lock()
        self.visionvec = np.zeros([20,20]) #TODO - gucken ob die größe gleich ist/wie ich die größe share
        self.othervecs = np.zeros(50)      #TODO - same
        self.timestamp = current_milli_time()
        self.hashco = ""
        self.alreadyread = True
        
    def update(self, visionvec, othervecs, hashco):
        logging.debug('Inputval-Update: Waiting for lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired lock')
            if self.hashco != hashco:
                self.visionvec = visionvec
                self.othervecs = othervecs
                self.timestamp = current_milli_time()
                self.alreadyread = False
                self.hashco = hashco
                print("Updated first vec to", self.othervecs)
        finally:
            self.lock.release()
            
    def reset(self):
        logging.debug('Inputval-Reset: Waiting for lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired lock')
            self.visionvec = np.zeros(20,20) #TODO - gucken ob die größe gleich ist/wie ich die größe share
            self.othervecs = np.zeros(50)    #TODO - same
            self.timestamp = current_milli_time()
            self.hashco = ""
            self.alreadyread = True
            logging.debug("Resettet input-value")
        finally:
            self.lock.release()

    def read(self):
        self.alreadyread = True
        return self.othervecs, self.visionvec
    



class NeuralNetworkThread(threading.Thread):
    def __init__(self):
        self.containers = None
        self.laststart = current_milli_time()                
        threading.Thread.__init__(self)
        tps = read_supervised.TPList(read_supervised.FOLDERNAME)
        self.normalizers = tps.find_normalizers()
        self.initNetwork()

    #TODO: diese beiden von der read_supervised nehmen und nicht nochmal neu definieren!
    @staticmethod
    def flatten_oneDs(AllOneDs):
        return np.array(read_supervised.flatten(AllOneDs))
    
    @staticmethod
    def normalize_oneDs(FlatOneDs, normalizers):
        FlatOneDs -= np.array([item[0] for item in normalizers])
        NormalizedOneDs = FlatOneDs / np.array([item[1] for item in normalizers])
        return NormalizedOneDs
        
    
    def run(self):
        while self.containers.KeepRunning:          
            if current_milli_time() > self.laststart - ANNCHECKALL:
                if not self.containers.inputval.alreadyread:   
                    print("Another ANN Starts")                   
                    returnstuff = self.performNetwork(self.containers.inputval)
                    self.containers.outputval.update(returnstuff, self.containers.inputval.timestamp)    
                self.laststart = current_milli_time()



#    def performSteerNetwork(self, inputval): 
#        inputvec, visionvec = inputval.read()
#
#        if inputvec:
#            inputvec = np.array(NeuralNetworkThread.normalize_oneDs(NeuralNetworkThread.flatten_oneDs(inputvec),self.normalizers))
#            if inputvec[0] > 0.3 and inputvec[0] < 0.4:
#                return "pleasereset"
#            else:
#                steer = read_supervised.dediscretize_steering((self.ffnn.run_inference(self.session, inputvec))[0])
#                result = "[0.1, 0, "+str(steer)+"]"
#                return result
#        else:
#            return

    def performNetwork(self, inputval):
        _, visionvec = inputval.read()
        
        check, networkresult = self.cnn.run_inference(self.session, visionvec)
        if check:
            throttle, brake, steer = read_supervised.dediscretize_all((networkresult)[0])
            result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
            return result
        else:
            return "[0,0,0.0]" #TODO: macht das sinn 0,0 als standard zu returnen?


#    def initSteerNetwork(self):
#        config = supervisedcnn.Config()
#        
#        with tf.Graph().as_default():    
#            initializer = tf.random_uniform_initializer(-0.1, 0.1)
#                                                 
#            with tf.name_scope("runAsServ"):
#                with tf.variable_scope("steermodel", reuse=None, initializer=initializer): 
#                    self.ffnn = supervisedcnn.FFNN_lookahead_steer(config)
#            
#            self.saver = tf.train.Saver({"W1": self.ffnn.W1, "b1": self.ffnn.b1, "W2": self.ffnn.W2, "b2": self.ffnn.b2}) 
#            self.session = tf.Session()
#            ckpt = tf.train.get_checkpoint_state(config.log_dir) 
#            assert ckpt and ckpt.model_checkpoint_path
#            self.saver.restore(self.session, ckpt.model_checkpoint_path)
#            print("network should be initialized")


    def initNetwork(self):
        config = supervisedcnn.Config()
        
        with tf.Graph().as_default():    
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
                                                 
            with tf.name_scope("runAsServ"):
                with tf.variable_scope("cnnmodel", reuse=None, initializer=initializer): 
                    self.cnn = supervisedcnn.CNN(config, is_training=False)
            
            self.saver = tf.train.Saver(self.cnn.trainvars)
            self.session = tf.Session()
            ckpt = tf.train.get_checkpoint_state(config.checkpoint_dir) 
            assert ckpt and ckpt.model_checkpoint_path
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            print("network should be initialized")




class OutputValContainer(object):    
    def __init__(self):
        self.lock = threading.Lock()
        self.value = ""
        self.timestamp = current_milli_time()
        self.alreadysent = True #nen leeres ding braucht er nicht schicken
        
    #you update only if the new input-timestamp > der alte (in case on ANN-Thread was superslow and thus outdated)
    def update(self, withwhatval, itstimestamp):
        logging.debug('Outputval-Update: Waiting for lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired lock')
            if self.timestamp < itstimestamp:
                self.value = withwhatval
                self.timestamp = itstimestamp #es geht nicht um jetzt, sondern um dann als das ANN gestartet wurde
                self.alreadysent = False
                print("Updated output-value to",withwhatval)
        finally:
            self.lock.release()

    def reset(self):
        logging.debug('Outputval-reset: Waiting for lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired lock')
            self.value = ""
            self.timestamp = current_milli_time() #so kann ich verhindern dass noch-laufende-ANNs den noch updaten
            self.alreadysent = True
            logging.debug("Resettet output-value")
        finally:
            self.lock.release()
            

    def read(self):
        if not self.alreadysent:
            self.alreadysent = True
            return self.value
        else:
            return False

  

###############################################################################

#the receiver-thread gets called when unity wants, and tries to update the 
#global inputval, containing the race-info, as often as possible.
class receiver_thread(threading.Thread):
    def __init__(self, clientsocket):
        threading.Thread.__init__(self)
        self.clientsocket = clientsocket
        self.containers = None
        
    def run(self):
        #print("Starting Thread")
        data = self.clientsocket.myreceive()
        if data: 
            #print("received data:", data)            
            #TODO - ERST den hash vom string vergleichen damit das updaten schneller geht?
            visionvec, allOneDs = cutoutandreturnvectors(data) 
            hashco = (hashlib.md5(data.encode('utf-8'))).hexdigest()
            
            self.containers.inputval.update(visionvec, allOneDs, hashco)
        self.clientsocket.close()
   
        
def cutoutandreturnvectors(string):
    allOneDs  = []
    visionvec = [[]]    
    def cutout(string, letter):
        return string[string.find(letter)+2:string[string.find(letter):].find(")")+string.find(letter)]
    
    if string.find("P(") > -1:
        #print("Progress as real Number",self.readOneDArrayFromString(cutout(data, "P(")))
        allOneDs.append(readOneDArrayFromString(cutout(string, "P(")))

    if string.find("S(") > -1:
        #print("SpeedStearVec",self.readOneDArrayFromString(cutout(data, "S(")))
        allOneDs.append(readOneDArrayFromString(cutout(string, "S(")))

    if string.find("T(") > -1:
        #print("CarStatusVec",self.readOneDArrayFromString(cutout(data, "T(")))
        allOneDs.append(readOneDArrayFromString(cutout(string, "T(")))
        
    if string.find("C(") > -1:
        #print("Visionvec",self.readTwoDArrayFromString(cutout(data, "V(")))
        allOneDs.append(readOneDArrayFromString(cutout(string, "C(")))
        
    if string.find("L(") > -1:
        #print("Visionvec",self.readTwoDArrayFromString(cutout(data, "V(")))
        allOneDs.append(readOneDArrayFromString(cutout(string, "L(")))
    
    if string.find("V(") > -1:
        #print("Visionvec",self.readTwoDArrayFromString(cutout(data, "V(")))
        visionvec = readTwoDArrayFromString(cutout(string, "V("))    
        
    return visionvec, allOneDs
        

def readOneDArrayFromString(string):
    tmpstrings = string.split(",")
    tmpfloats = []
    for i in tmpstrings:
        tmp = i.replace(" ","")
        if len(tmp) > 0:
            try:
                x = float(str(tmp))
                tmpfloats.append(x)
            except ValueError:
                print("I'm crying") #cry.
    return tmpfloats


def ternary(n):
    if n == 0:
        return '0'
    nums = []
    if n < 0:
        n*=-1
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums))


def readTwoDArrayFromString(string):
    tmpstrings = string.split(",")
    tmpreturn = []
    for i in tmpstrings:
        tmp = i.replace(" ","")
        if len(tmp) > 0:
            try:
                currline = []
                for j in tmp:
                    currline.append(int(j))
                tmpreturn.append(currline)
            except ValueError:
                print("I'm crying") #cry.
    return np.array(tmpreturn)
    
    
        

###############################################################################
        
#the sender-thread gets called when unity asks for the result of a new network iteration...
#so python aks the ValContainer outputval if it the flag alreadysent is false, if so, it returns it, 
#if not, it simply closes the connection.        
class sender_thread(threading.Thread):
    def __init__(self, clientsocket):
        threading.Thread.__init__(self)
        self.clientsocket = clientsocket
        self.containers = None
        
    def run(self):
        #print("Starting Thread")
        print("Unity asked for the result")
        outputvaltxt = self.containers.outputval.read() #false if already read.
        if outputvaltxt:
            print("Sending ANN result back, because Unity asked: ", outputvaltxt)
            self.clientsocket.mysend(outputvaltxt)
        else:
            self.clientsocket.mysend("[0.1, 0, 0.0]")
            #self.clientsocket.close()



###############################################################################
##################### ACTUAL STARTING OF THE STUFF#############################
###############################################################################

class Containers():
    def __init__(self):
        self.IExist = True
        self.KeepRunning = True
        
        
def create_socket(port):
    server_socket = MySocket()
    server_socket.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(TCP_IP, port)
    server_socket.listen(1)
    return server_socket

    
    
#now I need three threads:
#   one constantly looking for clients sending information here
#   one constantly running the ANN...
#   one constantly looking for clients demanding information here
    


class ReceiverListenerThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.containers = None
        
    def run(self):
        while self.containers.KeepRunning:
            #print("receiver connected")
            try:
                (client, addr) = self.containers.receiverportsocket.sock.accept()
                clt = MySocket(client)
                ct = receiver_thread(clt)
                ct.containers = self.containers
                ct.start()
            except socket.timeout:
                pass      
        
        
            
class SenderListenerThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.containers = None
        
    def run(self):
        while self.containers.KeepRunning:
            #print("sender connected")
            try:
                (client, addr) = self.containers.senderportsocket.sock.accept()
                clt = MySocket(client)
                ct = sender_thread(clt)
                ct.containers = self.containers
                ct.start()
            except socket.timeout:
                pass

    
def main():
    containers = Containers()
    containers.inputval = InputValContainer()
    containers.outputval = OutputValContainer()
    
    containers.receiverportsocket = create_socket(TCP_RECEIVER_PORT)
    containers.senderportsocket = create_socket(TCP_SENDER_PORT)
    
    
    #THREAD 1
    ANNThread = NeuralNetworkThread()
    ANNThread.containers = containers
    ANNThread.start()
    
    #THREAD 2
    ReceiverConnecterThread = ReceiverListenerThread()
    ReceiverConnecterThread.containers = containers
    ReceiverConnecterThread.start()
    
    #THREAD 3 
    SenderConnecterThread = SenderListenerThread()
    SenderConnecterThread.containers = containers
    SenderConnecterThread.start()

    try:        
        while True:
            None
    except KeyboardInterrupt:
        pass
    
    print("Server shutting down...")
    containers.KeepRunning = False
    ANNThread.join()
    ReceiverConnecterThread.join() #takes max. 10 seconds until socket timeouts
    SenderConnecterThread.join()
    print("Server shut down sucessfully.")


    
if __name__ == '__main__':        
    main()