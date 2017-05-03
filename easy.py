#!/usr/bin/env python

import socket
import threading
import time
import logging
import numpy as np
import tensorflow as tf
#====own functions====
import supervisedcnn 
import read_supervised

logging.basicConfig(level=logging.ERROR, format='(%(threadName)-10s) %(message)s',)


TCP_IP = 'localhost'
TCP_RECEIVER_PORT = 6435
TCP_SENDER_PORT = 6436
UPDATE_ONLY_IF_NEW = False #sendet immer nach jedem update -> Wenn False sendet er wannimmer er was kriegt

current_milli_time = lambda: int(round(time.time() * 1000))


class MySocket:

    def __init__(self, sock=None):
        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock
        self.sock.settimeout(1.0)
            
    def connect(self, host, port):
        self.sock.connect((host, port))

    def bind(self, host, port):
        self.sock.bind((host, port))
        
    def listen(self,queueup):
        self.sock.listen(queueup)
        
    def close(self):
        self.sock.close()
        
        
        
    def mysend(self, msg):
        length = str(len(msg))
        while len(length) < 5:
            length = "0"+length
        msg = length+msg 
        
        msg = msg.encode()
        totalsent = 0
        while totalsent < len(msg):
            sent = self.sock.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            totalsent = totalsent + sent


    def myreceive(self):
        chunks = []
        bytes_recd = 0
        try:
            what = self.sock.recv(5).decode('ascii')
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
        except socket.timeout:
            raise TimeoutError("Socket timed out")
  
        
  

###############################################################################

# there is a receiver-listener-thread, constantly waiting on the receiverport if unity wants to connect.
# Unity will connect only once, and if so, the receiverlistenerthread will create a new receiver_thread, 
# handling everything from there on. If unity wants to reconnect for any reason, it will create a new receiver_thread...
# where the new one should kill the old one, such that there is only one receiver_thread active most of the time.
class ReceiverListenerThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.containers = None
        
    def run(self):
        assert self.containers != None, "after creating the thread, you have to assign containers"
        while self.containers.KeepRunning:
            #print("receiver connected")
            try:
                (client, addr) = self.containers.receiverportsocket.sock.accept()
                clt = MySocket(client)
                ct = receiver_thread(clt)
                ct.containers = self.containers
                ct.start()
                self.containers.receiverthreads.append(ct)
            except socket.timeout:
                #simply restart and don't care
                pass   


# A receiver-thread represents a stable connection to unity. It runs constantly, getting the newest information from unity constantly.
# If unity deconnects, it stops. If any instance of this finds receiver_threads with older data, it will deem the thread deprecated 
# and kill it. The receiver_thread updates the global inputval (which is certainly only one!), containing the race-info, as soon as it gets new info.
class receiver_thread(threading.Thread):
    def __init__(self, clientsocket):
        threading.Thread.__init__(self)
        self.clientsocket = clientsocket
        self.containers = None
        self.killme = False
        self.timestamp = 0
        
    def run(self):
        print("Starting receiver_thread")
        while self.containers.KeepRunning and (not self.killme):
            try:            
                data = self.clientsocket.myreceive()
                if data: 
                    #print("received data:", data)       
                    if data[:11] == "resetServer":
                        self.containers.inputval.reset(data[11:])
                    elif data[:5] == "Time(":
                        self.timestamp = float(data[5:data.find(")")])
                        for i in self.containers.receiverthreads:
                            if int(i.timestamp) < int(self.timestamp):
                                i.killme = True
                    
                        #print(data)
                        visionvec, allOneDs = cutoutandreturnvectors(data) 
                        self.containers.inputval.update(visionvec, allOneDs, self.timestamp) #we MUST have the inputval, otherwise there wouldn't be the possibility for historyframes.           
                        thread = threading.Thread(target=self.containers.ANN.runANN, args=())
                        thread.start()
                        
                    
            except TimeoutError:
                if len(self.containers.receiverthreads) < 2:
                    pass
                else:
                    break
                
        self.containers.receiverthreads.remove(self)
        print("stopping receiver_thread")
        
        
        
        
class InputValContainer(object):   
    
    def __init__(self, config):
        self.lock = threading.Lock()
        self.config = config
        self.visionvec = np.zeros([config.image_dims[0], config.image_dims[1]])
        if self.config.history_frame_nr > 1:
            self.vvec_hist = np.zeros([config.history_frame_nr, config.image_dims[0], config.image_dims[1]]) 
        self.othervecs = np.zeros(config.vector_len)      
        self.timestamp = 0
        self.containers = None
        self.alreadyread = True
        self.msperframe = config.msperframe
        
        
    def update(self, visionvec, othervecs, timestamp):
        logging.debug('Inputval-Update: Waiting for lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired lock')
            self.visionvec = visionvec
            if self.config.history_frame_nr > 1:
                self.vvec_hist = [visionvec] + [i for i in self.vvec_hist[:-1]]            
            self.othervecs = othervecs
            self.alreadyread = False
            self.timestamp = timestamp
            print("Updated Input-Vec at", self.timestamp)
        finally:
            self.lock.release()
            
            
    def reset(self, interval):
        logging.debug('Inputval-Reset: Waiting for lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired lock')
            self.visionvec = np.zeros([self.config.image_dims[0], self.config.image_dims[1]])
            if self.config.history_frame_nr > 1:
                self.vvec_hist = np.zeros([self.config.history_frame_nr, self.config.image_dims[0], self.config.image_dims[1]]) 
            self.othervecs = np.zeros(self.config.vector_len)
            self.timestamp = 0
            self.msperframe = interval #da Unity im diesen Wert immer bei spielstart schickt, wird msperframe immer richtig sein
            assert int(self.msperframe) == int(self.config.msperframe)
            self.alreadyread = True
            logging.debug("Resettet input-value")
        finally:
            self.lock.release()


    def read(self):
        self.alreadyread = True
        #print(self.visionvec) #TODO: der sollte nicht leer sein wenn not UPDATE_ONLY_IF_NEW
        if self.config.history_frame_nr > 1:
            return self.othervecs, self.vvec_hist
        else:
            return self.othervecs, self.visionvec
        
        
    
    
        

class OutputValContainer(object):    
    def __init__(self):
        self.lock = threading.Lock()
        self.value = ""
        self.timestamp = 0
        self.containers = None
        #self.alreadysent = True #nen leeres ding braucht er nicht schicken
        
    #you update only if the new input-timestamp > der alte (in case on ANN-Thread was superslow and thus outdated)
    def update(self, withwhatval, itstimestamp):
        logging.debug('Outputval-Update: Waiting for lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired lock')
            if int(self.timestamp) < int(itstimestamp):
                self.value = withwhatval
                self.timestamp = itstimestamp #es geht nicht um jetzt, sondern um dann als das ANN gestartet wurde
                #self.alreadysent = False
                print("Updated output-value to",withwhatval)
                self.send_via_senderthread(self.value, self.timestamp)
        finally:
            self.lock.release()

    def reset(self):
        logging.debug('Outputval-reset: Waiting for lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired lock')
            self.value = ""
            self.timestamp = 0
            #self.alreadysent = True
            logging.debug("Resettet output-value")
        finally:
            self.lock.release()
            
            
    def send_via_senderthread(self, value, timestamp):
        #nehme die erste verbindung die keinen error schemißt!        
        assert len(self.containers.senderthreads) > 0, "There is no senderthread at all! How will I send?"
        for i in range(len(self.containers.senderthreads)):
            try:
                self.containers.senderthreads[i].send(value, timestamp)
            except (ConnectionResetError, ConnectionAbortedError):
                    #if unity restarted, the old connection is now useless and should be deleted
                    print("I assume you just restarted Unity.")
                    self.containers.senderthreads[i].delete_me();
                    if i >= len(self.containers.senderthreads)-1:
                        break
                
#                try:
#                    print("this should occur after restarting unity")
#                    self.containers.senderthreads[i].delete_me();
#                except AttributeError:
#                    print("UHM")
##                    So, das hier ist der Fall wo Unity nen weiteren senderthread hätte erstellen müssen, und python muss den vorher erkennen. 
##                    HIERBEI NÄCHSTE MAL WEITERMAHCN!!!!
                                                     
        
        
#    def read(self):
#        if not self.alreadysent:
#            self.alreadysent = True
#            return self.value
#        else:
#            return False


###############################################################################        


#TODO: mehrere Network-objects haben, und das immer ein gerade un-beschäftigtes ausführen lassen 
class NeuralNetwork(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.isinitialized = False
        self.containers = None        
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
        
    
    def runANN(self):
        if self.isinitialized:
            if UPDATE_ONLY_IF_NEW and self.containers.inputval.alreadyread:
                    return
                
            self.lock.acquire()
            try:
                print("Another ANN Starts")                   
                returnstuff = self.performNetwork(self.containers.inputval)
                self.containers.outputval.update(returnstuff, self.containers.inputval.timestamp)    
            finally:
                self.lock.release()
                

    def performNetwork(self, inputval):
        _, visionvec = inputval.read()
        check, networkresult = self.cnn.run_inference(self.session, visionvec)
        if check:
            throttle, brake, steer = read_supervised.dediscretize_all((networkresult)[0])
            result = "["+str(throttle)+", "+str(brake)+", "+str(steer)+"]"
            return result
        else:
            return "[0,0,0.0]" #TODO: macht das sinn 0,0 als standard zu returnen? -> Exception stattdessen


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
            self.isinitialized = True


###############################################################################
    
#just like the ReceiverListenerThread waits for Unity connecting to RECEIVE info, this one waits for unity to connnect to SEND info TO UNITY
class SenderListenerThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.containers = None
        
    def run(self):
        assert self.containers != None, "after creating the thread, you have to assign containers"
        while self.containers.KeepRunning:
            try:
                (client, addr) = self.containers.senderportsocket.sock.accept()
                clt = MySocket(client)
                ct = sender_thread(clt)
                ct.containers = self.containers
                self.containers.senderthreads.append(ct)
                ct.start()
            except socket.timeout:
                #simply restart and don't care
                pass   



class sender_thread(threading.Thread):
    def __init__(self, clientsocket):
        threading.Thread.__init__(self)
        self.clientsocket = clientsocket
        self.containers = None
        
    def run(self):
        print("Starting sender_thread")
        
    def delete_me(self):
        selfind = self.containers.senderthreads.index(self)
        del self.containers.senderthreads[selfind]
        print("stopping sender_thread")
        #TODO: wird er jetzt automatisch garbage-collected oder muss ich ihn noch löschen?
        
        
    def send(self, result, timestamp):
        tosend = str(result) + "Time(" +str(timestamp)+")"
        print("Sending", tosend)
        self.clientsocket.mysend(tosend)
        #TODO: Kann er das command "pleasereset" senden????
#        while True:
#            self.clientsocket.mysend("[0.1, 0, 0.0]")
#                #self.clientsocket.close()















###############################################################################
        
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
##################### ACTUAL STARTING OF THE STUFF#############################
###############################################################################

class Containers():
    def __init__(self):
        self.IExist = True
        self.KeepRunning = True
        self.receiverthreads = []
        self.senderthreads = []
        
        
def create_socket(port):
    server_socket = MySocket()
    server_socket.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(TCP_IP, port)
    server_socket.listen(1)
    return server_socket

    

   
        
        
            
#class SenderListenerThread(threading.Thread):
#    def __init__(self):
#        threading.Thread.__init__(self)
#        self.containers = None
#        
#    def run(self):
#        while self.containers.KeepRunning:
#            #print("sender connected")
#            try:
#                (client, addr) = self.containers.senderportsocket.sock.accept()
#                clt = MySocket(client)
#                ct = sender_thread(clt)
#                ct.containers = self.containers
#                ct.start()
#            except socket.timeout:
#                #simply restart and don't care
#                pass

    
def main(conf):
    containers = Containers()
    containers.inputval = InputValContainer(conf)
    containers.inputval.containers = containers #lol.    
    containers.outputval = OutputValContainer()
    containers.outputval.containers = containers
    
    
    containers.receiverportsocket = create_socket(TCP_RECEIVER_PORT)
    containers.senderportsocket = create_socket(TCP_SENDER_PORT)
    
    containers.ANN = NeuralNetwork()
    containers.ANN.containers = containers
    
    print("Everything initialized")
    
    #THREAD 1
    ReceiverConnecterThread = ReceiverListenerThread()
    ReceiverConnecterThread.containers = containers
    ReceiverConnecterThread.start()
    
    #THREAD 2
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
    ReceiverConnecterThread.join() #takes max. 1 second until socket timeouts
#    SenderConnecterThread.join()
    print("Server shut down sucessfully.")


if __name__ == '__main__':  
    conf = supervisedcnn.Config() #TODO: lass dir die infos instead von unity schicken.
    main(conf)
    