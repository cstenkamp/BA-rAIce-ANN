#!/usr/bin/env python

import numpy as np
import socket
import threading
import time
import logging
import hashlib

logging.basicConfig(level=logging.ERROR,
                    format='(%(threadName)-10s) %(message)s',)


TCP_IP = 'localhost'

TCP_RECEIVER_PORT = 6435
TCP_SENDER_PORT = 6436
ANNCHECKALL = 100
ANNTAKESTIME = 1

current_milli_time = lambda: int(round(time.time() * 1000))


class MySocket:

    def __init__(self, sock=None):
        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock

            
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
        what = self.sock.recv(5).decode('ascii')
        if not what:
            return
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

  
        
#-------------------------------------------------------

# alle X sekunden kommt ein leser und liest den InputVal aus (immer der neueste, mit timestamp!!), und updated damit 
# den OutputValContainer, falls der nicht schon einen neueren inputval-timestamp hat.
class InputValContainer(object):   
    def __init__(self):
        self.lock = threading.Lock()
        self.visionvec = np.zeros([20,20]) #TODO - gucken ob die größe gleich ist/wie ich die größe share
        self.othervecs = np.zeros(50)    #TODO - same
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
    


class NeuralNetworkPretenderThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        
    def run(self):
        if not inputval.alreadyread:   
            print("Another ANN Starts")                   
            returnstuff = self.performNetwork(inputval)
            outputval.update(returnstuff, inputval.timestamp)    
        

    def performNetwork(self, inputval): 
        inputvec, visionvec = inputval.read()
        
        time.sleep(ANNTAKESTIME) #this is where the ann would come
        if inputvec:
            if inputvec[0][0] > 30:
                return "pleasereset"
#            if (inputvec[1][0]) != 0:
#                return "turning"
            else:
                return ("answer: something")  #RETURN SOME KIND OF DATA
        else:
            return





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

  

#-------------------------------------------------------

#the receiver-thread gets called when unity wants, and tries to update the 
#global inputval, containing the race-info, as often as possible.
class receiver_thread(threading.Thread):
    def __init__(self, clientsocket):
        threading.Thread.__init__(self)
        self.clientsocket = clientsocket
        
    def run(self):
        #print("Starting Thread")
        data = self.clientsocket.myreceive()
        if data: 
            #print("received data:", data)
            
            alltwods  = []
            visionvec = [[]]
            
            #TODO - ERST den hash vom string vergleichen damit das updaten schneller geht?
            
            def cutout(string, letter):
                return string[string.find(letter)+2:string[string.find(letter):].find(")")+string.find(letter)]
            
            if data.find("P(") > -1:
                #print("Progress as real Number",self.readOneDArrayFromString(cutout(data, "P(")))
                alltwods.append(self.readOneDArrayFromString(cutout(data, "P(")))

            if data.find("S(") > -1:
                #print("SpeedStearVec",self.readOneDArrayFromString(cutout(data, "S(")))
                alltwods.append(self.readOneDArrayFromString(cutout(data, "S(")))

                
            if data.find("T(") > -1:
                #print("CarStatusVec",self.readOneDArrayFromString(cutout(data, "T(")))
                alltwods.append(self.readOneDArrayFromString(cutout(data, "T(")))
                
            if data.find("C(") > -1:
                #print("Visionvec",self.readTwoDArrayFromString(cutout(data, "V(")))
                alltwods.append(self.readOneDArrayFromString(cutout(data, "C(")))
                
            if data.find("L(") > -1:
                #print("Visionvec",self.readTwoDArrayFromString(cutout(data, "V(")))
                alltwods.append(self.readOneDArrayFromString(cutout(data, "L(")))
            
            
            if data.find("V(") > -1:
                #print("Visionvec",self.readTwoDArrayFromString(cutout(data, "V(")))
                visionvec = self.readTwoDArrayFromString(cutout(data, "V("))

            
            hashco = (hashlib.md5(data.encode('utf-8'))).hexdigest()
            
            inputval.update(visionvec, alltwods, hashco)
        self.clientsocket.close()
        
        
    def readOneDArrayFromString(self, string):
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
    
    
    def ternary(self, n):
        if n == 0:
            return '0'
        nums = []
        if n < 0:
            n*=-1
        while n:
            n, r = divmod(n, 3)
            nums.append(str(r))
        return ''.join(reversed(nums))
    
    
        
#    def readTwoDArrayFromString(self, string):
#        tmpstrings = string.split(",")
#        tmpreturn = []
#        for i in tmpstrings:
#            tmp = i.replace(" ","")
#            if len(tmp) > 0:
#                try:
#                    x = self.ternary(int(tmp))
#                    tmpreturn.append(x)
#                    print(x)
#                except ValueError:
#                    print("I'm crying") #cry.
#        return np.array(tmpreturn)

    def readTwoDArrayFromString(self, string):
        tmpstrings = string.split(",")
        tmpreturn = []
        for i in tmpstrings:
            tmp = i.replace(" ","")
            if len(tmp) > 0:
                try:
                    currline = []
                    for j in tmp:
                        currline.append(j)
                    tmpreturn.append(currline)
                except ValueError:
                    print("I'm crying") #cry.
        return np.array(tmpreturn)
    
    
        
        
#the sender-thread gets called when unity asks for the result of a new network iteration...
#so python aks the ValContainer outputval if it the flag alreadysent is false, if so, it returns it, 
#if not, it simply closes the connection.        
class sender_thread(threading.Thread):
    def __init__(self, clientsocket):
        threading.Thread.__init__(self)
        self.clientsocket = clientsocket
        
    def run(self):
        #print("Starting Thread")
        print("Unity asked for the result")
        outputvaltxt = outputval.read() #false if already read.
        if outputvaltxt:
            print("Sending ANN result back, because Unity asked: ", outputvaltxt)
            self.clientsocket.mysend(outputvaltxt)
        else:
            self.clientsocket.close()



###############################################################################
##################### ACTUAL STARTING OF THE STUFF#############################
###############################################################################

def create_socket(port):
    server_socket = MySocket()
    server_socket.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(TCP_IP, port)
    server_socket.listen(1)
    return server_socket

inputval = InputValContainer()
outputval = OutputValContainer()

receiverportsocket = create_socket(TCP_RECEIVER_PORT)
senderportsocket = create_socket(TCP_SENDER_PORT)




#now I need three threads:
#   one constantly looking for clients sending information here
#   one constantly running a new ANN-thread...
#   one constantly looking for clients demanding information here

           
    
class NeuralNetStarterThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.laststart = current_milli_time()
        
    def run(self):
        while True:
            if current_milli_time() > self.laststart - ANNCHECKALL:
                annthread = NeuralNetworkPretenderThread()
                annthread.start()
                self.laststart = current_milli_time()



class ReceiverListenerThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        
    def run(self):
        while True:
            #print "receiver connected"
            (client, addr) = receiverportsocket.sock.accept()
            clt = MySocket(client)
            ct = receiver_thread(clt)
            ct.start()


#THREAD 1
ANNStarterThread = NeuralNetStarterThread()
ANNStarterThread.start()

#THREAD 2
ReceiverConnecterThread = ReceiverListenerThread()
ReceiverConnecterThread.start()

#THREAD 3 (self)
while True:
    #print "sender  connected"
    (client, addr) = senderportsocket.sock.accept()
    clt = MySocket(client)
    ct = sender_thread(clt)
    ct.start()