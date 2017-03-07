#!/usr/bin/env python

import socket
import threading
import time
import logging

logging.basicConfig(level=logging.ERROR,
                    format='(%(threadName)-10s) %(message)s',
                    )


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
        self.value = ""
        self.timestamp = current_milli_time()
        self.alreadyread = False
        
    def update(self, withwhat):
        logging.debug('Waiting for lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired lock')
            if self.value != withwhat:
                self.value = withwhat
                self.timestamp = current_milli_time()
                self.alreadyread = False
                print("Updated value to",withwhat)
        finally:
            self.lock.release()

    def read(self):
        self.alreadyread = True
        return self.value
    


class NeuralNetworkPretenderThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        
    def run(self):
        if not inputval.alreadyread:   
            print("Another ANN Starts")                   
            returnstuff = self.performNetwork(inputval)
            outputval.update(returnstuff, inputval.timestamp)    
        

    def performNetwork(self, inputval): 
        inputdata = inputval.read()
        
        time.sleep(ANNTAKESTIME) #this is where the ann would come
        if inputdata:
            if inputdata[0] > 30:
                return "pleasereset"
            if (inputdata[3]) != 0:
                return "turning"
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
        logging.debug('Waiting for lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired lock')
            if self.timestamp < itstimestamp:
                self.value = withwhatval
                self.timestamp = current_milli_time()
                self.alreadysent = False
                print("Updated value to",withwhatval)
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
            #time.sleep(200)
            #print("received data:", data)
            tmpstrings = data.split(" ")
            tmpfloats = []
            for i in tmpstrings:
                tmp = i.replace(" ","")
                if len(tmp) > 0:
                    try:
                        x = float(str(tmp))
                        tmpfloats.append(x)
                    except ValueError:
                        print(tmp)
            inputval.update(tmpfloats)
        self.clientsocket.close()
        
        
        
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



#class sendandreceive_client_thread(threading.Thread):
#    def __init__(self, clientsocket):
#        threading.Thread.__init__(self)
#        self.clientsocket = clientsocket
#    def run(self):
#        #print("Starting Thread")
#        data = self.clientsocket.myreceive()
#        if data: 
#            #time.sleep(200)
#            #print("received data:", data)
#            tmpstrings = data.split(" ")
#            tmpfloats = []
#            for i in tmpstrings:
#                tmp = i.replace(" ","")
#                if len(tmp) > 0:
#                    x = float(str(tmp))
#                    tmpfloats.append(x)
#            val.update(tmpfloats)
#            if (tmpfloats[2]) != 0:
#                self.clientsocket.mysend("turning")
#            else:
#                self.clientsocket.mysend("answer: "+data)  #RETURN SOME KIND OF DATA
#        self.clientsocket.close()


#-------------------------------------------------------



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




#now I need too threads, one constantly looking for clients, the other constantly running a new ANN-thread...

           
    
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



#class SenderListenerThread(threading.Thread):
#    def __init__(self):
#        threading.Thread.__init__(self)
#        
#    def run(self):
#        while True:
#            #print "sender connected"
#            (client, addr) = s.sock.accept()
#            clt = MySocket(client)
#            ct = sender_thread(clt)
#            ct.start()





ANNStarterThread = NeuralNetStarterThread()
ANNStarterThread.start()

ReceiverConnecterThread = ReceiverListenerThread()
ReceiverConnecterThread.start()


while True:
    #print "sender  connected"
    (client, addr) = senderportsocket.sock.accept()
    clt = MySocket(client)
    ct = sender_thread(clt)
    ct.start()
            

