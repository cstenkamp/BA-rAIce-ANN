#!/usr/bin/env python

import socket
import threading
import time
import logging

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
        self.sock.settimeout(5.0)
            
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

#the receiver-thread gets called when unity wants, and tries to update the 
#global inputval, containing the race-info, as often as possible.
class receiver_thread(threading.Thread):
    def __init__(self, clientsocket):
        threading.Thread.__init__(self)
        self.clientsocket = clientsocket
        self.containers = None
        
    def run(self):
        print("Starting receiver_thread")
        while self.containers.KeepRunning:
            try:            
                data = self.clientsocket.myreceive()
                if data: 
                    print("received data:", data)       
            except TimeoutError:
                break
                
        print("stopping receiver_thread")
   
        


###############################################################################
    
class sender_thread(threading.Thread):
    def __init__(self, clientsocket):
        threading.Thread.__init__(self)
        self.clientsocket = clientsocket
        self.containers = None
        
    def run(self):
        print("Starting sender_thread")
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
                #simply restart and don't care
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
                #simply restart and don't care
                pass

    
def main():
    containers = Containers()
    
    containers.receiverportsocket = create_socket(TCP_RECEIVER_PORT)
    containers.senderportsocket = create_socket(TCP_SENDER_PORT)
    
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
    ReceiverConnecterThread.join() #takes max. 10 seconds until socket timeouts
    SenderConnecterThread.join()
    print("Server shut down sucessfully.")


if __name__ == '__main__':   
    main()
    