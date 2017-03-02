#!/usr/bin/env python

import socket
import threading
import time




TCP_IP = 'localhost'
TCP_PORT = 5005
MESSAGELONG = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
MESSAGESHORT = "OH MAN"

class MySocket:
    """demonstration class only
      - coded for clarity, not efficiency
    """

    def __init__(self, sock=None):
        if sock is None:
            self.sock = socket.socket(
                            socket.AF_INET, socket.SOCK_STREAM)
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

            
    #I think the speed would increase manyfold if I don't close the
    #connection right away but keep the sockets communicating
            
    def myreceive(self):
        chunks = []
        bytes_recd = 0
        what = self.sock.recv(5)
        if what[1] == " ":
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


class client_thread(threading.Thread):
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
                    x = float(str(tmp))
                    tmpfloats.append(x)
            print("received inputs:", tmpfloats)
            if (tmpfloats[2]) != 0:
                self.clientsocket.mysend("turning")
            else:
                self.clientsocket.mysend("answer: "+data)  #RETURN SOME KIND OF DATA
        self.clientsocket.close()



s = MySocket()
s.bind(TCP_IP, TCP_PORT)
s.listen(1)

#my solution here is having a seperate thread for each client...
#not sure if thats the best solution (-nonblocking zB.)
amount = 0;
while True:
    (client, addr) = s.sock.accept()
    #print('Connection address:', addr)
    clt = MySocket(client)
    ct = client_thread(clt)
    ct.start()