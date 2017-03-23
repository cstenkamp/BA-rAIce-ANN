#!/usr/bin/env python

import socket


TCP_IP = 'localhost'
TCP_PORT = 5005
MSGLENLONG = 4096
MSGLENSHORT = 6
MESSAGELONG = "Hallo das ist ein Test."
MESSAGESHORT = "OH MAN"
#-------------------------------------------------------

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
        length = str(len(msg))  # angeblich darf das nicht so: https://docs.python.org/3/howto/sockets.html
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
        what = self.sock.recv(5).decode()
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

s = MySocket()
s.connect(TCP_IP, TCP_PORT)


s.mysend(MESSAGELONG)
data = s.myreceive()
s.close()

print("received data:", data)