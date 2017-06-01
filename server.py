#!/usr/bin/env python

import socket
import threading
import time
import logging
import numpy as np
import sys
from collections import deque

#====own classes====
import playnet
import reinf_net
import cnn
from myprint import myprint as print
import infoscreen
current_milli_time = lambda: int(round(time.time() * 1000))

logging.basicConfig(level=logging.ERROR, format='(%(threadName)-10s) %(message)s',)

MININT = -sys.maxsize+1
TCP_IP = 'localhost'
TCP_RECEIVER_PORT = 6435
TCP_SENDER_PORT = 6436
NUMBER_ANNS = 1
UPDATE_ONLY_IF_NEW = False #sendet immer nach jedem update -> Wenn False sendet er wann immer er was kriegt

wrongdirtime = 0



class MySocket:

    def __init__(self, sock=None):
        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock
        self.sock.settimeout(2.0)
            
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
                    if self.handle_special_commands(data):
                        continue
                    elif data[:5] == "Time(":
                        self.timestamp = float(data[5:data.find(")")])
                        for i in self.containers.receiverthreads:
                            if int(i.timestamp) < int(self.timestamp):
                                i.killme = True
                    
                        #print(data)
                        visionvec, allOneDs = cutoutandreturnvectors(data) 
                        self.containers.inputval.update(visionvec, allOneDs, self.timestamp) #we MUST have the inputval, otherwise there wouldn't be the possibility for historyframes.           
                        
                        if len(self.containers.ANNs) == 1:
                            self.containers.ANNs[0].runANN(UPDATE_ONLY_IF_NEW)
                        else:                                                       
                            thread = threading.Thread(target=self.runOneANN, args=()) #immediately returns if UPDATE_ONLY_IF_NEW and alreadyreadthread = threading.Thread(target=self.runANN_SaveResult, args=())
                            thread.start() 
                        
                        
                        
                        
                    
            except TimeoutError:
                if len(self.containers.receiverthreads) < 2:
                    pass
                else:
                    break
                
        self.containers.receiverthreads.remove(self)
        print("stopping receiver_thread")
        
        
    def handle_special_commands(self, data):
        specialcommand = False
        if data[:5] == "Time(":
            data = data[data.find(")")+1:]
        
        if data[:11] == "resetServer":
            resetServer(self.containers, data[11:])
            specialcommand = True    
        if data[:7] == "wallhit":
            endEpisode(self.containers)
            specialcommand = True    
        return specialcommand
    
    
    def runOneANN(self):
        for currANN in self.containers.ANNs:
            if not currANN.isbusy:
                currANN.runANN(UPDATE_ONLY_IF_NEW)
                break
            

def resetUnity(containers, punish=0):
    containers.outputval.send_via_senderthread("pleasereset", containers.inputval.timestamp)
    if punish:
        punishLastAction(containers, punish)
    resetServer(containers, containers.inputval.msperframe)
    

def resetServer(containers, mspersec):
    endEpisode(containers)
    containers.outputval.reset()
    containers.inputval.reset(mspersec, nolock = True)
    

def endEpisode(containers):
    #bei actions, nach denen resettet wurde, soll er den folgestate nicht mehr beachten (später gucken wenn reset=true dann setze Q_DECAY auf quasi 100%)
    if not containers.play_only:
        lastmemoryentry = containers.memory.pop() #oldstate, action, reward, newstate
        if lastmemoryentry is not None:
            lastmemoryentry[4] = True
            containers.memory.append(lastmemoryentry)
        
        
def punishLastAction(containers, howmuch):
    if not containers.play_only:
        lastmemoryentry = containers.memory.pop() #oldstate, action, reward, newstate
        if lastmemoryentry is not None:
            lastmemoryentry[2] -= abs(howmuch)
            containers.memory.append(lastmemoryentry)    

        
        
class InputValContainer(object):   
    
    def __init__(self, config):
        self.lock = threading.Lock()
        self.config = config
        self.visionvec = np.zeros([config.image_dims[0], config.image_dims[1]])
        if self.config.history_frame_nr > 1:
            self.vvec_hist = np.zeros([config.history_frame_nr, config.image_dims[0], config.image_dims[1]]) 
        self.othervecs = np.zeros(config.vector_len)      
        self.timestamp = MININT
        self.containers = None
        self.alreadyread = True
        self.previous_action = None
        self.previous_visionvec = None
        self.previous_vvechist = None
        self.previous_othervecs = None
        self.msperframe = config.msperframe
        self.hit_a_wall = False
        
        
    def update(self, visionvec, othervecs, timestamp):
        global wrongdirtime
        
        def is_new(visionvec, othervecs):
            if self.config.history_frame_nr == 1:
                return not (np.all(self.visionvec == visionvec) and np.all(self.othervecs == othervecs))
            else:
                tmp_vvec_hist = [visionvec] + [i for i in self.vvec_hist[:-1]]
                allequal = True
                for i in range(len(tmp_vvec_hist)):
                    if not np.all(self.vvec_hist[i] == tmp_vvec_hist[i]):
                        allequal = False
                return not (allequal and np.all(self.othervecs == othervecs))
        
        logging.debug('Inputval-Update: Waiting for lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired lock')
            if is_new(visionvec, othervecs):
            
                self.visionvec = visionvec
                if self.config.history_frame_nr > 1:
                    self.vvec_hist = [visionvec] + [i for i in self.vvec_hist[:-1]]            
                self.othervecs = othervecs
                
                #wenn othervecs[3][0] >= 10 war und seitdem keine neue action kam, muss er >= 10 bleiben!
                if othervecs[3][0] >= 10:
                    self.hit_a_wall = True 
                if self.hit_a_wall:
                    self.othervecs[3][0] = 10
                              
                try:
                    if self.config.reset_if_wrongdirection:
                        if not self.othervecs[1][5]:
                            wrongdirtime += self.containers.sv_conf.msperframe
                            if wrongdirtime >= 2000:
                                resetUnity(self.containers, punish=100)
                        else:
                            wrongdirtime = 0
                except IndexError:
                    wrongdirtime = 0
                                  
                self.alreadyread = False
                self.timestamp = timestamp
                print("Updated Input-Vec at", timestamp, level=2)
            else:
                print("No Input-Vec upgrading needed at", timestamp, level=2)
        finally:
            self.lock.release()
            
    def addResultAndBackup(self, action):
        self.hit_a_wall = False #sobald ne action danach kommt ist es unrelated #TODO: gilt nur wenn walhit_means_reset
        self.previous_action = action
        self.previous_othervecs = self.othervecs
        if self.config.history_frame_nr > 1:
            self.previous_vvechist = self.vvec_hist
        else:
            self.previous_visionvec = self.visionvec
            
        
    def get_previous_state(self):
        if self.previous_action is None or self.previous_othervecs is None:
            return None, False
        try:
            self.previous_othervecs[0][0]
        except IndexError:
            return None, False
        
        if self.config.history_frame_nr > 1:
            state = (self.previous_vvechist, self.previous_othervecs[1][4]) #vision plus speed
        else:
            state = (self.previous_visionvec, self.previous_othervecs[1][4]) #vision plus speed
        action = self.previous_action
            
        return state, action


        
    def reset(self, interval, nolock = False):
        logging.debug('Inputval-Reset: Waiting for lock')
        if not nolock: 
            self.lock.acquire()
        try:
            logging.debug('Acquired lock')
            
            self.visionvec = np.zeros([self.config.image_dims[0], self.config.image_dims[1]])
            if self.config.history_frame_nr > 1:
                self.vvec_hist = np.zeros([self.config.history_frame_nr, self.config.image_dims[0], self.config.image_dims[1]]) 
            self.othervecs = np.zeros(self.config.vector_len)      
            self.timestamp = 0
            self.alreadyread = True
            self.previous_action = None
            self.previous_visionvec = None
            self.previous_vvechist = None
            self.previous_othervecs = None
            self.msperframe = interval #da Unity im diesen Wert immer bei spielstart schickt, wird msperframe immer richtig sein            
            assert int(self.msperframe) == int(self.config.msperframe)
            self.hit_a_wall = False
            
            logging.debug("Resettet input-value")
        finally:
            if not nolock:
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
            if (UPDATE_ONLY_IF_NEW and int(self.timestamp) < int(itstimestamp)) or (not UPDATE_ONLY_IF_NEW and int(self.timestamp) <= int(itstimestamp)):
                self.value = withwhatval
                self.timestamp = itstimestamp #es geht nicht um jetzt, sondern um dann als das ANN gestartet wurde
                #self.alreadysent = False
                print("Updated output-value to",withwhatval)
                self.send_via_senderthread(self.value, self.timestamp)
            else:
                print("Didn't update output-value because the new one wouldn't be newer")
        finally:
            self.lock.release()

    def reset(self):
        logging.debug('Outputval-reset: Waiting for lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired lock')
            self.value = ""
            self.timestamp = MININT
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

###############################################################################

      
#wenn ich hier thread-locks verwenden würde würde er jedes mal einen neuen receiver-thread starten. 
#TODO: auf richtigere weise thread-safe machen.
class Memory(object):
    def __init__(self, elemtype, size):
#        self.lock = threading.Lock()
        self.memory = deque(elemtype, size)
    
    def append(self, obj):
        self.memory.append(obj)
        
    def pop(self):
        try:
            return self.memory.pop()        
        except:
            return None
#        self.lock.acquire()
#        try:
#            self.memory.append(obj)
#        finally:
#            self.lock.release()
        


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
        print("Sending", tosend, level=3)
        self.clientsocket.mysend(tosend)















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
    def __init__(self, play_only):
        self.play_only = play_only
        self.IExist = True
        self.KeepRunning = True
        self.receiverthreads = []
        self.senderthreads = []
        self.ANNs = []
        self.reinfNetSteps = 0
        
        
def create_socket(port):
    server_socket = MySocket()
    server_socket.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(TCP_IP, port)
    server_socket.listen(1)
    return server_socket



def main(sv_conf, rl_conf, play_only, no_learn, show_screen, start_fresh):
    containers = Containers(play_only)
    containers.inputval = InputValContainer(sv_conf)
    containers.inputval.containers = containers #lol.    
    containers.outputval = OutputValContainer()
    containers.outputval.containers = containers
    containers.sv_conf = sv_conf
    containers.rl_conf = rl_conf
    
    containers.receiverportsocket = create_socket(TCP_RECEIVER_PORT)
    containers.senderportsocket = create_socket(TCP_SENDER_PORT)
    
    if show_screen:
        screenroot = infoscreen.showScreen(containers)
    
    if play_only:
        NeuralNet = playnet.PlayNet
    else:
        NeuralNet = reinf_net.ReinfNet
        containers.memory = Memory([], reinf_net.MEMORY_SIZE)
        
    for i in range(NUMBER_ANNS):
        ANN = NeuralNet(i, sv_conf, containers, rl_conf, start_fresh)
        containers.ANNs.append(ANN)
    
    print("Everything initialized", level=10)
    
    #THREAD 1
    ReceiverConnecterThread = ReceiverListenerThread()
    ReceiverConnecterThread.containers = containers
    ReceiverConnecterThread.start()
    
    #THREAD 2
    SenderConnecterThread = SenderListenerThread()
    SenderConnecterThread.containers = containers
    SenderConnecterThread.start()

    #THREAD 3 (learning)
    if not play_only and not no_learn:
        learnthread = threading.Thread(target=containers.ANNs[0].dauerLearnANN)
        learnthread.start()
    
   
    try:      
        if show_screen:
            screenroot.mainloop()            
        else:
            while True:
                pass
    except KeyboardInterrupt:
        pass
    
    print("Server shutting down...")
    containers.KeepRunning = False
    for senderthread in containers.senderthreads:
        senderthread.delete_me() 
    ReceiverConnecterThread.join() #takes max. 1 second until socket timeouts
    SenderConnecterThread.join()
    if not play_only and not no_learn:
        learnthread.join()
    time.sleep(0.1)
    print("Server shut down sucessfully.")
    
    


    
if __name__ == '__main__':  
    sv_conf = cnn.Config() #TODO: lass dir die infos instead von unity schicken.
    rl_conf = cnn.RL_Config()
    
    if "-nolearn" in sys.argv:
        reinf_net.minepsilon = 0
        reinf_net.epsilon = 0
                
    main(sv_conf, rl_conf, ("-playonly" in sys.argv), ("-nolearn" in sys.argv), not ("-noscreen" in sys.argv), ("-startfresh" in sys.argv))    