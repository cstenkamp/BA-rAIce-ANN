#!/usr/bin/env python
#SERVER ist UNABHÄNGIG VOM AGENT, und sorgt dafür dass die passenden Dinge an den agent weitergeleitet werden!


import socket
import threading
import time
import logging
import numpy as np
import sys
from collections import namedtuple
import copy

#====own classes====
import svPlayNetAgent
import reinfNetAgent #is an agent, inherits all agent's functions
import cnn
from myprint import myprint as print
import infoscreen
from read_supervised import cutoutandreturnvectors
from precisememory import Memory as Precisememory
from efficientmemory import Memory as Efficientmemory


current_milli_time = lambda: int(round(time.time() * 1000))
logging.basicConfig(level=logging.ERROR, format='(%(threadName)-10s) %(message)s',) #legacy


#this very long part is the comparable namedtuple otherinputs!
prespeedsteer = namedtuple('SpeedSteer', ['RLTorque', 'RRTorque', 'FLSteer', 'FRSteer', 'velocity', 'rightDirection'])
prestatusvector = namedtuple('StatusVector', ['velocity', 'FLSlip0', 'FRSlip0', 'RLSlip0', 'RRSlip0', 'FLSlip1', 'FRSlip1', 'RLSlip1', 'RRSlip1'])
                                             #1 elem     6 elems       9 elems         1 elem        15 elems         30 elems = 62 elems
preotherinputs = namedtuple('OtherInputs', ['progress', 'SpeedSteer', 'StatusVector', 'CenterDist', 'CenterDistVec', 'LookAheadVec'])
class speedsteer(prespeedsteer):
    def __eq__(self, other):
        return np.all([self[i] == other[i] for i in range(len(self))])
class statusvector(prestatusvector):
    def __eq__(self, other):
        return np.all([self[i] == other[i] for i in range(len(self))])
class otherinputs(preotherinputs):
    def __eq__(self, other):
        return self.progress == other.progress \
           and self.SpeedSteer ==  other.SpeedSteer \
           and self.StatusVector == other.StatusVector \
           and self.CenterDist == other.CenterDist \
           and np.all(self.LookAheadVec == other.LookAheadVec)
           #and np.all(self.CenterDistVec == other.CenterDistVec) \ #can be skipped because then the centerdist is also equal
           
empty_speedsteer = lambda: speedsteer(0, 0, 0, 0, 0, 0)
empty_statusvector = lambda: statusvector(0, 0, 0, 0, 0, 0, 0, 0, 0)
empty_inputs = lambda: otherinputs(0, empty_speedsteer(), empty_statusvector(), 0, np.zeros(15), np.zeros(30))
def make_otherinputs(othervecs):
    return otherinputs(othervecs[0][0], \
                       speedsteer(othervecs[1][0], othervecs[1][1], othervecs[1][2], othervecs[1][3], othervecs[1][4], othervecs[1][5]), \
                       statusvector(othervecs[2][0], othervecs[2][1], othervecs[2][2], othervecs[2][3], othervecs[2][4], othervecs[2][5], othervecs[2][6], othervecs[2][7], othervecs[2][8]), \
                       othervecs[3][0], \
                       othervecs[3][1:], \
                       othervecs[4])
#this very long part end


MININT = -sys.maxsize+1
TCP_IP = 'localhost' #TODO check if it also works over internet, in the CLuster
TCP_RECEIVER_PORT = 6435
TCP_SENDER_PORT = 6436
NUMBER_ANNS = 1 #only one of those will execute the learning, in dauerLearnANN in LearnThread
UPDATE_ONLY_IF_NEW = False #sendet immer nach jedem update -> Wenn False sendet er wann immer er was kriegt
SAVE_MEMORY_ON_EXIT = True


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
# handling everything from there on. If unity wants to reconnect for any reason, it will create a new receiver_thread
# which kills all old ones, such that there is only one receiver_thread active most of the time.
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
                if not self.containers.freezeInf:
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
                            
                            #CHANGE: only 1 agent 
    #                        if len(self.containers.ANNs) == 1:
    #                            self.containers.ANNs[0].runInference(UPDATE_ONLY_IF_NEW)
    #                        else:                                                       
    #                            thread = threading.Thread(target=self.runOneANN, args=()) #immediately returns if UPDATE_ONLY_IF_NEW and alreadyreadthread = threading.Thread(target=self.runInference_SaveResult, args=())
    #                            thread.start() 
                                #this thread here will also  send the result afterwards
                                
                            self.containers.myAgent.runInference(UPDATE_ONLY_IF_NEW)
                        
            except TimeoutError:
                if len(self.containers.receiverthreads) < 2:
                    pass
                else:
                    break
                
        self.containers.receiverthreads.remove(self)
        print("stopping receiver_thread")
        #thread.exit() only works with the thread object, but not with threading.Thread class object.
        
        
    def handle_special_commands(self, data):
        specialcommand = False
        if data[:5] == "Time(":
            data = data[data.find(")")+1:]
        
        if data[:11] == "resetServer":
            self.containers.myAgent.endEpisode()
            resetServer(self.containers, data[11:]) 
            specialcommand = True    
        if data[:7] == "wallhit":
            self.containers.myAgent.punishLastAction(20)   #ist das doppelt gemoppelt damit, dass er eh das if punish > 10 beibehält?       
            self.containers.myAgent.endEpisode()
            resetServer(self.containers, self.containers.sv_conf.msperframe) 
            specialcommand = True    
        return specialcommand
    
    #CHANGE: only 1 agent
#    #this will be run in a separate thread
#    def runOneANN(self):
#        for currANN in self.containers.ANNs:
#            if not currANN.isbusy:
#                currANN.runInference(UPDATE_ONLY_IF_NEW)
#                break
            

###############################################################################

def resetUnity(containers, punish=0):
    containers.outputval.send_via_senderthread("pleasereset", containers.inputval.timestamp)
    resetServer(containers, containers.inputval.msperframe, punish)
    

def resetServer(containers, mspersec, punish=0):
    containers.outputval.reset()
    containers.inputval.reset(mspersec, nolock = True)


def freezeUnity(containers):
    containers.outputval.send_via_senderthread("pleaseFreeze", containers.inputval.timestamp)

def unFreezeUnity(containers):
    containers.outputval.send_via_senderthread("pleaseUnFreeze", containers.inputval.timestamp)
    
   
###############################################################################
        
class InputValContainer(object):   
    
    def __init__(self, config):
        self.lock = threading.Lock()
        self.config = config
        self.visionvec = np.zeros([config.image_dims[0], config.image_dims[1]])
        if self.config.history_frame_nr > 1:
            self.vvec_hist = np.zeros([config.history_frame_nr, config.image_dims[0], config.image_dims[1]]) 
        self.otherinputs = empty_inputs() #defined at the top, is a namedtuple
        self.timestamp = MININT
        self.containers = None
        self.alreadyread = True
        self.previous_action = None
        self.previous_visionvec = None
        self.previous_vvechist = None
        self.previous_otherinputs = None
        self.msperframe = config.msperframe
        self.hit_a_wall = False
        
        
    def update(self, visionvec, othervecs, timestamp):
        
        def is_new(visionvec, otherinputs):
            if self.config.history_frame_nr == 1:
                return not (np.all(self.visionvec == visionvec) and self.otherinputs == otherinputs)
            else:
                tmp_vvec_hist = [visionvec] + [i for i in self.vvec_hist[:-1]]
                allequal = True
                for i in range(len(tmp_vvec_hist)):
                    if not np.all(self.vvec_hist[i] == tmp_vvec_hist[i]):
                        allequal = False
                return not (allequal and np.all(self.otherinputs == otherinputs))
        
        logging.debug('Inputval-Update: Waiting for lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired lock')
            otherinputs = make_otherinputs(othervecs) #is now a namedtuple instead of an array
            if is_new(visionvec, otherinputs):
            
                self.visionvec = visionvec
                if self.config.history_frame_nr > 1:
                    self.vvec_hist = [visionvec] + [i for i in self.vvec_hist[:-1]]            
                self.otherinputs = otherinputs
                
                #wenn otherinputs.CenterDist >= 10 war und seitdem keine neue action kam, muss er >= 10 bleiben!
                if otherinputs.CenterDist >= 10:
                    self.hit_a_wall = True 
                #wird erst sobald ne action kommt wieder false gesetzt.. und solange es true ist:
                if self.hit_a_wall:
                    self.otherinputs = self.otherinputs._replace(CenterDist = 10)
                    
                try:
                    if self.config.reset_if_wrongdirection:
                        if not self.otherinputs.SpeedSteer.rightDirection:
                            self.containers.wrongdirectiontime += self.containers.sv_conf.msperframe
                            if self.containers.wrongdirectiontime >= 2000:
                                resetUnity(self.containers, punish=100)
                        else:
                            self.containers.wrongdirectiontime = 0
                except IndexError:
                    self.containers.wrongdirectiontime = 0
                                  
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
        self.previous_otherinputs = copy.deepcopy(self.otherinputs) 
        if self.config.history_frame_nr > 1:
            self.previous_vvechist = copy.deepcopy(self.vvec_hist)
        else:
            self.previous_visionvec = copy.deepcopy(self.visionvec)
            
        
    def get_previous_state(self):
        if self.previous_action is None or self.previous_otherinputs is None:
            return None, False
        
        if self.config.history_frame_nr > 1:
            state = (self.previous_vvechist, self.previous_otherinputs.SpeedSteer.velocity) #vision plus speed
        else:
            state = (self.previous_visionvec, self.previous_otherinputs.SpeedSteer.velocity) #vision plus speed
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
            self.otherinputs = empty_inputs()
            self.timestamp = 0
            self.alreadyread = True
            self.previous_action = None
            self.previous_visionvec = None
            self.previous_vvechist = None
            self.previous_otherinputs = None
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
            return self.otherinputs, self.vvec_hist
        else:
            return self.otherinputs, self.visionvec
        
        
    
    
        

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
        if self.containers.KeepRunning:
            assert len(self.containers.senderthreads) > 0, "There is no senderthread at all! How will I send?"
            for i in range(len(self.containers.senderthreads)):
                try:
                    self.containers.senderthreads[i].send(value, timestamp)
                except (ConnectionResetError, ConnectionAbortedError):
                        #if unity restarted, the old connection is now useless and should be deleted
                        print("I assume you just restarted Unity.")
                        self.containers.senderthreads[i].delete_me()
                        self.containers.senderthreads[i].join()
                        if i >= len(self.containers.senderthreads)-1:
                            break


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
        #ist er jetzt wirklich ganz weg?
        
        
        
    def send(self, result, timestamp):
        tosend = str(result) + "Time(" +str(timestamp)+")"
        print("Sending", tosend, level=3)
        self.clientsocket.mysend(tosend)









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
        #self.ANNs = []
        self.myAgent = None
        #self.reinfNetSteps = 0
        self.wrongdirectiontime = 0
        self.freezeInf = self.freezeLearn = False
        #numIterations steckt in self.myAgent.numIterations
        
def create_socket(port):
    server_socket = MySocket()
    server_socket.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(TCP_IP, port)
    server_socket.listen(1)
    return server_socket



def main(sv_conf, rl_conf, only_sv, no_learn, show_screen, start_fresh, keep_memory):
    containers = Containers(only_sv)
    containers.inputval = InputValContainer(sv_conf)
    containers.inputval.containers = containers #lol.    
    containers.outputval = OutputValContainer()
    containers.outputval.containers = containers
    containers.sv_conf = sv_conf
    containers.rl_conf = rl_conf   
    containers.keep_memory = keep_memory #bei DQN und der keepmememory-variante speichert er das memory als pickle-file for later use
    
    containers.receiverportsocket = create_socket(TCP_RECEIVER_PORT)
    containers.senderportsocket = create_socket(TCP_SENDER_PORT)
    
    if show_screen:
        screenroot = infoscreen.showScreen(containers)
    
    if only_sv:
        agent = svPlayNetAgent.PlayNetAgent
    else:
        agent = reinfNetAgent.ReinfNetAgent #this one, inheriting from abstractRLAgent, will have a memory
        
    #CHANGE: only one agent. that agent can itself run the runInference in separate threads, if need be.
#    for i in range(NUMBER_ANNS): #only one of those will execute the learning, in dauerLearnANN in LearnThread
#        ANN = agent(i, sv_conf, containers, rl_conf, start_fresh)
#        containers.ANNs.append(ANN)
    
    containers.myAgent = agent(sv_conf, containers, rl_conf, start_fresh) #executes dauerLearnANN in LearnThread
                                                                          #executes runInference in receiver_thread
#    if not only_sv:
#        if rl_conf.useprecisebuthugememory:
#            containers.myAgent.memory = Precisememory(rl_conf.memorysize, containers)
#        else:
#            containers.myAgent.memory = Efficientmemory(rl_conf.memorysize, containers, rl_conf.history_frame_nr) 
    
    containers.myAgent.memory = Precisememory(rl_conf.memorysize, containers)
    containers.myAgent.memory2 = Efficientmemory(rl_conf.memorysize, containers, rl_conf.history_frame_nr) 

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
    if not only_sv and not no_learn:
        learnthread = threading.Thread(target=containers.myAgent.dauerLearnANN) #TODO: das hier geht nicht bei > 1 ANN #BUG
        learnthread.start()
    
   #THREAD 4 (self/GUI)
    try:      
        if show_screen:
            screenroot.mainloop()            
        else:
            while True:
                pass
    except KeyboardInterrupt:
        pass
    
    #AFTER KILLING:
        
    print("Server shutting down...")
    containers.KeepRunning = False
        
    for senderthread in containers.senderthreads:
        senderthread.delete_me() 
        senderthread.join()
    ReceiverConnecterThread.join() #takes max. 1 second until socket timeouts
    SenderConnecterThread.join()
    if not only_sv and not no_learn:
        learnthread.join()
        
    if SAVE_MEMORY_ON_EXIT:
        containers.myAgent.memory.save_memory()
        containers.myAgent.memory2.save_memory()
        
    time.sleep(0.1)
    print("Server shut down sucessfully.")
    
    


    
if __name__ == '__main__':  
    sv_conf = cnn.Config() #TODO: lass dir die infos instead von unity schicken.
    if ("-DQN" in sys.argv):
        rl_conf = cnn.DQN_Config()
    else:
        rl_conf = cnn.RL_Config()
    
    if "-nolearn" in sys.argv:
        rl_conf.minepsilon = 0
        rl_conf.startepsilon = 0 #or whatever random-value will be in 
        
                
    main(sv_conf, rl_conf, ("-svplay" in sys.argv), ("-nolearn" in sys.argv), not ("-noscreen" in sys.argv), ("-startfresh" in sys.argv), ("-keepmemory" in sys.argv or "-DQN" in sys.argv))    