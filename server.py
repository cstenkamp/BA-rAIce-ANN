#!/usr/bin/env python
#SERVER ist UNABHÄNGIG VOM AGENT, und sorgt dafür dass die passenden Dinge an den agent weitergeleitet werden!


import socket
import threading
import time
import logging
import numpy as np
import sys
from functools import partial
import copy

#====own classes====
from myprint import myprint as print
from read_supervised import empty_inputs, make_otherinputs 
import infoscreen
import config
from read_supervised import cutoutandreturnvectors
#die Agents werden unten nach bedarf imported, da sie abhängig von sys-arguments sind


import warnings
current_milli_time = lambda: int(round(time.time() * 1000))
if current_milli_time() < 1508101200000: warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR, format='(%(threadName)-10s) %(message)s',) #legacy

MININT = -sys.maxsize+1
TCP_IP = 'localhost' #TODO check if it also works over internet, in the Cluster
TCP_RECEIVER_PORT = 6435
TCP_SENDER_PORT = 6436


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
# which kills all old ones, such that there is only one receiver_thread active almost all the time.
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
                ct = receiver_thread(clt, self.containers)
                ct.start()
                self.containers.receiverthreads.append(ct)
            except socket.timeout:
                #simply restart and don't care
                pass   


# A receiver-thread represents a stable one-sided connection to unity. It runs constantly, getting the newest information from unity constantly.
# If unity disconnects, it stops. If any instance of this finds receiver_threads with older data, it will deem the thread deprecated 
# and kill it. The receiver_thread updates the global inputval (which is certainly only one!), containing the race-info, as soon as it gets new info.
class receiver_thread(threading.Thread):
    def __init__(self, clientsocket, containers):
        threading.Thread.__init__(self)
        self.clientsocket = clientsocket
        self.containers = containers
        self.containers.UnityConnected = True
        self.killme = False
        self.CTimestamp, self.STimestamp = MININT, MININT
        
    def run(self):
        print("Starting receiver_thread")
        while self.containers.KeepRunning and (not self.killme):
            try:        
                if not self.containers.freezeInf:
                    data = self.clientsocket.myreceive()
                    if data: 
                        #print("received data:", data, level=10)   
                        
                        if self.handle_special_commands(copy.deepcopy(data)):
                            continue
                        elif data[:6] == "STime(":
                        
                            #we MUST have the inputval, otherwise there wouldn't be the possibility for historyframes.           
                            STime, CTime, visionvec, vvec2, allOneDs = cutoutandreturnvectors(data) 
                            self.CTimestamp, self.STimestamp = CTime, STime
                            for i in self.containers.receiverthreads:
                                if int(i.STimestamp) < int(self.STimestamp):
                                    i.killme = True
                                    
                            print("PYTHON RECEIVES TIME:", STime, time.time()*1000, level=4)
                            self.containers.inputval.update(visionvec, vvec2, allOneDs, STime, CTime)  #note that visionvec and vvec2 can both be None                                                           
                            self.containers.myAgent.performAction(self.containers.inputval.read(), self.containers.inputval.read(pastState=True))
                        
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
        if data[:6] == "STime(":
            data = data[data.find(")")+1:]
    
        if data[:11] == "resetServer":
            if hasattr(self.containers.myAgent, "memory"):
                self.containers.myAgent.endEpisode("resetserver", self.containers.inputval.read())
            resetServer(self.containers, data[11:]) 
            specialcommand = True    
        if data[:7] == "wallhit":
            if hasattr(self.containers.myAgent, "memory"):
                self.containers.myAgent.punishLastAction(self.containers.myAgent.wallhitPunish)   #ist das doppelt gemoppelt damit, dass er eh das if punish > 10 beibehält?       
                if self.containers.conf.wallhit_ends_episode:
                    self.containers.myAgent.endEpisode("wallhit", self.containers.inputval.read())
            #resetServer(self.containers, self.containers.conf.msperframe) 
            specialcommand = True 
        if data[:8] == "endround":
            print("Round finished", level=6)
            #roundwasvalid = data[8]==1 #benötige ich momentan noch nicht
            if hasattr(self.containers.myAgent, "memory") and self.containers.conf.lapdone_ends_episode:
                self.containers.myAgent.endEpisode("lapdone", self.containers.inputval.read())
            specialcommand = True        
        return specialcommand
    

###############################################################################

def resetUnityAndServer(containers, punish=0):
    containers.outputval.send_via_senderthread("pleasereset", containers.inputval.CTimestamp, containers.inputval.STimestamp)
    resetServer(containers, containers.inputval.msperframe, punish)
    

def resetServer(containers, mspersec, punish=0):
    containers.outputval.reset()
    containers.inputval.reset(mspersec, nolock = True)

   
###############################################################################
        
#the InputValContainer contains the vectors (visionvec, visionvec2, othervecs) Unity send to python. To decrease the amount of what Unity has to send to python, 
#it always only sends the newest vectors. It is pythons job to keep track of the history-frames, in case the structure of the used network requires them.
#There will definitely only be one object of the inputvalcontainer-class, and every receiver-thread appends the newest vectors to this object. In case of multiple
#Agents, the speed of the threadsafe inputvalcontainer will be the bottleneck.
#In RL, there is the difference between the state of the environment and the state of the agent. The inputval will be the thing in between: It represents the maximum
#of a state the agent could know (historyframes times 2visionvecs + othervecs). The agent will then decide what from those it will use as its state, 
#for example visionvecs of the last 4 frames plus speed of the last frame.
class InputValContainer(object):   
        
    def __init__(self, conf, containers, agent):
        self.lock = threading.Lock()
        self.conf = conf
        self.containers = containers
        self.agent = agent
        if self.conf.use_cameras and self.agent.usesConv:
            if self.conf.use_second_camera:
                self.vvec_hist = np.zeros([(conf.history_frame_nr+1)*2, conf.image_dims[0], conf.image_dims[1]], dtype=conf.visionvecdtype) #+1 weil das past timestep da immer bei ist
            else:
                self.vvec_hist = np.zeros([(conf.history_frame_nr+1), conf.image_dims[0], conf.image_dims[1]], dtype=conf.visionvecdtype) 
        else:
            self.vvec_hist = None
        self.action_hist = [None]*(self.conf.history_frame_nr+1)
        self.otherinput_hist = [empty_inputs()]*(self.conf.history_frame_nr+1) #defined at the top, is a namedtuple #again +1 because of past timestep
        self.CTimestamp, self.STimestamp = MININT, MININT
        self.alreadyread = True
        self.msperframe = conf.msperframe
        self.hit_a_wall = False
        self.just_reset = True
        self.has_past_state = False
        
        
    def _read_vvec_hist(self, readPast=False):
#        if self.vvec_hist == None: 
#            return None
        hframes = self.conf.history_frame_nr 
        if hframes == 1 and self.conf.use_second_camera:
            return ("error", "error")
                
        if self.conf.use_second_camera:
            return (self.vvec_hist[1:hframes+1], self.vvec_hist[hframes+2:hframes*2+2]) if readPast else (self.vvec_hist[:hframes], self.vvec_hist[hframes+1:hframes*2+1])
        else:
            return (self.vvec_hist[1:hframes+1], None) if readPast else (self.vvec_hist[:hframes], None)

    #in der vvechist steht das älteste hinten: a = [4,3,2,1,0] -> [5] + a[:-1] -> [5,4,3,2,1]   (wobei die current vvec_hist [5,4,3,2] ist und die past [4,3,2,1])
    #wenn wir beide kameras nutzen ist es [4.1, 3.1, 2.1, 1.1, 0.1, 4.2, 3.2, 2.2, 1.2, 0.2]           
    def _append_vvec_hist(self, cam1, cam2):
        hframes = self.conf.history_frame_nr 
        visionvec = np.expand_dims(np.array(cam1, dtype=self.containers.conf.visionvecdtype), axis=0)
        vvec2 = np.expand_dims(np.array(cam2, dtype=self.containers.conf.visionvecdtype), axis=0)
        if self.conf.use_second_camera:  
            self.vvec_hist = np.concatenate((visionvec, self.vvec_hist[:hframes+1-1], vvec2, self.vvec_hist[hframes+1:-1]))
        else:
            self.vvec_hist = np.concatenate((visionvec, self.vvec_hist[:hframes+1-1]))
    
    def _append_other(self, toadd, original):
        original = [toadd]+original[:-1]
        return original

    def _read_other(self, what, readPast=False):
        hframes = self.conf.history_frame_nr
        return what[1:hframes+1] if readPast else what[:hframes]

    
    def update(self, visionvec, vvec2, othervecs, STimestamp, CTimestamp):
        self.lock.acquire()
        try:
            if not self.just_reset:
                assert self.action_hist[0] is not None, "the output-val didn't add the last action before running again!"
                self.has_past_state = True
            #20.7.: deleted the "if is_new..." functionality, as I think its absolutely not helpful
            otherinputs = make_otherinputs(othervecs).normalized() #is now a namedtuple instead of an array            
            
            if hasattr(self.containers.myAgent, "memory") and self.conf.time_ends_episode and  otherinputs.ProgressVec.Laptime >= self.conf.time_ends_episode:
                self.containers.myAgent.endEpisode("timeover", self.read())
                                          
            if self.conf.use_cameras and self.agent.usesConv:
                self._append_vvec_hist(visionvec, vvec2)
            self.otherinput_hist = self._append_other(otherinputs, self.otherinput_hist)
            self.containers.myAgent.humantakingcontrolstring = "(H)" if self.action_hist[0] is None or otherinputs.Action is None or np.any([abs(self.action_hist[0][i] - otherinputs.Action[i]) > 0.1 for i in range(len(otherinputs.Action))]) else ""
            self.action_hist[0] = tuple(otherinputs.Action) #it was already added in addAction, and will only overwritten here if humantakingcontrol changed it
            self.action_hist = self._append_other(None, self.action_hist)   #will be updated in addAction         
            
                                                 
            #wenn otherinputs.CenterDist >= 10 war und seitdem keine neue action kam, muss er >= 10 bleiben!
#            if self.otherinput_hist[0].CenterDist[0] >= 0.99: 
#                self.hit_a_wall = True 
#            #wird erst sobald ne action kommt wieder false gesetzt.. und solange es true ist:
#            if self.hit_a_wall:
#                self.otherinput_hist[0] = self.otherinput_hist[0]._replace(CenterDist = [1])
                
            try:
                if self.conf.reset_if_wrongdirection:
                    if not self.otherinput_hist[0].SpeedSteer.rightDirection:
                        self.containers.wrongdirectiontime += self.containers.conf.msperframe
                        if self.containers.wrongdirectiontime >= 2000: #bei 2 sekunden falsche richtung
                            #resetUnityAndServer(self.containers, punish=self.containers.myAgent.wrongDirPunish)
                            self.containers.myAgent.endEpisode("turnedaround", self.read())
                    else:
                        self.containers.wrongdirectiontime = 0
            except IndexError:
                self.containers.wrongdirectiontime = 0
                              
            self.alreadyread = False
            self.CTimestamp, self.STimestamp = CTimestamp, STimestamp
            print("Updated Input-Vec from", STimestamp, level=2)
            self.just_reset = False
        finally:
            self.lock.release()
            
            
    def addAction(self, action):
        self.hit_a_wall = False #sobald ne action danach kommt ist es unrelated #TODO: gilt nur wenn wallhit_means_reset in Unity
        self.action_hist[0] = action
            
        
    def reset(self, interval, nolock = False):
        if not nolock: 
            self.lock.acquire()
        try:      
            conf = self.conf
            if self.conf.use_cameras and self.agent.usesConv:
                if self.conf.use_second_camera:
                    self.vvec_hist = np.zeros([(conf.history_frame_nr+1)*2, conf.image_dims[0], conf.image_dims[1]], dtype=conf.visionvecdtype) #+1 weil das past timestep da immer bei ist
                else:
                    self.vvec_hist = np.zeros([(conf.history_frame_nr+1), conf.image_dims[0], conf.image_dims[1]], dtype=conf.visionvecdtype) 
            else:
                self.vvec_hist = None
            self.action_hist = [None]*(self.conf.history_frame_nr+1)
            self.otherinput_hist = [empty_inputs()]*(self.conf.history_frame_nr+1) #defined at the top, is a namedtuple #again +1 because of past timestep
            self.CTimestamp, self.STimestamp = MININT, MININT
            self.alreadyread = True
            self.msperframe = interval #da Unity im diesen Wert immer bei spielstart schickt, wird msperframe immer richtig sein            
            assert int(self.msperframe) == int(self.conf.msperframe)
            self.hit_a_wall = False
            self.just_reset = True
            self.has_past_state = False
            logging.debug("Resettet input-value")
        finally:
            if not nolock:
                self.lock.release() 

    def read(self, pastState=False):
        if not pastState:
            self.alreadyread = True
        if pastState and not self.has_past_state:
            return False
        if self.conf.use_cameras and self.agent.usesConv:
            return self._read_vvec_hist(pastState)[0], self._read_vvec_hist(pastState)[1], self._read_other(self.otherinput_hist,pastState), self._read_other(self.action_hist,pastState)
        else:
            return None, None, self._read_other(self.otherinput_hist,pastState), self._read_other(self.action_hist,pastState)
        #like I said, this return everything that could be used by an agent. Not every agent uses that. The standard-agent for example uses...
        #state = (vvec1_hist, vvec2_hist, otherinput_hist[0].SpeedSteer.velocity) #vision plus speed
        #action = action_hist[0]
 
        
#once the agent's network ran on what it got from the inputvec, it will update the value of the otputvalcontainer. 
#There is also guaranteedly only one outputvalcontainer, and the outputvalcontainer is also the one responsible for sending the result back to Unity as soon as its done.
#When the outputval is updated, it will also add the action python suggests back to the inputval, as some kinds of agents also require the last action as input of the net.
class OutputValContainer(object):    
    def __init__(self, containers):
        self.lock = threading.Lock()
        self.value = ""
        self.CTimestamp, self.STimestamp = MININT, MININT
        self.containers = containers
        
        
    #you update only if the new input-timestamp > der alte 
    def update(self, toSend, toSave, CTimestamp, STimestamp):
        self.lock.acquire()
        try:
            if int(self.STimestamp) < int(STimestamp):
                self.value = toSend
                self.containers.inputval.addAction(toSave)
                self.CTimestamp, self.STimestamp = CTimestamp, STimestamp #es geht nicht um jetzt, sondern um dann als das ANN gestartet wurde
                print("Updated output-value to", toSend, level=4)
                self.send_via_senderthread(self.value, self.CTimestamp, self.STimestamp)
            else:
                print("Didn't update output-value because the new one wouldn't be newer", level=10)
                #raise
        finally:
            self.lock.release()


    def reset(self):
        self.lock.acquire()
        try:
            self.value = ""
            self.CTimestamp, self.STimestamp = MININT, MININT
            logging.debug("Resettet output-value")
        finally:
            self.lock.release()
            
            
    def freezeUnity(self):
        self.send_via_senderthread("pleaseFreeze", self.containers.inputval.CTimestamp, self.containers.inputval.STimestamp)

    def unFreezeUnity(self):
        self.send_via_senderthread("pleaseUnFreeze", self.containers.inputval.CTimestamp, self.containers.inputval.STimestamp)
    
            
    def send_via_senderthread(self, value, CTimestamp, STimestamp):
        #nehme die erste verbindung die keinen error schemißt!   
        print("PYTHON SENDING TIME:", STimestamp, time.time()*1000, level=4)
        if self.containers.KeepRunning:
            assert len(self.containers.senderthreads) > 0, "There is no senderthread at all! How will I send?"
            for i in range(len(self.containers.senderthreads)):
                try:
                    self.containers.senderthreads[i].send(value, CTimestamp, STimestamp)
                except (ConnectionResetError, ConnectionAbortedError):
                        #if unity restarted, the old connection is now useless and should be deleted
                        print("I assume you just restarted Unity.")
                        self.containers.senderthreads[i].delete_me()
                        self.containers.senderthreads[i].join()
                        if i >= len(self.containers.senderthreads)-1:
                            break


###############################################################################
    
#just like the ReceiverListenerThread waits for Unity connecting to RECEIVE info, this one waits for unity to connnect to SEND info TO UNITY (they listen on different ports)
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
                ct = sender_thread(clt, self.containers)
                self.containers.senderthreads.append(ct)
                ct.start()
            except socket.timeout:
                #simply restart and don't care
                pass   



class sender_thread(threading.Thread):
    def __init__(self, clientsocket, containers):
        threading.Thread.__init__(self)
        self.clientsocket = clientsocket
        self.containers = containers
        self.containers.UnityConnected = True
        
    def run(self):
        print("Starting sender_thread")
        
    def delete_me(self):
        selfind = self.containers.senderthreads.index(self)
        del self.containers.senderthreads[selfind]
        print("stopping sender_thread")
        #ist er jetzt wirklich ganz weg?
        
        
    def send(self, result, CTimestamp, STimestamp):
        tosend = str(result) + "CTime(" +str(CTimestamp)+")"+ "STime(" +str(STimestamp)+")"
        print("Sending", tosend, level=3)
        self.clientsocket.mysend(tosend)



def showhelp():
    print("""Command-line arguments:
"-DQN" to run with the DQN-config
"-half_DQN" to run with a custom config, corresponding roughly to half the DQN-config in all respects
"-nolearn" to store agent's results to the memory, but not perform Reinforcement learning (sets the random-action-chance to 0)
"-noscreen" to turn off the screen showing Q-vals etc.
"-noplot" to turn off the plots evaluating each episode
"-startfresh" to use a RL-agent without and (supervised or reinforcement) pretraining
"-nomemorykeep" to not save the memory for this run.
"-help" shows this help and exits
Concerning agents:
Without any arguments, the agent defined in "dqn_rl_agent" is used
With the "-svplay"-argument, the agent defined in "dqn_sv_agent" is used
With the argument "--agent xyz", the agent defined in "xyz.py" is used""", level=999)
    

###############################################################################
##################### ACTUAL STARTING OF THE STUFF#############################
###############################################################################

class Containers():
    def __init__(self):
        self.KeepRunning = True
        self.receiverthreads = []
        self.senderthreads = []
        self.myAgent = None
        self.wrongdirectiontime = 0
        self.freezeInf = self.freezeLearn = False
        self.UnityConnected = False
        
        
def create_socket(port):
    server_socket = MySocket()
    server_socket.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(TCP_IP, port)
    server_socket.listen(1)
    return server_socket



def main(conf, agentname, no_learn, show_screen, show_plots, start_fresh, nomemorykeep):
    containers = Containers()
    containers.conf = conf
    containers.no_learn = no_learn
    
    containers.receiverportsocket = create_socket(TCP_RECEIVER_PORT)
    containers.senderportsocket = create_socket(TCP_SENDER_PORT)
        
    if no_learn:
        conf.learnMode = ""
        conf.minepsilon = 0
        conf.startepsilon = 0 #or whatever random-value will be in 
            
    agentclass = __import__(agentname).Agent
    containers.myAgent = agentclass(conf, containers, isPretrain=False, start_fresh=start_fresh) 
    containers.myAgent.initForDriving(keep_memory = (not nomemorykeep), show_plots=show_plots)
    
    if show_screen:
        screenroot = infoscreen.showScreen(containers)
    else:
        containers.showscreen = False    
                                                                          
    containers.inputval = InputValContainer(conf, containers, containers.myAgent)
    containers.outputval = OutputValContainer(containers)  
        
        
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
    if hasattr(containers.myAgent, "memory") and not no_learn and conf.learnMode == "parallel":
        dauerLearn = partial(containers.myAgent.dauerLearnANN, learnSteps=conf.train_for)
        learnthread = threading.Thread(target=dauerLearn)
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
    containers.UnityConnected = False
    if hasattr(containers.myAgent, "memory") and not no_learn and conf.learnMode == "parallel":
        learnthread.join()
        
    if containers.conf.save_memory_on_exit and hasattr(containers.myAgent, "memory") and containers.myAgent.keep_memory:
        containers.myAgent.memory.save_memory()
        
    time.sleep(0.1)
    print("Server shut down sucessfully.")
    
    


    
if __name__ == '__main__':  
    if "-help" in sys.argv:
        showhelp()
        exit()   
        
    if ("-DQN" in sys.argv):
        conf = config.DQN_Config()
    elif ("-half_DQN" in sys.argv):
        conf = config.Half_DQN_Config()
    else:
        conf = config.Config()
            

    if "--agent" in sys.argv:
        num = sys.argv.index("--agent")
        try:
            agentname = sys.argv[num+1]
            if agentname[0] == "-": raise IndexError
        except IndexError:
            print("With the '--agent'-Parameter, you need to specify an agent!")
            exit(0)
    else:
        if "-svplay" in sys.argv:
            agentname = "dqn_sv_agent"
        else:
            agentname = "dqn_rl_agent"
            
    main(conf, agentname, ("-nolearn" in sys.argv), not ("-noscreen" in sys.argv), not ("-noplot" in sys.argv), ("-startfresh" in sys.argv), ("-nomemorykeep" in sys.argv))    