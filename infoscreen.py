from tkinter import Text, END, Tk, X, Canvas
import queue
import time

class ThreadSafeConsole(Text):
    def __init__(self, master, **options):
        Text.__init__(self, master, **options)
        self.queue = queue.Queue()
        self.update_me()
    def write(self, line):
        self.queue.put(line)
    def clear(self):
        self.queue.put(None)
    def update_me(self):
        try:
            while 1:
                line = self.queue.get_nowait()
                if line is None:
                    self.delete(1.0, END)
                else:
                    self.insert(END, str(line))
                self.see(END)
                self.update_idletasks()
        except queue.Empty:
            pass
        self.after(100, self.update_me)
        
        
class ThreadSafeCanvas(Canvas):
    def __init__(self, master, **options):
        Canvas.__init__(self, master, **options)
        self.queue = queue.Queue(maxsize=1)
        self.update_me()
    def updateCol(self, content):
        self.queue.put(content)
    def clear(self):
        self.queue.put(None)
    def colorFromValue(self, val):
        col = min(255,max(0,int(val*255)))
        return '#%02x%02x%02x' % (col, col, col)
    def update_me(self):
        try:
            while 1:
                content = self.queue.get_nowait()
                if content is not None:
                    self.create_rectangle(10, 10, 190, 90, fill=self.colorFromValue(content)) 
                    self.update_idletasks()
        except queue.Empty:
            pass
        self.after(100, self.update_me)        

# this function pipes input to a widget
def print(*args, containers, wname):
    widget = containers.screenwidgets[wname]
    widget.clear()
    if wname != "Current Q Vals":
        text = wname+": "+" ".join([str(i) for i in args])
    else:
        text = wname+"\n"+" ".join([str(i) for i in args])
    widget.write(text)
    
    
def showScreen(containers):
    root = Tk()     
    root.title("rAIce-ANN - Started "+time.strftime("%H:%M:%S", time.gmtime()))                    
    lastcommand = ThreadSafeConsole(root, width=55, height=1)
    lastcommand.pack(fill=X)
    memorysize = ThreadSafeConsole(root, width=1, height=1)
    memorysize.pack(fill=X)
    lastmemory = ThreadSafeConsole(root, width=1, height=1)
    lastmemory.pack(fill=X)
    epsilon = ThreadSafeConsole(root, width=1, height=1)
    epsilon.pack(fill=X)
    lastpunish = ThreadSafeConsole(root, width=1, height=1)
    lastpunish.pack(fill=X)
    reinflearnsteps = ThreadSafeConsole(root, width=1, height=1)
    reinflearnsteps.pack(fill=X)
    lastepisode = ThreadSafeConsole(root, width=1, height=2)
    lastepisode.pack(fill=X)
    if containers.conf.showColorArea:
        colorarea = ThreadSafeCanvas(root, width=200, height=100)
        colorarea.pack();
    else:
        colorarea = None
    if not containers.myAgent.isContinuous:
        currentqvals = ThreadSafeConsole(root, width=55, height=containers.conf.dnum_actions+1)
        currentqvals.pack(fill=X)
    else:
        currentqvals = None
    x = root.winfo_screenwidth()-449
    y = 0 #root.winfo_screenheight()-200
    root.geometry('+%d+%d' % (x, y))   

    containers.showscreen = True
    containers.screenwidgets = {"Last command": lastcommand, "Memorysize": memorysize, "Last memory": lastmemory, "Epsilon": epsilon, "Last big punish": lastpunish, \
                                "ReinfLearnSteps": reinflearnsteps, "Current Q Vals": currentqvals, "Last Epsd": lastepisode, "ColorArea": colorarea}
    return root