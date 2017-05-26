from tkinter import *
import queue

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
    lastcommand = ThreadSafeConsole(root, width=1, height=1)
    lastcommand.pack(fill=X)
    memorysize = ThreadSafeConsole(root, width=1, height=1)
    memorysize.pack(fill=X)
    lastmemory = ThreadSafeConsole(root, width=1, height=1)
    lastmemory.pack(fill=X)
    currentqvals = ThreadSafeConsole(root, width=50, height=22)
    currentqvals.pack(fill=X)

    containers.showscreen = True
    containers.screenwidgets = {"Memorysize": memorysize, "Last command": lastcommand, "Last memory": lastmemory, "Current Q Vals": currentqvals}
    return root