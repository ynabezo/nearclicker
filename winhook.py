import pyWinhook
hm = pyWinhook.HookManager()

def on():
    global hm
    hm.HookMouse()
    return

def off():
    global hm
    hm.UnhookMouse()
    return

def up(p):
    global hm
    hm.MouseLeftUp = p
    return

def down(p):
    global hm
    hm.MouseLeftDown = p
    return
