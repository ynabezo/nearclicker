import pyautogui
pyautogui.FAILSAFE=False

def move(rx,ry):
    pyautogui.move(rx,ry)
    return

def click():
    pyautogui.click()
    return

def doubleclick():
    pyautogui.doubleClick()
    return

def down():
    pyautogui.mouseDown()
    return

def position():
    (x,y)=pyautogui.position()
    return (x,y)