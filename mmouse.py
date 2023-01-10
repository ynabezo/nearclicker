import mouse

def move(rx,ry):
    mouse.move(rx,ry,False)
    return

def click():
    mouse.click()
    return

def doubleclick():
    mouse.double_click()
    return

def down():
    mouse.press(mouse.LEFT)
    return

def position():
    (x,y)=mouse.get_position()
    return (x,y)