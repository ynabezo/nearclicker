import time
import pythoncom
import threading
import tkinter
import winhook as hook
#import mautoguy as mouse
import mmouse as mouse
import imageproc

from Socket_Singleton import Socket_Singleton
Socket_Singleton()

cns_area=50
cns_imgsize=120
gcnt=0
gdcnt=0
gucnt=0
gswevt=False
(gd,gx,gy)=(0,0,0)
gimg=None
gimgflg=True

app=tkinter.Tk()
def forminit(app):
    global canvas
    frame = tkinter.Frame(app)
    frame.pack(fill = tkinter.BOTH, padx=5, pady=10)
    button1=tkinter.Button(frame,width=11,height=3
            ,text="end")
    button1.pack(side="left")
    button2=tkinter.Button(frame,width=11,height=3
            ,text="min")
    button2.pack()
    button1.bind("<1>" ,e_btnend)
    button1.bind("<2>" ,e_btnend,'+')
    button1.bind("<3>" ,e_btnend,'+')
    button2.bind("<1>" ,e_btnmin)
    button2.bind("<3>" ,e_btnmin)
    canvas = tkinter.Canvas(
    app,
    image=None,
    width=cns_imgsize,
    height=cns_imgsize
    )
    canvas.pack()
    canvas.create_image(0,0,image=None,tag='img')
    app.title("nearclicker")
    app.resizable(0,0)
    #app.attributes("-topmost", True)
def e_btnend(event):
    ap_end()
    return
def ap_end():
    global app
    app.quit()
    exit
    return
def e_btnmin(e):
    print("iconic")
    app.state(newstate='iconic')
def ldown(e):
    global gdcnt,gcnt,gswevt
    if gswevt:
        print("down")
        return True
    print("ldown")
    gcnt=gcnt+1
    gdcnt+=1        #down count
    if gdcnt==1:
        mx=e.Position[0]        
        my=e.Position[1]
        t=threading.Timer(0.6,clickcheck)
        t.start()
        g=threading.Thread(target=p_getxy,args=(mx,my))
        g.start()
    return False
def lup(e):
    global gdcnt,gucnt,gcnt,gswevt
    if gswevt:
        print("up")
        return True
    if gdcnt==0:
        print("dragup")
        return True 
    gcnt=gcnt+1
    print("lup")
    gucnt+=1        #up count
    return False

def clickcheck(): 
    global gd,gx,gy,gimg
    global gdcnt,gucnt,gswevt,gcnt
    #print("check-s",gcnt)
    if gdcnt==1 and gucnt==1:
        print("click")
        gswevt=True
        hook.off()
        mouse.move(gx,gy)
        mouse.click()
    elif gdcnt==2 and gucnt==2:
        print("dblclick")
        gswevt=True
        hook.off()
        mouse.move(gx,gy)
        mouse.doubleclick()
    elif gdcnt==1 and gucnt==0:
        print("drag")
        gswevt=True
        hook.off()
        mouse.move(gx,gy)
        mouse.down()
    #pythoncom.PumpWaitingMessages
    gdcnt=0
    gucnt=0
    return
def p_getxy(mx,my):
    global gd,gx,gy,gimg,gimgflg
    (gd,gx,gy)=(0,0,0)
    gimg,gd,gx,gy=imageproc.getxy(mx,my,cns_area,cns_imgsize)
    gimgflg=True
    return

def main():
    global app,canvas,gimg,gimgflg,img_TK
    global gswevt
    if gswevt:
        time.sleep(0.2)
        print("--hookon", )
        hook.on()
        gswevt=False
    pythoncom.PumpWaitingMessages
    #time.sleep(0.2)
    #if tim > 0.25:
    if gimgflg==True:
        img_TK=imageproc.cv2tk(gimg)
        canvas.delete('img')
        canvas.create_image(0, 0, image=img_TK
                ,  anchor='nw', tag='img')
        gimgflg=False
    app.after(100, main)

if __name__ == "__main__":
    forminit(app)
    hook.up(lup)
    hook.down(ldown)
    hook.on()
    mx,my=mouse.position()
    gimg,d,x,y = imageproc.getxy(mx,my,cns_area,cns_imgsize)
    #to=threading.Timer(300,ap_end)
    #to.start()
    main()
    
    app.mainloop()

