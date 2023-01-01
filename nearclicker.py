from PIL import ImageGrab,Image,ImageTk
import cv2 as cv2
import numpy  as np
import time
#import pynput
import pyautogui
import cv2img
import pyWinhook
import pythoncom
import threading
import tkinter

cns_area=50
cns_imgsize=150

#mouse = pynput.mouse.Controller()
isclick=False
gtimeclick=time.perf_counter()
gcnt=0
gdcnt=0
gucnt=0
gswevt=False
#gswevt=True
(gd,gx,gy)=(0,0,0)
gimg=None
gimgflg=True

app=tkinter.Tk()
def forminit(app):
    global canvas
    frame = tkinter.Frame(app)
    frame.pack(fill = tkinter.BOTH, padx=5, pady=10)
    button1=tkinter.Button(frame,width=10,height=3
            ,text="end")
    button1.bind("<Button-1>" ,ebtnend)
    button1.bind("<Button-2>" ,ebtnend,'+')
    button1.bind("<Button-3>" ,ebtnend,'+')
    button1.pack()

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
    app.attributes("-topmost", True)
def ebtnend(event):
    global app
    app.quit()
    return
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
        t=threading.Timer(0.4,clickcheck,args=(mx,my))
        t.start()
        g=threading.Thread(target=p_getxy,args=(mx,my))
        g.start()
    return False
def lup(e):
    global gdcnt,gucnt,gcnt,gswevt
    if gswevt:
        print("up")
        return True
    print("lup")
    gcnt=gcnt+1
    if gdcnt==0:
        return True 
    gucnt+=1        #up count
    return False

def clickcheck(mx,my):
    global gd,gx,gy,gimg
    global gdcnt,gucnt,gswevt
    if gdcnt==1 and gucnt==1:
        print("click")
        gswevt=True
        pyautogui.move(gx,gy)
        pyautogui.click()
    elif gdcnt==2 and gucnt==2:
        print("dblclick")
        gswevt=True
        pyautogui.move(gx,gy)
        pyautogui.doubleClick()
    elif gdcnt==1 and gucnt==0:
        print("drag")
        gswevt=True
        pyautogui.move(gx,gy)
        pyautogui.mouseDown()
    
    gdcnt=0
    gucnt=0
    gswevt=False
    
def p_getxy(mx,my):
    global gd,gx,gy,gimg,gimgflg
    (gd,gx,gy)=(0,0,0)
    gimg,gd,gx,gy=getxy(mx,my)
    gimgflg=True
    return

def getmouseposition():
    (x,y)=pyautogui.position()
    return x, y

def getcapimg(x,y,r):
    rr=r//2
    #x=x*1.25
    #y=y*1.25
    x1=x-rr
    y1=y-rr
    x2=x+rr
    y2=y+rr
    #print(x,y,x1,y1,x2,y2)
    # スクリーンショット取得
    imgpil = ImageGrab.grab(bbox=(x1,y1,x2,y2))
    #img = np.asarray(img)
    return imgpil

def getcirclearea(img_WK, x,y,r):
    global cns_area
    #輪郭が細かい方(凹んでいる方）を反転してマスクとして利用
    img_NULL = cv2img.getNullImageGray(img_WK)
    img_INNER = cv2.circle(img_NULL,(x,y),r,(255,255,255),-1)
    #凹んでいる差分の画像を取得
    img_GRAYDIFF = cv2.bitwise_and(img_WK, img_INNER )
    return img_GRAYDIFF     #GRAY

def imgproc(imgpil):
    dicp={}
    img = np.array(imgpil, dtype=np.uint8)
    imgrgb = cv2img.cvtColorBGR2RGB(img)
    imgorg=imgrgb
    #img = img.astype(np.uint8)
    img=cv2img.cvtColorBGR2GRAY(img)
    #img=cv2img.cnvbinalize(img,"otsu")
    # 適応的しきい値処理
    #img = cv2.GaussianBlur(img, (15, 15), 0)
    img = cv2.adaptiveThreshold(img, 255, 1, 1, 9, 20)  
    #        cv2.THRESH_BINARY, 51, 20)
    #retval, img = cv2.threshold(img, 50, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/360, threshold=30, minLineLength=3  , maxLineGap=60)
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/2,
                    threshold=40, minLineLength=20, maxLineGap=0)
    cv2img.drawLines(img,lines,(0,0,0),1)
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/2,
                    threshold=40, minLineLength=20, maxLineGap=0)
    cv2img.drawLines(img,lines,(0,0,0),1)
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/1,
                    threshold=20, minLineLength=20, maxLineGap=0)
    cv2img.drawLines(img,lines,(0,0,0),1)

    kernel = np.ones((2,2),np.uint8)
    #img = cv2.erode(img,kernel,iterations = 1)
    kernel = np.ones((5,5),np.uint8)
    img = cv2.dilate(img,kernel,iterations = 1)
    x=int(imgorg.shape[0]/2)
    y=int(imgorg.shape[1]/2)

    img=getcirclearea(img,x,y,cns_area)
    #img=cv2img.cvtColorGRAY2BGR(img)
    #return img,dicp

    conts=cv2img.getContours(img)
    lstcont=[]
    for cont in conts:
        a = cv2.contourArea(cont)
        x,y,w,h = cv2.boundingRect(cont)
        if (w<7)or (w<12 and a<80):
            #print("#",w,h,a)
            continue
        #print(w,h,a)
        lstcont.append(cont)


    img=cv2img.cvtColorGRAY2BGR(img)
    #cv2img.drawContours(img,conts,(0,0,255),2)
    
    x=int(imgorg.shape[0]/2)
    y=int(imgorg.shape[1]/2)
    cv2.circle(imgorg,(x,y),4,(0,0,255),3)
    cv2.circle(imgorg,(x,y),cns_area,(255,0,0),3)    
    for cont in lstcont:
        rect = cv2img.getMinAreaRect(cont)
        cv2img.drawMinAreaRect(imgorg,rect,(0,255,0),1,0)
        M=cv2.moments(cont)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(imgorg,(cx,cy),2,(0,0,255),2)
        d=getdistance(x,y,cx,cy)
        dicp[d]=(cx-x,cy-y)
        
    #print(imgorg.shape)
    return imgorg,dicp

def getdistance(x1,y1,x2,y2):
    a=np.array([x1,y1])
    b=np.array([x2,y2])
    distance=np.linalg.norm(b-a)
    return distance

def getxy(mx,my):
    global cns_area,cns_imgsize
    print("getxy",mx,my)
    k=0
    (x,y)=(0,0)
    img=getcapimg(mx,my,cns_imgsize)
    img,dicp=imgproc(img)
    dicp=dict(sorted(dicp.items()))
    #print(dicp)
    for k,v in dicp.items():
        if k < cns_area:
            (x,y)=v
        break
    return img,k,x,y

def test():
    global app,canvas,gimg,time_sta,img_TK
    pythoncom.PumpWaitingMessages
    img_PIL=getcapimg(0,0,100)
        
    #img_PIL = Image.fromarray(img) # RGBからPILフォーマットへ変換
    img_TK  = ImageTk.PhotoImage(image=img_PIL) # ImageTkフォーマットへ変換
    print(img_TK)

    #canvas.delete('img')
    canvas.create_image(100, 100, image=img_TK , anchor='nw', tag='img')
    app.after(1000, test)

def main():
    global app,canvas,gimg,gimgflg,img_TK,time_sta
    pythoncom.PumpWaitingMessages
    time_end = time.perf_counter()
    tim = time_end- time_sta
    #if tim > 0.25:
    if gimgflg==True:
        img_PIL = Image.fromarray(gimg) # RGBからPILフォーマットへ変換
        img_TK  = ImageTk.PhotoImage(img_PIL) # ImageTkフォーマットへ変換
        canvas.delete('img')
        canvas.create_image(0, 0, image=img_TK
                ,  anchor='nw', tag='img')
        gimgflg=False
        #d,x,y = getxy(mx,my)
        time_sta = time.perf_counter()
    app.after(10, main)

if __name__ == "__main__":
    gcnt=0
    (x,y)=(0,0)
    forminit(app)
    hm = pyWinhook.HookManager()
    hm.MouseLeftDown = ldown
    hm.MouseLeftUp = lup
    hm.HookMouse()
    mx,my=getmouseposition()
    gimg,d,x,y = getxy(mx,my)

    time_sta = time.perf_counter()
    main()
    app.mainloop()
    #while True:
    #    main(time_sta)
        #if gcnt > 20:break
    #cap.release()
    #cv.destroyAllWindows()

