from PIL import ImageGrab,Image,ImageTk
import cv2 as cv2
import numpy  as np
import cv2img

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

def imgproc(imgpil,areasize):
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

    img=getcirclearea(img,x,y,areasize)
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
    cv2.circle(imgorg,(x,y),areasize,(255,0,0),3)    
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

def getcirclearea(img_WK, x,y,r):
    #輪郭が細かい方(凹んでいる方）を反転してマスクとして利用
    img_NULL = cv2img.getNullImageGray(img_WK)
    img_INNER = cv2.circle(img_NULL,(x,y),r,(255,255,255),-1)
    #凹んでいる差分の画像を取得
    img_GRAYDIFF = cv2.bitwise_and(img_WK, img_INNER )
    return img_GRAYDIFF     #GRAY

def getxy(mx,my,areasize,imgsize):
    print("getxy",mx,my)
    k=0
    (x,y)=(0,0)
    img=getcapimg(mx,my,imgsize)
    img,dicp=imgproc(img,areasize)
    dicp=dict(sorted(dicp.items()))
    #print(dicp)
    for k,v in dicp.items():
        if k < areasize:
            (x,y)=v
        break
    return img,k,x,y

def cv2tk(cvimg):
    img_PIL = Image.fromarray(cvimg) # RGBからPILフォーマットへ変換
    img_TK  = ImageTk.PhotoImage(img_PIL) # ImageTkフォーマットへ変換
    return img_TK