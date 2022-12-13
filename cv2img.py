import cv2
import numpy as np
##################################
class cv2video:
    def __init__(self):
        self.cap = None
        self.frame = None
    def read(self):
        if self.cap is None:
            return None
        ret, self.frame = self.cap.read()
        return ret
    def open(self, strURL):
        if self.cap is None:
            if len(strURL) == 1:
                self.cap = cv2.VideoCapture(int(strURL))
            else:
                self.cap = cv2.VideoCapture(strURL)
        return self.cap
    def isopen(self):
        if self.cap is None:
            return False
        else:
            return True
    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    def setsize(self, w, h):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
    def getsize(self):
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return w, h

class cv2pic:
    def __init__(self):
        self.frame = None
    def readpic(fname):
        self.frame = cv2.imread(fname)
        return True

def imread(fname):
    img_WK = cv2.imread(fname)
    return img_WK

def show(img_WK):
    cv2.namedWindow('window', cv2.WINDOW_NORMAL)
    cv2.imshow('window', img_WK)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cvtColorGRAY2BGR(img_TMP):
    try:
        img_WK = cv2.cvtColor(img_TMP, cv2.COLOR_GRAY2BGR)
    except:
        None
    return img_WK

def cvtColorBGR2GRAY(img_TMP):
    try:
        img_WK = cv2.cvtColor(img_TMP, cv2.COLOR_BGR2GRAY)
    except:
        None
    return img_WK

def cvtColorBGR2RGB(img_TMP):
    try:
        img_WK = cv2.cvtColor(img_TMP, cv2.COLOR_BGR2RGB)
    except:
        None
    return img_WK

def cnvcolor(img_WK,strtype):
    if strtype == "gray":     #"gray":
        img_OUT = cv2.cvtColor(img_WK, cv2.COLOR_BGR2GRAY)
    elif strtype == "decolor":     #"decolor":
        img_OUT, _ = cv2.decolor(img_WK)
    elif strtype == "bgr":
        img_OUT = img_WK.copy()
    elif strtype == "bgr-b":
        b,g,r = cv2.split(img_WK)
        img_OUT = b
    elif strtype == "bgr-g":
        b,g,r = cv2.split(img_WK)
        img_OUT = g
    elif strtype == "bgr-r":
        b,g,r = cv2.split(img_WK)
        img_OUT = r
    elif strtype == "hsv":
        img_OUT = cv2.cvtColor(img_WK, cv2.COLOR_BGR2HSV)
    elif strtype == "hsv-h":
        img_OUT1 = cv2.cvtColor(img_WK, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(img_OUT1)
        img_OUT = h
    elif strtype == "hsv-s":   #"lab":
        img_OUT1 = cv2.cvtColor(img_WK, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(img_OUT1)
        img_OUT = s
    elif strtype == "hsv-v":   #"lab":
        img_OUT1 = cv2.cvtColor(img_WK, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(img_OUT1)
        img_OUT = v
    elif strtype == "lab":     #"lab":
        img_OUT = cv2.cvtColor(img_WK, cv2.COLOR_BGR2Lab)
    elif strtype == "lab-l":     #"lab":
        img_OUT1 = cv2.cvtColor(img_WK, cv2.COLOR_BGR2Lab)
        l,a,b = cv2.split(img_OUT1)
        img_OUT = l
    elif strtype == "lab-a":     #"lab":
        img_OUT1 = cv2.cvtColor(img_WK, cv2.COLOR_BGR2Lab)
        l,a,b = cv2.split(img_OUT1)
        img_OUT = a
    elif strtype == "lab-b":     #"lab":
        img_OUT1 = cv2.cvtColor(img_WK, cv2.COLOR_BGR2Lab)
        l,a,b = cv2.split(img_OUT1)
        img_OUT = b
    elif strtype == "xyz":     #"XYZ":
        img_OUT = cv2.cvtColor(img_WK, cv2.COLOR_BGR2XYZ)
    elif strtype == "xyz-x":     #"XYZ":
        img_OUT1 = cv2.cvtColor(img_WK, cv2.COLOR_BGR2XYZ)
        x,y,z = cv2.split(img_OUT1)
        img_WK = x
    elif strtype == "xyz-y":     #"XYZ":
        img_OUT1 = cv2.cvtColor(img_WK, cv2.COLOR_BGR2XYZ)
        x,y,z = cv2.split(img_OUT1)
        img_OUT = y
    elif strtype == "xyz-z":     #"XYZ":
        img_OUT1 = cv2.cvtColor(img_WK, cv2.COLOR_BGR2XYZ)
        x,y,z = cv2.split(img_OUT1)
        img_OUT = z
    elif strtype == "hls":     #"HLS":
        img_OUT = cv2.cvtColor(img_WK, cv2.COLOR_BGR2HLS)
    elif strtype == "hls-h":     #"HLS":
        img_OUT1 = cv2.cvtColor(img_WK, cv2.COLOR_BGR2HLS)
        h,l,s = cv2.split(img_OUT1)
        img_OUT = h
    elif strtype == "hls-l":     #"HLS":
        img_OUT1 = cv2.cvtColor(img_WK, cv2.COLOR_BGR2HLS)
        h,l,s = cv2.split(img_OUT1)
        img_OUT = l
    elif strtype == "hls-s":     #"HLS":
        img_OUT1 = cv2.cvtColor(img_WK, cv2.COLOR_BGR2HLS)
        h,l,s = cv2.split(img_OUT1)
        img_OUT = s

    elif strtype == "4pos":     #"HLS":
        # ポスタリゼーション
        lut = np.zeros((256, 1), dtype="uint8") # テーブルの初期化
        for i in range(256):
            if i >= 0 and i < 32:
                lut[i][0] = 0
            elif i >= 32 and i < 64:
                lut[i][0] = 32
            elif i >= 64 and i < 96:
                lut[i][0] = 64
            elif i >= 96 and i < 128:
                lut[i][0] = 96
            elif i >=128 and i < 160:
                lut[i][0] = 128
            elif i >= 160 and i < 192:
                lut[i][0] = 160
            elif i >= 192 and i < 224:
                lut[i][0] = 192
            else :
                lut[i][0] = 224
        img_OUT = cv2.LUT(img_WK, lut)
    return img_OUT

def cnvbinalize(img_TMP, inttype):
    #２値化
    if inttype == "bin128":
        ret2, img_WK = cv2.threshold(img_TMP, 128, 255, cv2.THRESH_BINARY)
    if inttype == "canny":    #canny
        img_WK = cv2.Canny(img_TMP, 30, 60)
    if inttype == "otsu":      #otsu
        ret2, img_WK  = cv2.threshold(img_TMP, 0, 255, cv2.THRESH_OTSU)
    if inttype == "triangle":      #triangle
        ret2, img_WK  = cv2.threshold(img_TMP, 0, 255, cv2.THRESH_TRIANGLE)
    if inttype == "laplacian":      #laplacian
        img_WK = cv2.Laplacian(img_TMP, cv2.CV_32F, ksize=3)
    if inttype == "dilatediff":      #dilatediff
        kernel5 = np.ones((5, 5), np.uint8)
        img_DIL = cv2.dilate(img_TMP, kernel5, iterations=1)
        img_SUB1 = cv2.subtract(img_DIL, img_TMP)
        img_SUB2 = 255 - img_SUB1 
        _, img_WK = cv2.threshold(img_SUB2, 225, 255, cv2.THRESH_BINARY)
    return img_WK

def cnvhist(img_TMP):
    #ヒストグラム平坦化
    img_WK = cv2.equalizeHist(img_TMP)
    return img_WK

def cnvbilateral(img_TMP):
    #バイラテラル平坦化
    # 近傍円直径,色空間標準偏差,座標空間標準偏差
    #img_WK = cv2.bilateralFilter(img_WK,5,10,10)     #0.32 くらい
    #img_WK = cv2.bilateralFilter(img_WK,5,30,30)     #0.34
    img_WK = cv2.bilateralFilter(img_TMP,5,50,50)    #0.35
    #img_WK = cv2.bilateralFilter(img_WK,5,80,80)     #0.33
    #img_WK = cv2.bilateralFilter(img_WK,7,80,10)    #0.39
    #img_WK = cv2.bilateralFilter(img_WK,7,30,30)    #0.39
    #img_WK = cv2.bilateralFilter(img_WK,7,50,50)    #0.39
    #img_WK = cv2.bilateralFilter(img_WK,7,80,80)    #0.41
    #img_WK = cv2.bilateralFilter(img_WK,9,30,30)    #0.51
    #img_WK = cv2.bilateralFilter(img_WK,9,80,80)    #0.48
    return img_WK

def cnvedgepreservingfilter(img_TMP):
    #エッジ保存平滑化フィルタ
    #1:高速 2:標準  座標空間(0~200):2  色範囲(0~1):0.75
    #img_WK = cv2.edgePreservingFilter(img_WK,None, 1, 2, 0.75) 
    img_WK = cv2.edgePreservingFilter(img_TMP,None, 1, 60, 0.6) 
    return img_WK

def cnvbokasi(img_TMP,strtype):
    ##ぼかし
    if strtype=="median":
        img_TMP1 = cv2.medianBlur(img_TMP, 5)
    else:
        img_TMP1 = img_TMP
    if strtype=="dilate":
        kernel = np.ones((6, 6), np.uint8)
        img_TMP2 = cv2.dilate(img_TMP1, kernel)
    else:
        img_TMP2 = img_TMP1
    if strtype=="erode":
        kernel = np.ones((6, 6), np.uint8)
        img_TMP3 = cv2.erode(img_TMP2, kernel)
    else:
        img_TMP3 = img_TMP2
    if strtype=="denoise":
        img_WK = cv2.fastNlMeansDenoising(img_TMP3,h=20)
    else:
        img_WK = img_TMP3.copy()
    return img_WK

def cnvgamma(img_TMP, gamma):
    #gamna 0-1-2
    if gamma == 1.0:
        img_WK = img_TMP.copy()
        return img_WK
    lookUpTable = np.zeros((256, 1), dtype = 'uint8')
    for i in range(256):
        lookUpTable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
    img_WK = cv2.LUT(img_TMP, lookUpTable)
    return img_WK

def cnvdenoise(img_TMP,strtype):
    ##ぼかし
    if strtype=="denoise":
        img_WK = cv2.fastNlMeansDenoising(img_TMP,h=20)
    else:
        img_WK = img_TMP.copy()
    return img_WK

def cnvreverse(img_TMP,strtype):
    img_WK = img_TMP.copy()
    ##反転
    if strtype=="reverse":
        ## OpenCVの輪郭検出は，黒い背景から白い物体の輪郭を検出することと仮定している
        img_WK = cv2.bitwise_not(img_TMP)
    else:
        img_WK = img_TMP.copy()

    return img_WK

def getNullImage(img_WK):
    height = img_WK.shape[0]
    width = img_WK.shape[1]
    img_NULL = np.zeros((height,width, 3), dtype=np.uint8)
    return img_NULL     #BGR

def getNullImageGray(img_WK):
    img_NULL = getNullImage(img_WK)
    img_NULL = cvtColorBGR2GRAY(img_NULL)
    return img_NULL     #GRAY

def getHeight(img_WK):
    return  img_WK.shape[0]
def getWidth(img_WK):
    return  img_WK.shape[1]

def getContours(img_WK):
    if img_WK is None:
        return None
    try:
        contours, hierarchy = cv2.findContours(img_WK, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        _,contours, hierarchy = cv2.findContours(img_WK, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def drawContours(img_WK, contours, color, width):
    if contours is None:
        return
    cv2.drawContours(img_WK, contours, -1, color, width)      #緑
    ####return img_WK

def getApproxLine(contours, factor ):
    #およその輪郭ポイント配列の取得
    #factor : %
    approxlines = []
    for cont in contours:
        ##境界から近似輪郭の取得
        #epsilon = 弧長の10% epsilon = 0.1*cv2.arcLength(cont,True)
        epsilon = factor*cv2.arcLength(cont,True)
        approx = cv2.approxPolyDP(cont,epsilon,True) 
        approxlines.append(approx)
    return approxlines

def drawApproxLine(img_WK, approxlines, color, width ):
    #color : (255,0,0) 青
    if approxlines is None:
        return
    if len(approxlines) == 0:
        return
    for approxline in approxlines:
        cv2.drawContours(img_WK, [approxline], -1, color, width)
    ####return img_WK

def getConvexLine(contours):
    #凸形を取得
    convexlines = []
    for cont in contours:
        convex = cv2.convexHull(cont)
        convexlines.append(convex)
    return convexlines

def getDiffApproxLines(contours, para):
    #凸形を取得
    contourlines = []
    convexlines = []
    for cont in contours:
        convex = cv2.convexHull(cont)           #凸包
        area1 = cv2.contourArea(cont, False)
        area2 = cv2.contourArea(convex, False)
        if area2 >= area1 * float(para) and float(para) > 0:
            #凸包面積>輪郭面積の２倍なら無視
            pass
        else:
            #面積が2倍以内なら有効な差分とする
            contourlines.append(cont)
            convexlines.append(convex)
    return contourlines, convexlines


def getGrayDiffArea(img_WK, lines_outer, lines_inner):
    #輪郭が細かい方(凹んでいる方）を反転してマスクとして利用
    img_NULL = getNullImage(img_WK)
    img_INNER = img_NULL.copy()
    ##cv2.fillPoly(img_INNER, pts=lines_inner, color=(255,255,255) )
    cv2.drawContours(img_INNER, lines_inner, -1, color=(255, 255, 255), thickness=-1)
    img_INNER = cv2.cvtColor(img_INNER, cv2.COLOR_BGR2GRAY)
    img_INNER = cv2.bitwise_not(img_INNER)      #反転
    #外側の輪郭をベースとして利用

    img_OUTER = img_NULL.copy()
    ##cv2.fillPoly(img_OUTER, pts=lines_outer, color=(255,255,255) )
    cv2.drawContours(img_OUTER, lines_outer, -1, color=(255, 255, 255), thickness=-1)
    img_OUTER = cv2.cvtColor(img_OUTER, cv2.COLOR_BGR2GRAY)
    #凹んでいる差分の画像を取得
    img_GRAYDIFF = cv2.bitwise_and(img_OUTER, img_INNER )
    return img_GRAYDIFF     #GRAY


def getDiffArea2(img_WK, factor):
    #近似輪郭の描画(1/5) 
    #img_WK = cv2.cvtColor(img_WK, cv2.COLOR_BGR2GRAY)   
    contours = getContours(img_WK)
    approxlines_inner = getApproxline(contours, factor)
    convexlines = getConvexLines(contours)
    img_GRAYDIFF = getDiffArea(img_WK, convexlines, approxlines_inner)
    return img_GRAYDIFF, convexlines, approxlines_inner

def drawDiffLineDel(img_GRAYDIFF):
    #ノイズ線の検出
    #img_GR = cv2.cvtColor(img_GRAYDIFF, cv2.COLOR_BGR2GRAY)   
    lines = cv2.HoughLinesP(img_GRAYDIFF, rho=1, theta=np.pi/360, threshold=30, minLineLength=3  , maxLineGap=60)
    drawDiffLineDelCom(img_GRAYDIFF, lines)

def drawDiffLineDel2(img_GRAYDIFF):
    #ノイズ線の検出
    lines = cv2.HoughLinesP(img_GRAYDIFF, rho=1, theta=np.pi/720, threshold=30, minLineLength=3  , maxLineGap=60)
    drawDiffLineDelCom(img_GRAYDIFF, lines)

def drawDiffLineDel3(img_GRAYDIFF):
    #ノイズ線の検出
    lines = cv2.HoughLinesP(img_GRAYDIFF, rho=1, theta=np.pi/360, threshold=5, minLineLength=4  , maxLineGap=60)
    drawDiffLineDelCom(img_GRAYDIFF, lines)

def drawDiffLineDelCom(img_GRAYDIFF, lines):
    #線の除去(黒で描画)
    #print("line:" + str(len(lines)))
    drawLines(img_GRAYDIFF,lines,(0,0,0),1)

def drawLines(img,lines,color,width):
    if lines is None: return
    if len(lines) == 0: return
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 赤線を引く
        cv2.line(img, (x1,y1), (x2,y2), color, width)

def drawDiffBoxDel(img_GRAYDIFF, fltPara, strDic):
    #細長領域の外接回転矩形を黒く塗りつぶす
    diffcontours = getContours(img_GRAYDIFF)
    for i in range(len(diffcontours)):
        rect = cv2.minAreaRect(diffcontours[i])
        #cont = diffcontours[i]
        #convex = cv2.convexHull(cont)           #凸包
        #print( "getDiffminAreaRect ", str(rect) )
        w, h = rect[1]
        #sqrwh = w * h 
        #ritu = convex / sqrwh 
        sw = False
        msg = ""
        #刃が反り返っている部分の対応
        #dicData = {1:100, 7:10, 5:20, 2:30 }
        dicData = eval(strDic)
        w = int(w * 10) / 10
        h = int(h * 10) / 10
        if w == 0 or h == 0:
            #長さだけ(0)の差分は消す
            sw = True
            ####msg = str(w) + " " + str(h)
        #elif w < 1 or h < 1:
        #elif w < 0.9 or h < 0.9:
        elif w < fltPara or h < fltPara:
            #１より小さい差分（輪郭の差分の誤差？）は消す
            sw = True
            ####msg = str(w) + " " + str(h) + " : " + str(w / h)
        else:
            if w > h:
                msg = str(w) + " " + str(h) + " : " + str(w / h)
            else:
                msg = str(h) + " " + str(w) + " : " + str(h / w)
            for k, v in dicData.items():
                if (w > k and ((h / w) > v)):
                    sw = True
                    break
                elif (h > k and ((w / h) > v)):
                    sw = True
                    break
        if sw is True:
            #基本輪郭と凸包輪郭の傾きの誤差は消す
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #黒く描画して領域を消す
            cv2.fillConvexPoly(img_GRAYDIFF,box,(0,0,0),8 ,0)
            ####if msg != "": print("del " + msg)
        else:
            #if msg != "": print("box " + msg)
            pass
    return img_GRAYDIFF

def getBoundingRect(contour):
    #外接矩形を描画
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h

def drawDiffRect(img_WK, x, y, w, h, color, offset):
    x = x - offset
    y = y - offset
    w = w + (offset * 2)
    h = h + (offset * 2)
    #cv2.rectangle(img_WK, (x, y), (x + w, y + h), (0, 0, 255), 4)
    cv2.rectangle(img_WK, (x, y), (x + w, y + h), color, 4)
    ####return img_WK
    #外接円
    #(x,y), radius = cv2.minEnclosingCircle(diffcontours[i])
    #center = (int(x),int(y))
    #radius = int(radius + 2 )
    #img_WK = cv2.circle(img_WK, center, radius, (0,0,255), 2)

def drawLightBlueArea(img_ORG, img_GRAYDIFF, offset, color):
    img_GRAYWK = img_GRAYDIFF.copy()
    img_hsv = cv2.cvtColor(img_ORG, cv2.COLOR_BGR2HSV)
    # h:色  s:彩度  v:明度
    #90水色 120青 45緑
    blue_min = np.array([60, 64, 128], np.uint8)
    blue_max = np.array([120, 255, 255], np.uint8)
    img_blue = cv2.inRange(img_hsv, blue_min, blue_max)
    conts = getContours(img_blue)
    for cont in conts:
        rect = getMinAreaRect(cont)
        x,y = rect[0]
        w,h = rect[1]
        w = w + (offset * 2)
        h = h + (offset * 2)
        wkrect = []
        wkrect = ((x, y),(w, h),rect[2])
        box = cv2.boxPoints(wkrect)
        box = np.int0(box)
        #黒く塗りつぶして領域を消す
        img_GRAYWK = cv2.fillConvexPoly(img_GRAYWK,box, color, 8 ,0)
    return img_GRAYWK


def getMinAreaRect(contour):
    rect = cv2.minAreaRect(contour)
    return rect

def drawMinAreaRect(img_WK, rect, color, offset, para):
    wkrect = []
    x,y = rect[0]
    w,h = rect[1]
    le = h
    wi = w
    if w > h:
        le = w
        wi = h
    le = int(le * para * 10 ) / 10
    wi = int(wi * para * 10 ) / 10
    wkrect = ((x,y),(w + (offset * 2), h + (offset * 2)),rect[2])
    box = cv2.boxPoints(wkrect)
    box = np.int0(box)
    cv2.drawContours(img_WK, [box],0, color, 4)
    if para == 0:
        pass
    else:
        msg = str(le) + "x" + str(wi)
        cv2.putText(img_WK,  msg, (int(x) + 20, int(y) + 20 ), cv2.FONT_HERSHEY_PLAIN,
                    1.5  , color, 1, cv2.LINE_AA)
    ####return img_WK
    #外接円
    #(x,y), radius = cv2.minEnclosingCircle(diffcontours[i])
    #center = (int(x),int(y))
    #radius = int(radius + 2 )
    #img_WK = cv2.circle(img_WK, center, radius, (0,0,255), 2)

#def fillMinAreaRect(img_WK, rect, color, offset):
#    wkrect = []
#    x,y = rect[0]
#    w,h = rect[1]
#    wkrect = ((x,y),(w + (offset * 2), h + (offset * 2)),rect[2])
#    box = cv2.boxPoints(wkrect)
#    box = np.int0(box)
#    # 太さマイナスは塗りつぶし
#    cv2.drawContours(img_WK, [box],0, color, -1)
#    return img_WK
#
# def getImgCut(img_WK, w, h, w1, h1):
#     MASK = getNullImageGray(img_WK)
#     MASK = cv2.rectangle(MASK, (w, h), (int(w1), int(h1)), (255,255,255), -1)
#     #MASK = cv2.bitwise_not(MASK)
#     img_WK = cv2.bitwise_and(img_WK, MASK)
#     return img_WK

def getAVG(img_WK, npAVG, fltF):
    if npAVG is None:
        #print("##########npavg")
        npAVG = img_WK.astype(np.float32)
    img_W = img_WK.astype(np.float32)
    cv2.accumulateWeighted(img_W, npAVG, 1.0 / float(fltF))
    img_AVG = npAVG.astype(np.uint8)
    ####_, img_AVG = cv2.threshold(img_AVG, 128, 255, cv2.THRESH_BINARY)
    #### しきい値
    #_, img_AVG = cv2.threshold(img_AVG, 104, 255, cv2.THRESH_BINARY)
    #_, img_AVG = cv2.threshold(img_AVG, 112, 255, cv2.THRESH_BINARY)
    _, img_AVG = cv2.threshold(img_AVG, 192, 255, cv2.THRESH_BINARY)
    return img_AVG, npAVG

def getDiffImage(img_TMP, baseimg):
    img_WK = cv2.absdiff(img_WK, baseimg)
    _, frameDelta = cv2.threshold(img_WK, 104, 255, cv2.THRESH_BINARY)
    return frameDelta

def drawCorner(img_TMP, img_CON, strcorner, val1, val2):
    #コーナ検出用の輪郭からコーナーを検出し描画
    if strcorner == "none":
        return img_TMP
    elif strcorner == "GFTT":
        ##近似輪郭から特徴抽出
        if val1 == 0:
            return img_TMP
        img_WKCON = cv2.cvtColor(img_CON,cv2.COLOR_BGR2GRAY)
        corners = getcorners(img_WKCON, strcorner, val1, val2)
        if corners is not None:
            if len(corners) != 0:
                corners = np.int0(corners)
                for i in corners:
                    x,y = i.ravel()
                    ##特徴点に円を描画
                    cv2.circle(img_WK,(x,y),30,(0,0,255),5)
    elif strcorner == "ORB":
        detector = cv2.ORB_create()
        keypoints = detector.detect(img_CON)
        img_WK = keypointpaint(img_TMP, keypoints)
    elif strcorner == "AKAZE":
        detector = cv2.AKAZE_create()
        keypoints = detector.detect(img_CON)
        img_WK = keypointpaint(img_TMP, keypoints)
    elif strcorner == "BRIST":
        detector = cv2.BRIST_create()
        keypoints = detector.detect(img_CON)
        img_WK = keypointpaint(img_TMP, keypoints)
    return img_WK



def keypointpaint(img_WK, keypoints):
    for keyp in keypoints:
        ##特徴点に円を描画
        cv2.circle(img_WK,(int(keyp.pt[0]),int(keyp.pt[1])),30,(0,0,255),5)
    #return img_WK

def getcorners(img_WK, v, val1, val2):
    if v == "none": return []
    #画像認識角を探す
    #100個 ,最低限の質,2つのコーナのユークリッド距離
    if v == "GFTT":
        print(val1, val2)
        corners = cv2.goodFeaturesToTrack(img_WK,100, (val1/ 100) ,val2)
    else:
        corners = []
        #return img_WK
        ##corners = cv2.goodFeaturesToTrack(img_WK,100,0.31,30)
    return corners

def detectkeypoint():
    # 特徴量検出機を作成し解析
    detector = cv2.FastFeatureDetector_create()
    detector.setNonmaxSuppression(False)
    keypoints = detector.detect(cut)
    # 画像への特徴点の書き込み
    cv2.drawKeypoints(cut, keypoints, None)


def Resize(img_TMP, x, y):
    #height = img_WK.shape[0]
    #idth = img_WK.shape[1]
    #INTERPOLATIONS = [cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
    INTERPOLATIONS = cv2.INTER_AREA
    img_WK = cv2.resize(img_TMP, dsize=(int(x),int(y)), interpolation=cv2.INTER_AREA)
    return img_WK

def PointResize(npAry, fx, fy):
    ary1 = np.array([fx, fy])
    npAry = npAry * ary1
    return npAry

def DrawText(img_WK, strText, x,y):
    fontsize = 3
    fonttick = 4
    cv2.putText(img_WK, strText, (x, y), cv2.FONT_HERSHEY_PLAIN,
                fontsize    , (0,0,0), fonttick + 3 , cv2.LINE_AA)
    cv2.putText(img_WK, strText, (x, y), cv2.FONT_HERSHEY_PLAIN,
                fontsize    , (247,247,247), fonttick, cv2.LINE_AA)
    #####return img_WK

