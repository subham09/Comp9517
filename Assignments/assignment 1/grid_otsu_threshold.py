import cv2
import sys
import numpy as np
import warnings, os

warnings.filterwarnings("ignore")

a = str(sys.argv[2])
n1 = str(sys.argv[3])
b = str(sys.argv[5])
n = int(n1)
img = cv2.imread(a,0)
height, width = img.shape[:2]
#print("height", height, "width", width)
x = width//n #whatever the geid size is will replace the 2
#print("x",x)
z = width//n # 636/5 = 127
y = 0
counter = 0
width1 = width
for k in range(0, n):

    #print("value of k", k, "value of y", y, "value of x", x)
    s = img[:int(height), y:int(x)]  # 1st time it will go from 0 - 127, 127-254 and so on
##    if k == 1:
##        cv2.imwrite("1th.jpg", s)
    #print("width", width1, "y", y, "x", x)
    #print("s value of i = ",i)
    #print(s)
    #-------------------------------
    blur = cv2.GaussianBlur(s,(5,5),0)

    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([s],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.max()
    #print("hist max",hist.max())
    Q = hist_norm.cumsum()

    bins = np.arange(256)
    #print("bins", bins)

    fn_min = np.inf
    thresh = 0

    for i in xrange(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        b1,b2 = np.hsplit(bins,[i]) # weights

        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

        # calculates the minimization function
        fn = v1*q1 + v2*q2
        #print(v1*q1)
        if fn < fn_min:
            fn_min = fn
            thresh = i
    #print("thresh",thresh)

    for i in range (0, len(s)):
        for j in range(0, len(s[i])):
            if (s[i][j] > thresh):
                s[i][j] = 255
            else:
                s[i][j] = 0
    #------------------------
    if counter == n - 1:
        #print("jaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        #cv2.imwrite("4th.jpg", s)
        #y+=z
        x += width1
        #print("my values",x,y)
        s = img[:int(height), y:int(x)]
    else:
        width1 -= z
        y+=z
        x+=z

    if counter == 0:
        cv2.imwrite(b, s)
        
    else:
        something = cv2.imread(b,0)
        h1, w1 = something.shape[:2]
        h2, w2 = s.shape[:2]
        #print("h2 w2 counter",h2,w2, counter)
        vis = np.zeros((max(h1, h2), w1+w2), np.uint8)  #concatenating each part
        vis[:h1, :w1] = something
        vis[:h2, w1:w1+w2] = s

        #vis = np.concatenate((something, s), axis=1)
        #print(counter)
        cv2.imwrite(b, vis)
    counter+=1
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(vis, cv2.MORPH_OPEN, kernel)
cv2.imwrite("edited.jpg", opening)
