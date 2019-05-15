import cv2
import sys
import numpy as np
import warnings, os

warnings.filterwarnings("ignore")

a = str(sys.argv[2])
b = str(sys.argv[4])

img = cv2.imread(a,0)

blur = cv2.GaussianBlur(img,(5,5),0)

# normalized_histogram, cumulative distribution function
hist = cv2.calcHist([img],[0],None,[256],[0,256])

hist_norm = hist.ravel()/hist.max()

Q = hist_norm.cumsum()

bins = np.arange(256)

fn_min = np.inf
thresh = 0

for i in xrange(1,256):
    
    # probabilities
    p1,p2 = np.hsplit(hist_norm,[i])
    
    # cum sum of classes
    q1,q2 = Q[i],Q[255]-Q[i]

    # weights
    b1,b2 = np.hsplit(bins,[i]) 

    # means and variances
    m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

    # minimization function
    fn = v1*q1 + v2*q2

    if fn < fn_min:
        fn_min = fn
        thresh = i


for i in range (0, len(img)):
    for j in range(0, len(img[i])):
        if (img[i][j] > thresh):
            img[i][j] = 255
        else:
            img[i][j] = 0

cv2.imwrite(b, img)
if len(sys.argv) == 6:
    print(thresh)

