import cv2
import sys
import numpy as np

a = str(sys.argv[2])
b = str(sys.argv[4])

img = cv2.imread(a)


median = cv2.medianBlur(img,5)
median = cv2.bilateralFilter(median,9,75,75)
cv2.imwrite(b,median)
