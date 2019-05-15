import cv2
import numpy as np
import warnings
import sys
import random

print 'Loading obstacle.png image...'
img = cv2.imread("obstacle.png")
ret,img = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
print 'Applying GaussianBlur...'
blur = cv2.GaussianBlur(img, (5,5), 0)

label=[]
labels=[]
lab=1
tree = [[]]
print 'Removing noise using a version of connected components...'
for n,i in enumerate(img):
    label=[]
    for m,j in enumerate(i):
            label.append(0)
    labels.append(label)

for n,i in enumerate(img):
    for m,j in enumerate(i):
        left,top= -1,-1
        if j.all() == 0:
            if n:
                if labels[n-1][m]:
                    left=labels[n-1][m]
            if m:
                if labels[n][m-1]:
                    top=labels[n][m-1]
            if (top == -1 and left == -1):
                labels[n][m]=lab
                lab=lab+1
            elif (top == -1 or left == -1):
                labels[n][m]=(max(top,left))
            else:
                labels[n][m]=(min(top,left))
                if top!=left:
                    count = 0
                    while True:
                        if len(tree[count])==0:
                            tree[count].extend([top,left])
                            tree.append([])
                            break
                        if top in tree[count] or left in tree[count]:
                            if top in tree[count] and left not in tree[count]:
                                tree[count].append(left)
                            elif left in tree[count] and top not in tree[count]:
                                tree[count].append(top)
                            break
                        else:
                            count = count + 1

for n,i in enumerate(img):
    for m,j in enumerate(i):
        if labels[n][m]!=0:
            for o,k in enumerate(tree):
                if labels[n][m] in k:
                    labels[n][m]=min(k)
                    break
maxi=0
counts=0
for i in labels:
    if max(i) > maxi:
        maxi=max(i)
count={}
for i in range(1,maxi+1):
    count[i]=0
for i in labels:
    for j in i:
        if j !=0:
            count[j]=count[j]+1
modules=[]
for i in range(1,maxi+1):
    if count[i] > 1000:
        counts=counts+1
        if i not in modules:
            modules.append(i)
temp=0
new=[]
news=[]
for k,i in enumerate(img):
    new=[]
    for l,j in enumerate(i):
        if labels[k][l] in modules:
            new.append([0,0,0])
        else:
            new.append([255,255,255])
    news.append(new)
dt = np.dtype('f8')
print 'Image saved as rectified_depth.png...'
new=np.array(news,dtype=dt)                
cv2.imwrite("rectified_depth.png",new)
