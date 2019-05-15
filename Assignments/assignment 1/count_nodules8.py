import cv2
import numpy
import sys

a = str(sys.argv[2])
b = str(sys.argv[4])
c = int(b)
e = str(sys.argv[6])
img = cv2.imread(a, 0)

height, width = img.shape[:2]

lebel = []
for i in range(len(img)):
    lvl = []
    for j in range(len(img[i])):
        lvl.append(999999)
    lebel.append(lvl)
counter = 1
maxi = []
l1=[[]]
for i in range(0, 2000):
    l1.append([])
#print("lebel --->",lebel[:2])
for i in range(0, len(img)):
    for j in range(0, len(img[i])):
        maxi = []
        n1 = 999999
        n2 = 999999
        n3 = 999999
        n4 = 999999
        if img[i][j] == 0:
            #lebel[i][j] = counter
            #counter += 1
            if j:
                if lebel[i][j-1] < 999999:
                    n1 = lebel[i][j-1]
                    maxi.append(n1)
            if i and j < len(img)-2:
                if lebel[i-1][j+1] < 999999:
                    n4 = lebel[i-1][j+1]
                    maxi.append(n4)
            if i:
                if lebel[i-1][j] < 999999:
                    n3 = lebel[i-1][j]
                    maxi.append(n3)
            if j and i:
                if lebel[i-1][j-1] < 999999:
                    n2 = lebel[i-1][j-1]
                    maxi.append(n2)

            if n1 == 999999 and n3 == 999999 and n2 == 999999 and n4 == 999999:
                lebel[i][j] = counter
                counter += 1
            else:
                
                lebel[i][j] = min(maxi)
                for k in range(0, len(l1)):
                    if len(l1[k]) == 0:
                        l1[k].extend(set(maxi))
                        #l1.append([])
                        break
                    if len(set(l1[k]).intersection(maxi)):
                        l1[k] = list(set(l1[k]).union(set(maxi)))
                        break
#print(l1[:1])
                        

for i in range(0,len(img)):
    for j in range(0, len(img[i])):
        if lebel[i][j] != 999999:
            for k in range(0, len(l1)):
                if lebel[i][j] in l1[k]:
                    lebel[i][j] = min(l1[k])
                    break

#print(lebel[-1:])
d = {}

for i in range(1, counter+1):
    max1 = 0
    for j in range(0, len(lebel)):
        max1 += lebel[j].count(i)
    d[i] = max1


#print(d)
counter2 = 0
color = 100
key_list = []
for key,value in d.iteritems():
    if value > c:
        counter2 += 1
        #print(key, "value-->", value)
        key_list.append(key)
        for i in range(0, len(lebel)):
            for j in range(0, len(lebel[i])):
                if lebel[i][j] == key:
                    img[i][j] = color
                else:
                    if key not in key_list:
                        img[i][j] = 255
        color = color + 2
        if len(sys.argv) == 7:
            #print("hello")
            cv2.imwrite(e, img)
print(counter2)
                        
