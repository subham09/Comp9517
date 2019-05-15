import cv2
import numpy
import sys

a = str(sys.argv[2])
b = str(sys.argv[4])
c = str(sys.argv[6])
n = int(b)

img = cv2.imread(a,0)

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

##for i in range(0, 2000):
##    l1.append([])

for i in range(0, len(img)):
    for j in range(0, len(img[i])):
        maxi = []
        n1 = 999999
       
        n3 = 999999
        
        if img[i][j] == 0:
            if j:
                if lebel[i][j-1] < 999999:
                    n1 = lebel[i][j-1]
                    
            if i:
                if lebel[i-1][j] < 999999:
                    n3 = lebel[i-1][j]

            maxi.append(n1)
        
            maxi.append(n3)
            
            if n1 == 999999 and n3 == 999999:
                lebel[i][j] = counter
                counter += 1
            elif (999999 in maxi and min(maxi) != 999999):
                lebel[i][j] = min(maxi)
            else:
                lebel[i][j] = min(maxi)
                if (n1 != 999999 and n3 != 999999) and n1 != n3:
                    for k in range(0, len(l1)):
                        if len(l1[k]) == 0:
                            l1[k].append(n1)
                            l1[k].append(n3)
                            l1.append([])
                            break
                        if n1 in l1[k] or n3 in l1[k]:
                            if n1 in l1[k] and n3 in l1[k]:
                                pass
                            if n1 in l1[k] and n3 not in l1[k]:
                                l1[k].append(n3)
                            if n3 in l1[k] and n1 not in l1[k]:
                                l1[k].append(n1)
                            break
                        


for i in range(0,len(img)):
    for j in range(0, len(img[i])):
        if lebel[i][j] != 999999:
            for k in range(0, len(l1)):
                if lebel[i][j] in l1[k]:
                    lebel[i][j] = min(l1[k])
                    break

#print(lebel[:2])
d = {}

for i in range(1, counter+1):
    max1 = 0
    for j in range(0, len(lebel)):
        max1 += lebel[j].count(i)
    d[i] = max1


counter2 = 0
color = 100
key_list = []
for key,value in d.iteritems():
    if value > n:
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
            
            cv2.imwrite(c, img)
print(counter2)
