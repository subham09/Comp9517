import cv2

img = cv2.imread("rectified_depth.png")

# Calculating the width and height of the image
img_y = len(img)
img_x = len(img[0])

white_left = 0
white_right = 0
width_left = []
width_right = []

#Calculating the white spaces on the left and
#right side of the obstacle

for i in img:
    white_left = 0
    white_right = 0
    for j in i:
        if j[0] == 255:
            white_left = white_left + 1
        elif j[0] == 0:
            break
    width_left.append(white_left)
    for j in reversed(i):
        if j[0] == 255:
            white_right = white_right + 1
        elif j[0] == 0:
            break
    width_right.append(white_right)
print "\nApparent free gap to the left of obstacle is", min(width_left),"Pixels"
print "\nApparent free gap to the right of obstacle is", min(width_right),"Pixels\n"
            
