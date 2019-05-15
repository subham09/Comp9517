# import the necessary packages
#from PIL import Image
import pytesseract
import cv2
import string
import numpy as np

# load the example image and convert it to grayscale
print 'Loading image...'
image = cv2.imread('rectified_text.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print 'Applying pytesseract to recognise text from image...'
text = pytesseract.image_to_string(gray)
text =text.encode("utf-8")
for i in string.punctuation:
        text=text.replace(i,'')
text = text.split()
text_final =''
print('\nText detected in the image is:\n')
for i in range(len(text)-1):
        text_final = text_final + text[i] + ' '
print text_final

