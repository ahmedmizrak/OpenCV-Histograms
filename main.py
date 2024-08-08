import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
#####step 1#####
img = cv2.imread(os.path.join("image.png"))
assert img is not None,"Error! Check os.path.join()"
#####step 2#####
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#####step 3#####
cv2.imshow("Gray Image", img_gray)

color = ('b','g','r')

#####step 4-- step 5#####
for i,col in enumerate(color):
    hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
    plt.plot(hist,color = col)
    plt.xlim([0,256])
plt.show()
#####step 6--step7--step8#####
equ = cv2.equalizeHist(img_gray)
res = np.hstack((img_gray,equ))
cv2.imshow("1-origin-2-Equalize",res)
cv2.imwrite('res.png',res)

for i,col in enumerate(color):
    hist = cv2.calcHist([res],[0],None,[256],[0,256])
    plt.plot(hist,color= col)
    plt.xlim([0,256])
plt.show()

'''********    Part 2    *********'''
###Step 1 ###
img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

###Step 2 ###
hist, bins = np.histogram(img_hsv.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.savefig("step2.png")
#plt.show()

img_turning_back_rgb = cv2.imread(os.path.join("step2.png"))
img_rgb = cv2.cvtColor(img_turning_back_rgb,cv2.COLOR_HSV2RGB)

#cv2.imshow("rgb_image",img_rgb)
#step 3
color2 = ('r', 'g', 'b')
plt.figure(figsize=(10, 5))

for i, col in enumerate(color2):
    histr = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])

plt.title('Histogram for RGB channels')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.legend(['Red', 'Green', 'Blue'])
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()