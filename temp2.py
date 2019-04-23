import cv2 as cv
import numpy as np
IMAGE_PATH1 = './images/a.jpg'
IMAGE_PATH2 = './images/c.jpg'


img1 = cv.imread(IMAGE_PATH1)
img2 = cv.imread(IMAGE_PATH2)

key1 = cv.KeyPoint(0,0,1)
key2 = cv.KeyPoint(0,0,1)


img_matches = np.empty((max(img1.shape[0], img2.shape[0]),img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)

print(img2[199,132])
print(img1.shape)
print(img2.shape)
print(img_matches.shape)
img_matches[:img1.shape[0],:img1.shape[1]] = img1
img_matches[:img2.shape[0],img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

cv.circle(img_matches,(1,100),10,(0,0,255))
cv.circle(img_matches,(img1.shape[0] + 10,10),10,(0,0,255))
cv.line(img_matches,(10,10),(img1.shape[0] + 10,10),(0,0,255))
cv.imshow("test",img_matches)
cv.waitKey()