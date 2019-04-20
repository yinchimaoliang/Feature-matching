import cv2 as cv
import math
import numpy as np


IMAGE_PATH1 = './images/a.jpg'
IMAGE_PATH2 = './images/b.jpg'
RADIUS = 5



class main():
    def __init__(self):
        self.img1 = cv.imread(IMAGE_PATH1)
        self.img2 = cv.imread(IMAGE_PATH2)
        self.img1 = cv.copyMakeBorder(self.img1,RADIUS,RADIUS,RADIUS,RADIUS,cv.BORDER_REPLICATE)
        self.img2 = cv.copyMakeBorder(self.img2, RADIUS, RADIUS, RADIUS, RADIUS, cv.BORDER_REPLICATE)
        print(self.img1.shape)

    def findCircle(self,x,y,r):
        result = []
        for i in range(x - r,x + r):
            for j in range(y - r,y + r):
                dist = math.sqrt((i - x) ** 2 + (j - y) ** 2)
                if dist >= r and dist < r + 1:
                    result.append([i,j])

        return result



    def myORB(self):

        pass

    def mainMethod(self):
        print(self.findCircle(5,6,RADIUS))
        # cv.imshow("test",self.img2)
        # cv.waitKey()

if __name__ == '__main__':
    a = main()
    a.mainMethod()