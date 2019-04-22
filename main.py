import cv2 as cv
import math
import numpy as np


IMAGE_PATH1 = './images/a.jpg'
IMAGE_PATH2 = './images/c.jpg'
RADIUS = 15
THRESHOLD_LOW = 30
THRESHOLD_HIGH = 60

class main():
    def __init__(self):
        self.img1 = cv.imread(IMAGE_PATH1)
        self.img2 = cv.imread(IMAGE_PATH2)
        self.img1_height = self.img1.shape[0]
        self.img1_width = self.img1.shape[1]
        self.img2_height = self.img2.shape[0]
        self.img2_width = self.img2.shape[1]
        self.img1 = cv.copyMakeBorder(self.img1,RADIUS,RADIUS,RADIUS,RADIUS,cv.BORDER_REPLICATE)
        # print(self.img1.shape)
        self.img2 = cv.copyMakeBorder(self.img2, RADIUS, RADIUS, RADIUS, RADIUS, cv.BORDER_REPLICATE)

        self.img1_features = [None for i in range(self.img1_height * self.img1_width)]
        self.img2_features = [None for i in range(self.img2_height * self.img2_height)]

    def findCircle(self,x,y,r):
        result = []
        for i in range(x - r,x + r):
            for j in range(y - r,y + r):
                dist = math.sqrt((i - x) ** 2 + (j - y) ** 2)
                if dist >= r and dist < r + 1:
                    result.append([i,j])

        return result



    def myORB(self,img_no,r):
        if img_no == 1:
            img = self.img1
            height = self.img1_height
            width = self.img1_width
        else:
            img = self.img2
            height = self.img2_height
            width = self.img2_width
        img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
        # cv.imshow("test",img)
        print(img.shape)
        cv.waitKey()
        cand = self.findCircle(r, r, r)
        for i in range(height):
            for j in range(width):
                feature = ''
                max_num = 0
                max_val = 0
                for k in range(len(cand)):
                    val = abs(img[i + r][j + r] - img[cand[k][0] + i][cand[k][1] + j])
                    if val < THRESHOLD_LOW:
                        feature += '0'
                    else:
                        if val < THRESHOLD_HIGH:
                            feature += '1'
                        else:
                            feature += '2'
                    if val > max_val:
                        max_val = val
                        max_num = k
                feature = feature[max_num:] + feature[:max_num]
                # print(feature)
                if img_no == 1:
                    self.img1_features[i * width+ j] = feature
                else:
                    self.img2_features[i * width+ j] = feature


    def matching(self):
        self.img1_sims = []
        for i in range(len(self.img1_features)):
            if i % 100 == 0:
                print("finish %d points" % (i))
            if self.img1_features[i] in self.img2_features:
                self.img1_sims.append([int(i / self.img1_width),i - int(i / self.img1_width) * self.img1_width])
        print(self.img1_sims)
    def mainMethod(self):
        # print(self.findCircle(5,6,RADIUS))
        # cv.imshow("test",self.img2)
        # cv.waitKey()
        self.myORB(1,RADIUS)
        self.myORB(2,RADIUS)
        self.matching()

if __name__ == '__main__':
    a = main()
    a.mainMethod()