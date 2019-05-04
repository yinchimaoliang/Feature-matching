import cv2 as cv
import math
import numpy as np


IMAGE_PATH1 = './images/a.jpg'
IMAGE_PATH2 = './images/b.jpg'
RADIUS = 20
THRESHOLD_LOW = 100
THRESHOLD_HIGH = 200

class main():
    def __init__(self):
        self.img1 = cv.imread(IMAGE_PATH1)
        self.img1 = cv.GaussianBlur(self.img1,(3,3),1.5)
        self.img2 = cv.imread(IMAGE_PATH2)
        self.img2 = cv.GaussianBlur(self.img2,(3,3),1.5)
        self.img1_height = self.img1.shape[0]
        self.img1_width = self.img1.shape[1]
        self.img2_height = self.img2.shape[0]
        self.img2_width = self.img2.shape[1]
        self.img1 = cv.copyMakeBorder(self.img1,RADIUS,RADIUS,RADIUS,RADIUS,cv.BORDER_REPLICATE)
        # print(self.img1.shape)
        self.img2 = cv.copyMakeBorder(self.img2, RADIUS, RADIUS, RADIUS, RADIUS, cv.BORDER_REPLICATE)

        self.img1_features = [None for i in range(self.img1_height * self.img1_width)]
        self.img2_features = [None for i in range(self.img2_height * self.img2_height)]

        self.matching_table = []

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
        print(cand)
        self.feature_num = len(cand)
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
        count = 0
        for i in range(len(self.img1_features)):
            if self.img1_features[i].count('1') + 2 * self.img1_features[i].count('2') > self.feature_num:
                count += 1
            if i % 1000 == 0:
                print("finish %d points" % (i))
            if self.img1_features[i].count('1') + 2 * self.img1_features[i].count('2') > self.feature_num  and self.img1_features[i] in self.img2_features:
                img2_index = self.img2_features.index(self.img1_features[i])
                self.matching_table.append([[int(i / self.img1_width),i - int(i / self.img1_width) * self.img1_width],[int(img2_index / self.img2_width),img2_index - int(img2_index / self.img2_width) * self.img2_width]])
                self.img1_sims.append([int(i / self.img1_width),i - int(i / self.img1_width) * self.img1_width])
        print(len(self.img1_sims))
        print(count)

    def show(self):
        img1 = cv.imread(IMAGE_PATH1)
        img2 = cv.imread(IMAGE_PATH2)
        img_matches = np.empty(
            (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)

        img_matches[:img1.shape[0], :img1.shape[1]] = img1
        img_matches[:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

        for i in self.matching_table:
            cv.circle(img_matches,(i[0][1],i[0][0]),3,(0,255,255))
            cv.circle(img_matches,(img1.shape[1] + i[1][1],i[1][0]),3,(0,255,255))
            # cv.line(img_matches,(i[0][1],i[0][0]),(img1.shape[1] + i[1][1],i[1][0]),(0,0,255))
        cv.imshow("result",img_matches)
        cv.waitKey()


    def mainMethod(self):
        # print(self.findCircle(5,6,RADIUS))
        # cv.imshow("test",self.img2)
        # cv.waitKey()
        self.myORB(1,RADIUS)
        self.myORB(2,RADIUS)
        self.matching()
        self.show()

if __name__ == '__main__':
    a = main()
    a.mainMethod()