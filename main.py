import cv2 as cv
import math
import numpy as np


IMAGE_PATH1 = './images/d.jpg'
IMAGE_PATH2 = './images/e.jpg'
RADIUS = 4
THRESHOLD = 100
CONTINUS = 0.7
GAUSSIAN_RADIUS = 5
GAUSSIAN_SIGMA = 2

class main():
    def __init__(self):
        self.img1 = cv.imread(IMAGE_PATH1)
        self.img1 = cv.GaussianBlur(self.img1,(GAUSSIAN_RADIUS,GAUSSIAN_RADIUS),GAUSSIAN_SIGMA)
        self.img2 = cv.imread(IMAGE_PATH2)
        self.img2 = cv.GaussianBlur(self.img2,(GAUSSIAN_RADIUS,GAUSSIAN_RADIUS),GAUSSIAN_SIGMA)
        self.img1_height = self.img1.shape[0]
        self.img1_width = self.img1.shape[1]
        self.img2_height = self.img2.shape[0]
        self.img2_width = self.img2.shape[1]
        self.img1 = cv.copyMakeBorder(self.img1,RADIUS,RADIUS,RADIUS,RADIUS,cv.BORDER_REPLICATE)
        # print(self.img1.shape)
        self.img2 = cv.copyMakeBorder(self.img2, RADIUS, RADIUS, RADIUS, RADIUS, cv.BORDER_REPLICATE)

        self.img1_points = []
        self.img2_points = []
        self.img1_features = []
        self.img2_features = []

        self.matching_table = []

    def findCircle(self,x,y,r):
        result = []
        for i in range(x - r,x + r + 1):
            for j in range(y - r,y + r + 1):
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
        forbidden_list = []
        for i in range(height):
            for j in range(width):
                # print([i,j],forbidden_list)
                if i * width + j in forbidden_list:
                    continue
                feature = ''
                max_num = 0
                max_val = 0
                for k in range(len(cand)):
                    val = abs(img[i + r][j + r] - img[cand[k][0] + i][cand[k][1] + j])
                    if val < THRESHOLD:
                        feature += '0'
                    else:
                        feature += '1'
                    if val > max_val:
                        max_val = val
                        max_num = k
                feature = feature[max_num:] + feature[:max_num]
                # 如果连续是1的个数到达阈值，则将其定义为角点
                if img_no == 1:
                    constant = 0
                    for k in range(2 * self.feature_num):
                        if constant > CONTINUS * self.feature_num:
                            self.img1_points.append([i,j])
                            self.img1_features.append(int(feature))
                            #特征点周围不再设置特征点
                            for m in range(-RADIUS,RADIUS + 1):
                                for n in range(-RADIUS,RADIUS + 1):
                                    forbidden_list.append((i + m) * width + j + n)
                            break


                        if feature[k % self.feature_num] == '1':
                            constant += 1
                        else:
                            constant = 0
                else:
                    constant = 0
                    for k in range(1 * self.feature_num):
                        if constant > CONTINUS * self.feature_num:
                            self.img2_points.append([i, j])
                            self.img2_features.append(int(feature))
                            for m in range(RADIUS + 1):
                                for n in range(RADIUS + 1):
                                    forbidden_list.append((i + m) * width + j + n)
                            break

                        if feature[k % self.feature_num] == '1':
                            constant += 1
                        else:
                            constant = 0

    #进行特征匹配
    def matching(self):

        img1_points = self.img1_points[:]
        img1_features = self.img1_features[:]
        img2_points = self.img2_points[:]
        img2_features = self.img2_features[:]
        print(len(img1_features),len(img1_points))
        i = 0
        #以img1为基础进行匹配
        while i < len(img1_features):
            for j in range(len(img2_features)):
                if img1_features[i] ^ img2_features[j] == 0:
                    #匹配到一点后将其丢弃
                    self.matching_table.append([img1_points[i],img2_points[j]])
                    img2_points.pop(j)
                    img2_features.pop(j)
                    img1_points.pop(i)
                    img1_features.pop(i)
                    i -= 1
                    break

            i += 1
        i = 0
        # 以img1为基础进行匹配
        while i < len(img2_features):
            for j in range(len(img1_features)):
                if img2_features[i] ^ img1_features[j] == 0:
                    # 匹配到一点后将其丢弃
                    self.matching_table.append([img2_points[i],img1_points[j]])
                    img1_points.pop(j)
                    img1_features.pop(j)
                    img2_points.pop(i)
                    img2_features.pop(i)
                    i -= 1
                    break

            i += 1
        print(len(self.matching_table))
        # self.img1_sims = []
        # count = 0
        # for i in range(len(self.img1_features)):
        #     if self.img1_features[i].count('1') > 0.9 * self.feature_num:
        #         count += 1
        #     if i % 1000 == 0:
        #         print("finish %d points" % (i))
        #     if self.img1_features[i].count('1') > self.feature_num * 0.9 and self.img1_features[i] in  self.img2_features:
        #         img2_index = self.img2_features.index(self.img1_features[i])
        #         self.matching_table.append([[int(i / self.img1_width),i - int(i / self.img1_width) * self.img1_width],[int(img2_index / self.img2_width),img2_index - int(img2_index / self.img2_width) * self.img2_width]])
        #         self.img1_sims.append([int(i / self.img1_width),i - int(i / self.img1_width) * self.img1_width])
        # print(len(self.img1_sims))
        # print(count)

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
            cv.line(img_matches,(i[0][1],i[0][0]),(img1.shape[1] + i[1][1],i[1][0]),(0,0,255))
        cv.imshow("result",img_matches)
        cv.waitKey()

    def temp(self):
        color_table = []

        for i in range(0,256,10):
            for j in range(0,256,10):
                for k in range(0,256,10):
                    color_table.append([i,j,k])

        color_num = len(color_table)
        img1 = cv.imread(IMAGE_PATH1)
        img2 = cv.imread(IMAGE_PATH2)
        img_matches = np.empty(
            (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
        img_matches[:img1.shape[0], :img1.shape[1]] = img1
        img_matches[:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

        for i in range(len(self.img1_features)):
            cv.circle(img_matches,(self.img1_points[i][1],self.img1_points[i][0]),3,color_table[i % color_num])
        for i in range(len(self.img2_features)):
            cv.circle(img_matches,(img1.shape[1] + self.img2_points[i][1] , self.img2_points[i][0]),3,color_table[i % color_num])

        for i in range(len(self.matching_table)):
            cv.line(img_matches,(self.matching_table[i][0][1],self.matching_table[i][0][0]),(img1.shape[1] + self.matching_table[i][1][1],self.matching_table[i][1][0]),color_table[i % color_num])
        cv.imshow("result",img_matches)
        cv.waitKey()

    def mainMethod(self):
        # print(self.findCircle(5,6,RADIUS))
        # cv.imshow("test",self.img2)
        # cv.waitKey()
        self.myORB(1,RADIUS)
        self.myORB(2,RADIUS)
        print(len(self.img1_features))
        self.matching()
        self.temp()
        # self.matching()
        # self.show()

if __name__ == '__main__':
    a = main()
    a.mainMethod()