import cv2 as cv
import math
import numpy as np


IMAGE_PATH1 = './images/g.jpeg'
IMAGE_PATH2 = './images/f.jpg'
SIGMA = 1.6
S = 4
K = 0.5

class main():
    def __init__(self):
        self.img1 = cv.imread(IMAGE_PATH1,0)
        self.img2 = cv.imread(IMAGE_PATH2,0)

    def genDoG(self,img):
        height = len(img)
        width = len(img[0])
        group_num = int(math.log(min(width,height),2)) - S
        height *= 2
        width *= 2
        weight = 1
        pyramid = []
        DoG_pyramid = []
        for i in range(group_num):
            img = cv.resize(img,(width,height))
            group = []
            for j in range(S):

                t = cv.GaussianBlur(img,(int(height / 2) * 2 + 1,int(width / 2) * 2  + 1),weight * K ** j * SIGMA)

                group.append(t)
            height = int(height / 2)
            width = int(width / 2)
            # weight *= 2
            pyramid.append(group)

        for i in range(group_num):
            group = []
            for j in range(S - 1):
                t = pyramid[i][j + 1] - pyramid[i][j]
                cv.imshow("test", t)
                cv.waitKey()
                group.append(t)
            DoG_pyramid.append(group)



    # #显示函数
    # def show(self):
    #     color_table = []
    #
    #     for i in range(0,256,10):
    #         for j in range(0,256,10):
    #             for k in range(0,256,10):
    #                 color_table.append([i,j,k])
    #
    #     color_num = len(color_table)
    #     img1 = cv.imread(IMAGE_PATH1)
    #     img2 = cv.imread(IMAGE_PATH2)
    #     #将两张图放在一个框中
    #     img_matches = np.empty(
    #         (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    #     img_matches[:img1.shape[0], :img1.shape[1]] = img1
    #     img_matches[:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2
    #
    #     for i in range(len(self.img1_features)):
    #         cv.circle(img_matches,(self.img1_points[i][1],self.img1_points[i][0]),3,color_table[i % color_num])
    #     for i in range(len(self.img2_features)):
    #         cv.circle(img_matches,(img1.shape[1] + self.img2_points[i][1] , self.img2_points[i][0]),3,color_table[i % color_num])
    #
    #     for i in range(len(self.matching_table)):
    #         cv.circle(img_matches,(self.matching_table[i][0][1],self.matching_table[i][0][0]),3,color_table[i % color_num])
    #         cv.circle(img_matches,(img1.shape[1] + self.matching_table[i][1][1],self.matching_table[i][1][0]),3,color_table[i % color_num])
    #         cv.line(img_matches,(self.matching_table[i][0][1],self.matching_table[i][0][0]),(img1.shape[1] + self.matching_table[i][1][1],self.matching_table[i][1][0]),color_table[i % color_num])
    #     cv.imshow("result",img_matches)
    #     cv.waitKey()

    def mainMethod(self):
        self.genDoG(self.img1)
        # self.show()

if __name__ == '__main__':
    a = main()
    a.mainMethod()