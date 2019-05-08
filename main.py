import cv2 as cv
import math
import numpy as np


IMAGE_PATH1 = './images/g.jpeg'
IMAGE_PATH2 = './images/f.jpg'
SIGMA = 1.6
S = 4
K = 2 ** (1 / S)

class main():
    def __init__(self):
        self.img1 = cv.imread(IMAGE_PATH1,0)
        self.img2 = cv.imread(IMAGE_PATH2,0)

    def genDoG(self,img):
        K = 2
        height = len(img)
        width = len(img[0])
        group_num = int(math.log(min(width,height),2)) - S
        height *= 2
        width *= 2
        # weight = 1
        pyramid = []
        DoG_pyramid = []
        for i in range(group_num):
            img = cv.resize(img,(width,height))
            group = []
            for j in range(S):

                t = cv.GaussianBlur(img,(int(width / 2) * 2  + 1,int(height / 2) * 2 + 1),2 ** i * K ** j * SIGMA)

                group.append(t)
            height = int(height / 2)
            width = int(width / 2)
            # weight *= 2
            pyramid.append(group)

        for i in range(group_num):
            group = []
            for j in range(S - 1):
                t = pyramid[i][j + 1] - pyramid[i][j]
                # cv.imshow("test", t)
                # cv.waitKey()
                group.append(t)
            DoG_pyramid.append(group)

        return group_num,DoG_pyramid
    def findEdge(self,img):
        group_num, DoG_pyramid = self.genDoG(img)
        height = len(img)
        width = len(img[0])
        edge = [[0 for i in range(width)] for j in range(height)]

        mid_row = int(height / 2)
        mid_col = int(width / 2)
        edges = []
        h = height * 2
        w = width * 2
        for i in range(group_num):
            print(i)
            mid_r = int(h / 2)
            mid_c = int(w / 2)
            for j in range(1,S - 2):
                cand = []
                for r in range(1,h - 1):
                    # print(r)
                    for c in range(1,w - 1):
                        b_flag = 0
                        s_flag = 0
                        flag = 0
                        for m in range(-1,2):
                            if flag:
                                break
                            for n in range(-1,2):


                                if DoG_pyramid[i][j - 1][r + m][r + n] > DoG_pyramid[i][j][r][c]:
                                    b_flag = 1

                                if DoG_pyramid[i][j - 1][r + m][r + n] < DoG_pyramid[i][j][r][c]:
                                    s_flag = 1

                                if DoG_pyramid[i][j + 1][r + m][r + n] > DoG_pyramid[i][j][r][c]:
                                    b_flag = 1

                                if DoG_pyramid[i][j + 1][r + m][r + n] < DoG_pyramid[i][j][r][c]:
                                    s_flag = 1

                                if DoG_pyramid[i][j][r + m][r + n] > DoG_pyramid[i][j][r][c]:
                                    b_flag = 1

                                if DoG_pyramid[i][j][r + m][r + n] < DoG_pyramid[i][j][r][c]:
                                    s_flag = 1

                                if b_flag and s_flag:
                                    flag = 1
                                    break
                        if flag:
                            continue


                        else:
                            origin_row = mid_row + int(height / h) * (r - mid_r)
                            origin_col = mid_col + int(width / w) * (c - mid_c)
                            edge[origin_row][origin_col] = 1
            h = int(h / 2)
            w = int(w / 2)
        print(edge.count(1))
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
        self.findEdge(self.img1)
        # self.genDoG(self.img1)
        # self.show()

if __name__ == '__main__':
    a = main()
    a.mainMethod()