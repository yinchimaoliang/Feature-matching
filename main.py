import cv2 as cv
import numpy as np


IMAGE_PATH1 = './images/a.jpg'
IMAGE_PATH2 = './images/b.jpg'

class main():
    def __init__(self):
        self.img1 = cv.imread(IMAGE_PATH1)
        self.img2 = cv.imread(IMAGE_PATH2)

    def mainMethod(self):

        # -- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
        detector = cv.ORB_create()

        keypoints1, descriptors1 = detector.detectAndCompute(self.img1, None)
        keypoints2, descriptors2 = detector.detectAndCompute(self.img2, None)

        # -- Step 2: Matching descriptor vectors with a brute force matcher
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptors1, descriptors2)

        # Sort matches in the order of their distances
        matches = sorted(matches, key=lambda x: x.distance)
        # -- Draw matches
        img_matches = np.empty((max(self.img1.shape[0], self.img2.shape[0]), self.img1.shape[1] + self.img2.shape[1], 3), dtype=np.uint8)
        cv.drawMatches(self.img1, keypoints1, self.img2, keypoints2, matches[:10], img_matches)

        # -- Show detected matches
        cv.imshow('Matches', img_matches)
        cv.waitKey(0)


if __name__ == '__main__':
    a = main()
    a.mainMethod()