import cv2
import numpy as np


class Camera:
    width: int
    height: int
    K: np.ndarray
    dist: np.ndarray
    newK: np.ndarray
    pose: np.ndarray

    def __init__(self, K, dist, width, height, pose, newK=None):
        self.width = width
        self.height = height
        self.K = np.array(K)
        self.dist = np.array(dist)
        # self.newK = newK
        self.pose = np.array(pose)
        if newK is None:
            newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (width, height), 0, (width, height))
        self.newK = np.array(newK)

    def resize(self, size):
        """

        Parameters
        ----------
        size: dst size. (width, height)

        Returns Camera
        -------

        """

        assert len(size) == 2
        width, height = size
        resized_K = self.K.copy()
        resized_K[0] *= width / self.width
        resized_K[1] *= height / self.height
        resized_new_K = self.newK.copy()
        resized_new_K[0] *= width / self.width
        resized_new_K[1] *= height / self.height
        return Camera(resized_K, self.dist, width, height, self.pose, resized_new_K)

    def crop(self, box):
        """

        Parameters
        ----------
        box: (x1,y1,x2,y2)

        Returns
        -------

        """
        x1, y1, x2, y2 = box
        cropped_K = self.K.copy()
        cropped_K[0, 2] -= x1
        cropped_K[1, 2] -= y1
        cropped_new_K = self.newK.copy()
        cropped_new_K[0, 2] -= x1
        cropped_new_K[1, 2] -= y1
        return Camera(cropped_K, self.dist, x2 - x1, y2 - y1, self.pose, cropped_new_K)

    def undistort_image(self, img):
        """

        Parameters
        ----------
        img: height,width,3 or 4. np.ndarray. uint8

        Returns
        -------

        """
        assert img.shape[0] == self.height and img.shape[1] == self.width
        # todo: support image with different size from self.size
        dst = cv2.undistort(img, self.K, self.dist, None, self.newK)
        return dst
