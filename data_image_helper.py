# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:55:34 2019

@author: 50568
"""

import data_file_helper as fh
import tensorflow as tf
# import tensorflow.contrib as tf_contrib
import numpy as np
import cv2
import visualization
import dlib
import imutils
from imutils import face_utils


class data_image_helper:
    def __init__(self):
        # self.detector = detector
        self.detector = dlib.get_frontal_face_detector()
        # predictor = dlib.shape_predictor(args["shape_predictor"])
        #
        # # load the input image, resize it, and convert it to grayscale
        # image = cv2.imread(args["image"])
        self.predictor = dlib.shape_predictor(
            './pre_train/shape_predictor_68_face_landmarks.dat')

    def mouth_detector(self, frame):
        # image = imutils.resize(frame, width=500)
        image = frame
        # image = hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # frame = imutils.resize(frame, width=500)
        # image = cv2.GaussianBlur(image, (5, 5), 0)
        # image = cv2.bilateralFilter(image, 9, 75, 75)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = self.detector(gray, 1)

        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the face parts individually
            # for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            name = 'mouth'
            i = 48
            t = 60
            j = 68
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            clone = image.copy()
            zero = np.zeros_like(clone)
            # cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            #             (0, 0, 255), 2)

            # loop over the subset of facial landmarks, drawing the
            # specific face part

            # fl_60_x, fl_60_y = shape[59]
            # start_x, start_y = shape[i]

            # cv2.polylines(zero, (start_x,start_y), (fl_60_x, fl_60_y), (0, 0, 255))
            temp = []
            for (x, y) in shape[i:t]:
                # cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                # cv2.polylines(zero, (start_x,start_y), (x, y), (0, 0, 255))
                # start_x = x
                # start_y = y
                temp.append((x, y))
            cv2.polylines(zero, np.array([temp]), True, (255, 255, 255), 1,
                          cv2.LINE_AA)
            # fl_68_x, fl_68_y = shape[67]
            # start_x, start_y = shape[t]
            temp = []
            # cv2.polylines(zero, (start_x,start_y), (fl_68_x, fl_68_y), (0, 0, 255))
            for (x, y) in shape[t:j]:
                # cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                # cv2.polylines(zero, (start_x,start_y), (x, y), (0, 0, 255))
                # cv2.polylines
                # start_x = x
                # start_y = y
                temp.append((x, y))
            cv2.polylines(zero, np.array([temp]), True, (255, 255, 255), 1,
                          cv2.LINE_AA)
            # import pdb; pdb.set_trace()
            # extract the ROI of the face region as a separate image
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            zero = cv2.GaussianBlur(zero, (3, 3), 0)
            roi = zero[y:y + h, x:x + w]
            try:
                roi = cv2.resize(roi, (64, 32), cv2.INTER_CUBIC)
            except Exception:
                return None
            return roi

    def read_img(self, path, shape, size, begin=0, end=0):
        """
            Video_Read is used to extract the image of mouth from a video;\n
            parameter:\n
            Path: the string path of video\n
            Shape: the (min, max) size tuple of the mouth you extract from the video\n
            Size: the (high, weight) size tuple of the mouth image you save
        """
        cap = cv2.VideoCapture(path)
        images = []
        mouth = None
        cnt = 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        v_length = frames / fps
        if (end == 0 or end >= v_length):
            end = v_length

        if (cap.isOpened() == False):
            print("Read video failed!")
            return None

        # get detector
        # classifier_face_default = cv2.CascadeClassifier(
        #     "./cascades/haarcascade_frontalface_default.xml")
        # classifier_face_alt = cv2.CascadeClassifier(
        #     "./cascades/haarcascade_frontalface_alt.xml")

        cap.set(cv2.CAP_PROP_POS_FRAMES, begin * fps)
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        while (pos <= end * fps and end <= v_length):
            ret, img = cap.read()
            '''
                第一个参数ret的值为True或False，代表有没有读到图片
                第二个参数是frame，是当前截取一帧的图片
            '''
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if ret == False:
                break
            # img = contrast_enhancement(img)
            # cv2.imwrite('./samples/frame_' + str(cnt) + '.jpg', img)
            mouth = self.mouth_detector(img)
            # mouth = contrast_enhancement(mouth)
            # cv2.imwrite('./samples/lip_' + str(cnt)+ '.jpg', mouth)
            if mouth is not None:
                if np.sum(mouth) > 0:
                    images.append(mouth)
                    cnt += 1

        cap.release()
        cv2.destroyAllWindows()
        return images, cnt / frames

    def get_raw_dataset(self, path, shape=(8, 8), size=(32, 32)):

        video, cnt = self.read_img(path, shape, size, 0, 0)
        # import pdb
        # pdb.set_trace()
        video = np.array(video, np.float32) / 255
        # video = tf.keras.applications.resnet50.preprocess_input(video, mode='tf')
        # for k, v in enumerate(video):
        #     cv2.imshow('image', v)
        #     cv2.waitKey()
        return video, cnt
        # return tf.data.Dataset.from_generator(generator, tf.float32)

    def prepare_data(
            self,
            paths,
            batch_size,
            shape=(20, 20),
            size=(224, 224),
    ):

        dataset = []
        length = []
        for path in paths:
            video, cnt = self.read_img(path, shape, size, 0.5, 1)
            video = np.array(video) / 255.0
            video = video.astype(np.float32)
            dataset.append(video)
            length.append(cnt)

        def generator():
            for d, c in zip(dataset, length):
                yield d, c

        raw_dataset = tf.data.Dataset.from_generator(generator,
                                                     (tf.float32, tf.int32))
        batch_dataset = raw_dataset.padded_batch(batch_size,
                                                 padded_shapes=(tf.TensorShape(
                                                     [None, 109, 109,
                                                      5]), tf.TensorShape([])))

        return batch_dataset, raw_dataset


def contrast_enhancement(img):
    # img = cv2.resize(img, (300,300), interpolation =cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab",lab)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    # cv2.imshow('l_channel', l)
    # cv2.imshow('a_channel', a)
    # cv2.imshow('b_channel', b)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # cv2.imshow('CLAHE output', cl)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))
    # cv2.imshow('limg', limg)

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


# if __name__ == '__main__':
# video, txt = fh.read_file(
#     '/Users/barid/Documents/workspace/batch_data/lip_data')
# print(video[:5])
# print(txt[:5])

# helper = data_image_helper()
# # b, d = helper.prepare_data(paths = ['D:/lip_data/ABOUT/train/ABOUT_00003.mp4'], batch_size = 64)
# b, cnt = helper.get_raw_dataset(
#     path=
#     '/Users/barid/Documents/workspace/batch_data/5550881446804971382/00012.mp4'
# )

# print(b.shape)
# print(cnt)
#for (i,(x, l)) in enumerate(b):
#    print(x)
#    print(l)
