# coding: utf-8
from tiny_fd import TinyFacesDetector
import sys
import cv2 as cv
import time
if __name__ == '__main__':

    start = time.time()
    detector = TinyFacesDetector(model_root='./', prob_thresh=0.5, gpu_idx=0)

    print(f' time = {time.time()-start}')
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = '../img.jpg'
    img = cv.resize(cv.imread(path), (224, 224))
    resized_img = cv.resize(img, (112, 112))
    start = time.time()
    boxes = detector.detect(resized_img)
    print(f' time = {time.time()-start}')
    print('Faces detected: {}'.format(boxes.shape[0]))
    print(boxes)
    for r in boxes:
        for i in range(len(r)):
            r[i] = r[i] * 2
        cv.rectangle(img, (r[0],r[1]), (r[2], r[3]), (0,0,255),3)
    # cv.namedWindow('Tiny FD', cv.WINDOW_NORMAL)
    cv.imshow('Tiny FD', img)
    cv.waitKey(1000)
