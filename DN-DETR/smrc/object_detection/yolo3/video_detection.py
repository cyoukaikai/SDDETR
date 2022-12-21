#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image

from .yolo import YOLO
from .preprocessing import non_max_suppression
from .detection import Detection
# from deep_sort.tracker import Tracker
# from tools import generate_detections as gdet
import imutils.video
from .videocaptureasync import VideoCaptureAsync
warnings.filterwarnings('ignore')


def yolo_detection(
        nms_thd=0.5, score_thd=0.15,
        video_path=None, output_path='output_yolov3.avi'
):
    yolo = YOLO()

    writeVideo_flag = False
    asyncVideo_flag = False

    if video_path is None:
        video_path = 'video.webm'
    else:
        # does not support using captured video
        # cap = VideoCaptureAsync('407156.avi')
        asyncVideo_flag = False

    if asyncVideo_flag:
        video_capture = VideoCaptureAsync(video_path)
    else:
        video_capture = cv2.VideoCapture(video_path)

    if asyncVideo_flag:
        video_capture.start()

    if writeVideo_flag:
        if asyncVideo_flag:
            w = int(video_capture.cap.get(3))
            h = int(video_capture.cap.get(4))
        else:
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            output_path, fourcc, 30, (w, h)
        )
        frame_index = -1

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
             break

        t1 = time.time()

        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        boxs = yolo.detect_image(image)[0]
        # confidence = yolo.detect_image(image)[1]
        #
        # features = [None] * len(boxs)  # here we do not need features
        # detections = [Detection(bbox, confidence, feature)
        #               for bbox, confidence, feature in
        #               zip(boxs, confidence, features)
        #               if confidence >= score_thd]
        #
        # # Run non-maxima suppression.
        # boxes = np.array([d.tlwh for d in detections])
        # scores = np.array([d.confidence for d in detections])
        # indices = non_max_suppression(boxes, nms_thd, scores)
        # detections = [detections[i] for i in indices]
        #
        # # # Call the tracker
        # # tracker.predict()
        # # tracker.update(detections)
        # #
        # # for track in tracker.tracks:
        # #     if not track.is_confirmed() or track.time_since_update > 1:
        # #         continue
        # #     bbox = track.to_tlbr()
        # #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
        # #     cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
        #
        # for det in detections:
        #     bbox = det.to_tlbr()
        #     score = "%.2f" % round(det.confidence * 100, 2)
        #     cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
        #     cv2.putText(frame, score + '%', (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 130, (0,255,0),2)
        #
        cv2.imshow('', frame)

        if writeVideo_flag: # and not asyncVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps_imutils.update()

        fps = (fps + (1./(time.time()-t1))) / 2
        print("FPS = %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()


# if __name__ == '__main__':
#     yolo_detection(video_path='6604.avi')
