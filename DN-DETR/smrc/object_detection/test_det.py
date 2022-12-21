# import sys
# sys.path.append("../..")
# import os
# print(os.getcwd())
# sys.path.append("../..")
import sys
sys.path.append("..")
# import os
# print(os.getcwd())

print(f'sys.path = {sys.path} ...')
# import smrc

# # import smrc
# from smrc.not_used.annotate import VisualizeLabel


# print(f'sys.path = {sys.path} ...')
# import smrc
from smrc.object_detection.yolo3.video_detection import yolo_detection
# 284010.avi
yolo_detection(video_path='284010.avi')  # 6604