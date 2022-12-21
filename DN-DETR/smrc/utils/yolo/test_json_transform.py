import smrc.utils

########################################
# load yolov4 json files yolov3 files to detection to dictionary
############################################
from smrc.utils import json_yolov4_to_yolov3

v4_json_root_dir = 'result/results_yolov4'
v3_json_root_dir = 'result'

smrc.utils.generate_dir_if_not_exist(v3_json_root_dir)
json_yolov4_to_yolov3(
    json_yolov4_dir=v4_json_root_dir,
    json_yolov3_dir=v3_json_root_dir
)
