import smrc.utils
import sys
from smrc.utils.annotate import VisualizeOutlier

#
# IMAGE_DIR = 'Truck-sampleData114videos'
# LABEL_DIR = 'Truck_SampleData-labels-thd0.05'
# CLASS_LIST_FILE = 'class_list_license_plate.txt'
#

IMAGE_DIR = 'data/trail_sample_data/image_data/Truck-sampleData114videos'
LABEL_DIR = 'data/trail_sample_data/refined_labels_v1_2019_09/Truck_SampleData_Facelabels'

CLASS_LIST_FILE = 'config/class_list_face.txt'

# # get the statistics of the annotated videos MASK_DIR
# smrc.not_used.estimate_annotation_statistics(LABEL_DIR)  #FINAL_MASK_DIR

txt_file_list = smrc.utils.get_txt_file_list_for_specific_annotated_classes(
    LABEL_DIR, [0]
)

# for ann_path in txt_file_list:
#     print(f'Deleting {ann_path}')
#     os.remove(ann_path)

print(txt_file_list)
if len(txt_file_list) == 0:
    sys.exit(0)
# print('args_5d_det.image_dir =' , args_5d_det.image_dir)
## we can specify the active directory by using args_5d_det.input_dir
visualization_tool = VisualizeOutlier(
    image_dir=IMAGE_DIR, 
    label_dir=LABEL_DIR, 
    class_list_file=CLASS_LIST_FILE,
    specified_image_list=None,
    specified_ann_path_list=txt_file_list
)
visualization_tool.main_loop()

