import os
import smrc.utils

# The library path for the project.  # '/disks/cnn1/kaikai/project/DN-DETR'
LIB_ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '', '..'))

# LIB_ROOT_DIR = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))

# Set the path of the nuscene dataset, make sure you have the nuscenes dataset saved in the following relative path
COCO_DATASET_ROOT_DIR = os.path.join(LIB_ROOT_DIR, 'coco')
# Set the path of the nuscene dataset if use absolute path (deprecated).
# NUSCENES_DATASET_ROOT_DIR = '/media/sirius/T/project/dataset/nuScene/nuscenes'

# NUSCENES_DATASET_REL_ROOT_DIR = '../data/nuscenes'  # Relative path
# EVAL_RESULT_ROOT_DIR = os.path.join(NUSCENES_DATASET_ROOT_DIR, 'eval')
# TEST_DATA_ROOT_DIR = os.path.join(LIB_ROOT_DIR, 'test_data')
# SCENE_KEYWORD_DATA_DIR = os.path.join(LIB_ROOT_DIR, 'tti/config/scene_descriptions')
VISUALIZATION_DIR = os.path.join(LIB_ROOT_DIR, 'visualize')

# The test results of this library will be saved here.
LIB_RESULT_DIR = os.path.join(LIB_ROOT_DIR, 'logs')

LIB_OUTPUT_DIR = os.path.join(LIB_ROOT_DIR, 'results_')
smrc.utils.generate_dir_if_not_exist(LIB_OUTPUT_DIR)