####################################
# worked version
# 1. terminal call
# (mask-rcnn) kai@kai:~/tuat/smrc/annotate$ python ttt1.py
# 2. pycharm call
###########################################
#
# import sys
# sys.path.append("../..")
# # import os
# # print(os.getcwd())
# print(f'sys.path = {sys.path} ...')
# from smrc.not_used.annotate import VisualizeLabel
#
# tool = VisualizeLabel(
#     image_dir='../test_data/inside-car-image',
#     label_dir='../test_data/processed_training_data',
#     class_list_file='../config/class_list_driver.txt',
#     auto_load_directory='label_dir'
# )
#
# tool.main_loop()

#####################################
#  3. python environment
###########################################
"""
    (mask-rcnn) kai@kai:~/tuat$ python
    Python 3.7.7 (default, Mar 26 2020, 15:48:22)
    [GCC 7.3.0] :: Anaconda, Inc. on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from smrc.not_used.annotate import VisualizeLabel
    pygame 1.9.6
    Hello from the pygame community. https://www.pygame.org/contribute.html
    >>> tool = VisualizeLabel(
    ...     image_dir='smrc/test_data/inside-car-image',
    ...     label_dir='smrc/test_data/processed_training_data',
    ...     class_list_file='smrc/config/class_list_driver.txt',
    ...     auto_load_directory='label_dir'
    ... )
    ===================================== Information for visualization
    self.IMAGE_DIR: smrc/test_data/inside-car-image
    self.LABEL_DIR: smrc/test_data/processed_training_data
    self.auto_load_directory: label_dir
    self.class_list_file: smrc/config/class_list_driver.txt
    self.user_name: None
    self.CLASS_LIST: ['non-driver', 'driver']
    =====================================
    >>> tool.main_loop()
    self.play_music_on =False
    Number of joysticks: 0
    pressed_key= 13
    Enter key is pressed.
    675401   selected
    Start visualize the annotation or object_detection result for directory 675401
    self.CLASS_LIST =  ['non-driver', 'driver']
    image size: height=480, width=640 for smrc/test_data/inside-car-image/675401
    self.play_music_on =False
    Number of joysticks: 0
    pressed_key= 27
    Esc key is pressed, quit the program.
"""


#####################################
# 4. Use the information of system root dir
# to relocate the directory
# 1). terminal call
# (mask-rcnn) kai@kai:~/tuat/smrc/annotate$ python ttt1.py
# 2). pycharm call
#################################

# import sys
# sys.path.append("../..")
# # import os
# # print(os.getcwd())
# print(f'sys.path = {sys.path} ...')
# from smrc.not_used.annotate import VisualizeLabel
#
# from smrc.not_used.annotate.sys_path import *
#
# # /home/kai/tuat/smrc
# print(SMRC_ROOT_PATH)
# tool = VisualizeLabel(
#     image_dir=os.path.join(SMRC_ROOT_PATH, 'test_data/inside-car-image'),
#     label_dir=os.path.join(SMRC_ROOT_PATH, 'test_data/processed_training_data'),
#     class_list_file=os.path.join(SMRC_ROOT_PATH, 'config/class_list_driver.txt'),
#     auto_load_directory='label_dir'
# )
#
# tool.main_loop()


#####################################
# 5. Use the information of system root dir
# to relocate the directory
# 1). terminal call
# (mask-rcnn) kai@kai:~/tuat/smrc/annotate$ python ttt1.py
# 2). pycharm call
#################################
import sys
sys.path.append("../..")
# import os
# print(os.getcwd())
# print(f'sys.path = {sys.path} ...')

from smrc.utils.annotate import VisualizeLabel
from smrc.annotate.future_work.ann_tool_argparse_set import *

# changing dir works but it does not change anything
# # /home/kai/tuat/smrc
# print(SMRC_ROOT_PATH)
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

tool = VisualizeLabel(
    image_dir=os.path.join(SMRC_ROOT_PATH, 'test_data/inside-car-image'),
    label_dir=os.path.join(SMRC_ROOT_PATH, 'test_data/processed_training_data'),
    class_list_file=os.path.join(SMRC_ROOT_PATH, 'config/class_list_driver.txt'),
    auto_load_directory='label_dir'
)

tool.main_loop()


######################################
# for outside code, it's simple, just do what
# I did. For my own application, this file is enough.
# call inside the directory:
# # 1). terminal call
# # (mask-rcnn) kai@kai:~/tuat/smrc/annotate$ python ttt1.py
###########################################