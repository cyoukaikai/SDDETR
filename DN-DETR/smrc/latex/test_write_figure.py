
import smrc.utils
from smrc.latex.fig import *


image_dir = '../../data/segmentation_results'
label_dir = '../../data/segmentation_labels'
##########################################
# for sampled images in the parent image dir
########################################
latex_file_name = 'random_sample_latex.txt'
latex_page_for_parent_image_dir(
    image_dir, latex_file_name,
    num_rows=6, num_cols=3, dir_list=None,
    random_sample=True, num_sample=18
)

##########################################
# for all images in the parent image dir
########################################
result_dir = image_dir + '_annotation_visualization'
latex_file_name = 'test_latex_write_figure.txt'
# dir_list = smrc.not_used.get_dir_list_in_directory(result_dir)
latex_page_for_parent_image_dir(
    result_dir, latex_file_name,
    num_rows=6, num_cols=3
)
