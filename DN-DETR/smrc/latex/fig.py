import os
import numpy as np
import smrc.utils
import random


def write_latex_figure(ImagePaths, SubCaptions, Caption, Label, Width):
    """
    # %\subfigure[CIFAR10-QUICK (not fixing weights)]
    {\includegraphics[width=0.24\textwidth]{ML_Samples_20181109b/105189/0290.jpg}}
    :param ImagePaths:
    :param SubCaptions:
    :param Caption:
    :param Label:
    :param Width:
    :return:
    """
    table_str = """\\begin{figure}[th!]
\centering\n"""

    for image_path, sub_caption in zip(ImagePaths, SubCaptions):
        table_str += '\subfigure[' + sub_caption + \
                     ']{\includegraphics[width=' + '{:.2}'.format(Width) + '\\textwidth]{' + image_path + '}}\n'
    table_str += '\caption{' + Caption + '}\n'
    # \label{fig:fixing-weights}
    table_str += '\label{fig:' + Label + '}\n'
    table_str += '\end{figure}\n'
    return table_str


def latex_page_for_image_list(latex_file_name, image_path_list, num_rows=6, num_cols=3):

    num_image = len(image_path_list)
    num_image_per_table = num_rows * num_cols
    num_table = int(np.ceil(num_image / num_image_per_table))

    text = ''
    for k in range(num_table):
        image_str_id, image_end_id = k * num_image_per_table, \
                                     min(num_image_per_table * (k + 1), num_image)
        ImagePaths = image_path_list[image_str_id:image_end_id]

        SubCaptions = [smrc.utils.file_path_last_two_level(image_path) for image_path in ImagePaths]
        Caption = f'image visualization'
        Label = f'table{k}'
        Width = 1 / num_cols - 0.02
        text += write_latex_figure(ImagePaths, SubCaptions, Caption, Label, Width)
        text += '\n\n\clearpage'

    f = open(latex_file_name, 'w')
    f.write(text)
    f.close()


def latex_page_for_image_list_more_details(
        latex_file_name, image_path_list, sub_captions=None, captions=None,
        num_rows=6, num_cols=3
):

    num_image = len(image_path_list)
    num_image_per_table = num_rows * num_cols
    num_table = int(np.ceil(num_image / num_image_per_table))

    text = ''
    for k in range(num_table):
        image_str_id, image_end_id = k * num_image_per_table, \
                                     min(num_image_per_table * (k + 1), num_image)
        ImagePaths = image_path_list[image_str_id:image_end_id]

        if sub_captions is not None:
            # SubCaptions = sub_captions[k]
            SubCaptions = sub_captions[image_str_id:image_end_id]
        else:
            SubCaptions = [smrc.utils.file_path_last_two_level(image_path) for image_path in ImagePaths]

        if captions is not None:
            Caption = captions[k]
        else:
            Caption = f'Image visualization'
        Label = f'table{k}'
        Width = 1 / num_cols - 0.02
        text += write_latex_figure(ImagePaths, SubCaptions, Caption, Label, Width)
        text += '\n\n\clearpage'

    f = open(latex_file_name, 'w')
    f.write(text)
    f.close()
    print(f'Generating {os.path.abspath(latex_file_name)} done.')


# not used
def latex_one_page_image_list(ImagePaths, Caption, Label, Width, SubCaptions=None):
    text = ''
    if SubCaptions is None:
        SubCaptions = [smrc.utils.file_path_last_two_level(image_path) for image_path in ImagePaths]

    if Caption is None:
        Caption = f'image visualization'

    text += write_latex_figure(ImagePaths, SubCaptions, Caption, Label, Width)
    text += '\n\n\clearpage'


def latex_page_for_parent_image_dir(image_dir, latex_file_name,
                                    num_rows=6, num_cols=3, dir_list=None,
                                    random_sample=False, num_sample=1):
    image_path_list = []
    if dir_list is None:
        dir_list = smrc.utils.get_dir_list_in_directory(image_dir)

    for dir_name in dir_list:
        image_paths = smrc.utils.get_file_list_recursively(
            os.path.join(image_dir, dir_name)
        )
        # if we random sample the image from the path
        if random_sample:
            image_paths = random.choices(image_paths, k=num_sample)
            image_paths = sorted(image_paths)

        image_path_list.extend(
            [os.path.abspath(image_path) for image_path in image_paths]
        )
    latex_page_for_image_list(latex_file_name, image_path_list,
                              num_rows=num_rows, num_cols=num_cols)


# def generate_resulting_clusters_html_file(
#         self, clusters, filename, resize=True, title=None
# ):
#
#     if filename is None: filename = 'resulting_clusters' + smrc.not_used.time_stamp_str() + '.html'
#     f = open(filename, 'w')
#
#     message = f"""
#     <html>
#     <head></head>
#     <body>
#     <h1>{title}</h1>
#     <table border="1" align = 'center' >
#     """
#     obj_height, obj_width = 100, 40
#
#     # if self.html_detection_image_dir is None:
#     #     image_dir_name = 'clusters/html_images/detections'
#     # else:
#     image_dir_name = self._generate_image_dir_signature_prefix()  # 'clusters/html_images/detections'
#
#     for i, cluster in enumerate(clusters):
#         message += f"""<tr>
#                     <td>
#                     <font color = "red">cluster {i} (len: {len(cluster)})</font>
#                     </td>
#                     """
#
#         # generate table for one cluster
#         message += f"""<td><table border="1" align = 'left' ><tr>"""
#         for j, global_bbox_id in enumerate(cluster):
#             # src_image_path = os.path.join(image_dir_name, 'obj' + str(i) + '_' + str(j) + '.jpg')
#             src_image_path = os.path.join(image_dir_name, str(global_bbox_id) + '.jpg')
#             if resize:
#                 message += f"""
#                 <td><img src="{src_image_path}" height = "{obj_height}", width = "{obj_width}" >
#                 </br>{self.get_image_id(global_bbox_id)}</td>"""
#             else:
#                 message += f"""
#                 <td><img src="{src_image_path}"></br>{self.get_image_id(global_bbox_id)}</td>"""
#
#             if (j+1) % 50 == 0 and j > 0 and j != len(cluster)-1:
#                 message += f"""</tr><tr>"""
#         message += f"""</tr></table></td>"""
#
#         message += "</tr>"
#     message += "</table>"
#     # appearance_dist_matrix[spatial_dist_matrix == float('inf')] = -1
#
#     message += """
#     </body>
#     </html>"""
#
#     f.write(message)
#     f.close()
#
#     # webbrowser.open_new_tab(filename)