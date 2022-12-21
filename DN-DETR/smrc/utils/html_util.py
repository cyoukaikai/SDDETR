import os
import webbrowser
import numpy as np


def generate_pure_image_html(
        image_path_list, filename, image_captions=None, num_cols=5,
        resize=True, resize_h_w_dim=None, title=None, open_file=False
):
    """generate html file for resulting clusters
    <img src="#" width="50%" height="50%"> # thid did not work, becuase the image will occupy 50% of the <tb>
    not the whole screen.
    <img src="#" width="173" height="206.5">
    :param image_rows: a list, [0] , caption of the row: [1], a dict for everything else including subcaption, image paths
    :param filename:
    :param resize:
    :param title:
    :param open_file:
    :param resize_h_w_dim: (height, width) tuple
    :return:
    """
    # size = "12", size = "8",
    if filename is None:
        filename = 'resulting_file_' + smrc.utils.time_stamp_str() + '.html'
    f = open(filename, 'w')

    message = f"""<html>
<head></head>
<body>
<h1  style="text-align:center">{title}</h1>
<table border="1" align = 'center'>
"""
    img_h, img_w = resize_h_w_dim[0], resize_h_w_dim[1]

    num_image = len(image_path_list)
    num_rows = int(np.ceil(num_image / num_cols))

    for k in range(num_rows):
        # <td>
        # <font color = "red">  row {k}</font>
        # </td>
        message += f"""<tr>
        """  # {i}:
        # generate table for one cluster
        message += f"""
<td><table border="1" align = 'center' ><tr>"""

        image_str_id, image_end_id = k * num_cols, \
                                     min(num_cols * (k + 1), num_image)
        row_image_list = image_path_list[image_str_id:image_end_id]

        for j, src_image_path in enumerate(row_image_list):
            if resize:
                message += f"""
        <td align='center'><img src="{src_image_path}" height = "{img_h}", width = "{img_w}">"""
            else:
                message += f"""
                <td align='center'><img src="{src_image_path}">"""

            if image_captions is not None:
                message += f"""
</br>{image_captions[k * num_cols + j]}</td>"""
            if (j+1) % 50 == 0 and j > 0 and j != len(image_list)-1:
                message += f"""</tr><tr>"""
        message += f"""</tr></table></td>"""

        message += "</tr>"
    message += "</table>"
    # appearance_dist_matrix[spatial_dist_matrix == float('inf')] = -1

    message += """\n</body>
    </html>"""

    f.write(message)
    f.close()
    if open_file:
        webbrowser.open_new_tab(filename)


def generate_image_html_file(
        image_rows, filename, resize=True, resize_h_w_dim=None, title=None, open_file=False
):
    """generate html file for resulting clusters
    <img src="#" width="50%" height="50%"> # thid did not work, becuase the image will occupy 50% of the <tb>
    not the whole screen.
    <img src="#" width="173" height="206.5">
    :param image_rows: a list, [0] , caption of the row: [1], a dict for everything else including subcaption, image paths
    :param filename:
    :param resize:
    :param title:
    :param open_file:
    :param resize_h_w_dim: (height, width) tuple
    :return:
    """
    # size = "12", size = "8",
    if filename is None:
        filename = 'resulting_file_' + smrc.utils.time_stamp_str() + '.html'
    f = open(filename, 'w')

    message = f"""<html>
<head></head>
<body>
<h1  style="text-align:center">{title}</h1>
<table border="1" align = 'center'>
"""
    img_h, img_w = resize_h_w_dim[0], resize_h_w_dim[1]

    for i, one_row in enumerate(image_rows):
        row_title, row_content = one_row[0], one_row[1]
        message += f"""<tr>
<td>
<font color = "red">  {row_title}</font>
</td>
        """  # {i}:
        image_list = row_content['image_list']

        # generate table for one cluster
        message += f"""
<td><table border="1" align = 'center' ><tr>"""
        for j, src_image_path in enumerate(image_list):

            if resize:
                message += f"""
        <td align='center'><img src="{src_image_path}" height = "{img_h}", width = "{img_w}">"""
            else:
                message += f"""
                <td align='center'><img src="{src_image_path}">"""

            if 'image_captions' in row_content:
                message += f"""
</br>{row_content['image_captions'][j]}</td>"""

            if (j+1) % 50 == 0 and j > 0 and j != len(image_list)-1:
                message += f"""</tr><tr>"""
        message += f"""</tr></table></td>"""

        message += "</tr>"
    message += "</table>"
    # appearance_dist_matrix[spatial_dist_matrix == float('inf')] = -1

    message += """\n</body>
    </html>"""

    f.write(message)
    f.close()
    if open_file:
        webbrowser.open_new_tab(filename)


# table of content
def generate_toc_html_file(
        toc_rows, filename,  title=None, open_file=False
):
    """generate html file for table of content

    :param toc_rows: a list, [0] , caption of the row: [1], description and link of the content
    :param filename:
    :param title:
    :param open_file:

    :return:
    """
    # size = "12", size = "8",
    if filename is None:
        filename = 'resulting_file_' + smrc.utils.time_stamp_str() + '.html'
    f = open(filename, 'w')

    message = f"""<html>
<head></head>
<body>
<h1  style="text-align:center">{title}</h1>
<table border="1" align = 'center'>
"""

    for i, one_row in enumerate(toc_rows):
        row_title, row_content, link = one_row[0], one_row[1], one_row[2]
        message += f"""<tr>
<td>{row_title}</td>
        """
        if link is not None:
            # <a href="https://www.WordPress.com" target="_blank">WordPress Homepage</a>
            message += f"""<td><a href="{link}" target="_blank">{row_content}</a></td>"""
        else:
            message += f"""<td>{row_content}</td>"""
        message += "</tr>"
    message += "</table>"
    message += """\n</body>
    </html>"""

    f.write(message)
    f.close()
    if open_file:
        webbrowser.open_new_tab(filename)