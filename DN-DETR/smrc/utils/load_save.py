######################################################
# load/save list from/to file
######################################################


import csv
import os
import pickle
import xlsxwriter
import numpy as np
from lxml import etree

from .file_path import non_blank_lines


def load_1d_list_from_file(filename):
    assert os.path.isfile(filename)

    with open(filename) as f_directory_list:
        resulting_list = list(non_blank_lines(f_directory_list))

    return resulting_list


def load_class_list_from_file(class_list_file):
    # class_list_file = os.path.join(
    #     os.path.dirname(os.path.abspath(__file__)), class_list_file
    # )
    with open(class_list_file) as f:
        class_list = list(non_blank_lines(f))
    return class_list


def load_directory_list_from_file(filename):
    return load_1d_list_from_file(filename)


def load_multi_column_list_from_file(filename, delimiter=','):

    resulting_list = []
    assert os.path.isfile(filename)

    with open(filename, 'r') as old_file:
        lines = list(non_blank_lines(old_file))

    # print('lines = ',lines)
    for line in lines:
        line = line.strip()
        result = line.split(delimiter)

        resulting_list.append(result)
    return resulting_list


def save_1d_list_to_file(file_path, list_to_save):
    with open(file_path, 'w') as f:
        for item in list_to_save:
            f.write("%s\n" % item)


def save_1d_list_to_file_incrementally(file_path, list_to_save):
    with open(file_path, 'a') as f:
        for item in list_to_save:
            f.write("%s\n" % item)


def save_multi_dimension_list_to_file(filename, list_to_save, delimiter=','):
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(list_to_save)  # considering my_list is a list of lists.


def save_multi_dim_list_to_file_incrementally(filename, list_to_save, delimiter=','):
    with open(filename, 'a') as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(list_to_save)  # considering my_list is a list of lists.


def load_pkl_file(pkl_file_name, verbose=True):
    if os.path.isfile(pkl_file_name):
        f = open(pkl_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        if verbose:
            print(f'data loaded from {pkl_file_name}.')
        return data
    else:
        return None


def generate_pkl_file(pkl_file_name, data, verbose=True):
    f = open(pkl_file_name, 'wb')
    pickle.dump(data, f)
    f.close()
    if verbose:
        print(f'{pkl_file_name} saved ...')

    # with open(pkl_file_name, 'wb') as f:
    #     pickle.dump(data, f)


def save_matrix_to_txt(filename, mat, num_digit=2):
    # np.savetxt('final_similar_matrix', final_similar_matrix, fmt='%.3f')
    np.savetxt(filename, mat, fmt=f'%.{num_digit}f')


def save_excel_file(list_2d, result_file_name, field_name_list=None):
    """
    Save the data into excel file.
    :param result_file_name: e.g., 'Expenses01.xlsx'
    :param list_2d:
    :param field_name_list: if given, then write into the first row of the resulting file
    :return:
    """
    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(result_file_name)
    worksheet = workbook.add_worksheet('video_information')

    row_id = 0
    if field_name_list is not None:
        for k, item in enumerate(field_name_list):
            worksheet.write(0, k, item)
        row_id += 1

    # Iterate over the data and write it out row by row.
    for i, row_contents in enumerate(list_2d):
        for j, col_contents in enumerate(row_contents):
            worksheet.write(row_id + i, j, col_contents)
    # Write a total using a formula.
    # worksheet.write(row, 0, 'Total')
    # worksheet.write(row, 1, '=SUM(B1:B4)')

    workbook.close()


def list_to_excel_file(list_2d, result_file_name_prefix, sheet_name='Sheet1'):
    """
    Save the data into excel file.
    :return:
    @param list_2d:
    @param result_file_name_prefix:
    @param sheet_name:
    """
    result_file_name = result_file_name_prefix + '.xlsx'
    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(result_file_name)
    worksheet = workbook.add_worksheet(sheet_name)

    row_id = 0
    # Iterate over the data and write it out row by row.
    for i, row_contents in enumerate(list_2d):
        for j, col_contents in enumerate(row_contents):
            worksheet.write(row_id + i, j, col_contents)

    # Write a total using a formula.
    # worksheet.write(row, 0, 'Total')
    # worksheet.write(row, 1, '=SUM(B1:B4)')
    workbook.close()


def write_xml(xml_str, xml_path):
    """
    Copied from OpenLabeling
    """
    # remove blank text before prettifying the xml
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(xml_str, parser)
    # prettify
    xml_str = etree.tostring(root, pretty_print=True)
    # save to file
    with open(xml_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)