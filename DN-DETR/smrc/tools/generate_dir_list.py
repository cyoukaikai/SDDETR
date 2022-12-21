import argparse
import os
import smrc.utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate directory list')
    parser.add_argument('-i', '--data_root_dir', default=None, type=str, help='Path to data root directory')
    parser.add_argument('-o', '--output_dir_list_file', default=None, type=str, help='Path to output dir list file')
    args = parser.parse_args()

    smrc.utils.assert_dir_exist(args.data_root_dir)

    # if os.path.isfile(args_5d_det.output_dir_list_file):
    #     print(f'| File {args_5d_det.output_dir_list_file} already exists, please remove it before generate a new one.')
    if os.path.isfile(args.output_dir_list_file):
        print(f'| File {args.output_dir_list_file} already exists, removed ...')
        os.remove(args.output_dir_list_file)

    dir_list = smrc.utils.get_dir_list_in_directory(args.data_root_dir)
    smrc.utils.save_1d_list_to_file(file_path=args.output_dir_list_file, list_to_save=dir_list)
