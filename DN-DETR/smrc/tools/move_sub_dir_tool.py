import os
import argparse
# from smrc.flp.drdb_masking.generate_mask_ import DRDBDatMaskGeneration
import smrc.utils


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SMRC')
    parser.add_argument('-i', '--source_dir', default=None, type=str, help='Path to source directory')
    parser.add_argument('-o', '--target_dir', default=None, type=str, help='Path to target directory')
    parser.add_argument('-d', '--directory_list_file', default=None, type=str, help='Path of dir list file')
    args = parser.parse_args()

    smrc.utils.assert_dir_exist(args.source_dir)
    smrc.utils.generate_dir_if_not_exist(args.target_dir)

    if args.directory_list_file is not None:
        print(f'{os.path.abspath(args.directory_list_file)}')
        smrc.utils.assert_file_exist(args.directory_list_file)
        directory_list = list(set(smrc.utils.load_directory_list_from_file(
            args.directory_list_file
        )))
        dir_list = [dir_name for dir_name in directory_list if
                    os.path.isdir(os.path.join(args.source_dir, dir_name))]
        print(f'| Total {len(smrc.utils.get_dir_list_in_directory(args.source_dir))} '
              f'directories in {args.source_dir}')
        print(f'| To copy {len(dir_list)} directories from given {len(directory_list)} '
              f'directories in {args.directory_list_file}.')

        smrc.utils.move_sub_dirs(
            source_data_dir=args.source_dir,
            target_data_dir=args.target_dir,
            dir_list=dir_list,
            skip_if_not_exist=True
        )
