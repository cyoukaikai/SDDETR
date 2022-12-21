from tti.tti_conf import LIB_ROOT_DIR
import os
import smrc.utils
import json
import argparse
from tti.plot_utils import *
from pathlib import Path


def extract_coco_eval_result(test_dir):
    dir_list = smrc.utils.get_dir_list_in_directory(test_dir)

    for dir_name in dir_list:
        dir_path = os.path.join(test_dir, dir_name)
        print(f'dir_path = {dir_path}')
        # dfs, logs = plot_logs(logs=Path(dir_path),
        #                       fields=('mAP', 'class_error', 'loss_bbox_unscaled'),
        #                       plot_name=f'{dir_path}/coco_eval.png')
        #
        # coco_eval_mAPs = plot_mAP(dfs, logs, plot_name=f'{dir_path}/coco_mAP.png')

        log_files = smrc.utils.get_file_list_in_directory(dir_path, ext_str='log')
        for log_file in log_files:
            # os.path.isfile(os.path.join(dir_path, 'log.txt')):
            log_name = os.path.basename(log_file)
            print(f'| {log_file}')
            if log_file.find('.txt') > -1:
                coco_eval_mAPs = load_data(logs=[Path(dir_path)], log_name=log_name)
                # if coco_eval_mAPs

                coco_eval_mAP = coco_eval_mAPs[0]
                smrc.utils.save_multi_dimension_list_to_file(
                    filename=f'{dir_path}/coco_eval.txt',
                    list_to_save=list(coco_eval_mAP)
                )

                for map in coco_eval_mAP:
                    print(f'{int(map[0])}  {f"%.3f" % map[1]}')


                # log_file = os.path.join(dir_path, 'log.txt')
                # if os.path.isfile(log_file):
                #     plot_logs(logs=log_file)
                # with open(log_file) as f:
                #     data = f.read()
                #
                # results = json.loads(data)
                #
                # results_new = []
                # for rc in results:
                #     epoch = rc['epoch']
                #     test_coco_eval_bbox = rc['test_coco_eval_bbox']
                #     results_new.append([epoch] + test_coco_eval_bbox)
                # print(f'results_extracted = {results_new}')

def test():
    tmp_test_dir = os.path.join(
        LIB_ROOT_DIR,
        'logs/gt_only_exp-reclaim_padded_region-'
        'no_bg_token_remove-16-gt_fg_scale_fake-fg_scale_class_small_medium_random')
    extract_coco_eval_result(tmp_test_dir)


if __name__ == "__main__":
    """
    (mm3det) kaikai@cnn:/disks/cnn1/kaikai/project/DN-DETR$ python tti/read_coco_results_tool.py -i  logs/gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-fg_scale_class_small_medium_random

    """
    parser = argparse.ArgumentParser(description='Extract coco results')
    parser.add_argument('-i', '--input_dir', default='', type=str, help='Path to input directory')
    args = parser.parse_args()

    smrc.utils.assert_dir_exist(args.input_dir)
    extract_coco_eval_result(args.input_dir)
