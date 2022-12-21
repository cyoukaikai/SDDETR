from tti.tti_conf import VISUALIZATION_DIR

import smrc.utils
import os
import numpy as np
import cv2
from util.visualizer import COCOVisualizer, renorm
import json
import matplotlib.pyplot as plt
from models.sgdt.scoring_gt import unnormalize_box, TokenScoringGTGenerator
from models.sgdt.scoring_eval import TokenEval
from models.sgdt.scoring_visualize import VisualizeToken
from models.sgdt.scoring_gt import extract_proposals_as_targets
from datetime import datetime
from tqdm import tqdm
from tqdm import tqdm


class AnalyzeSplit:
    def __init__(
            self
    ):
        self.data_root_dir = VISUALIZATION_DIR


    def count_statistics(self):
        batch_size = [2, 4, 8]  # 1 has no split
        dir_list = [f'token_split_view_batchsize{k}_split_pkl' for k in batch_size]

        for dir_name in dir_list:

            dir_path = os.path.join(self.data_root_dir, dir_name)
            pkl_file_list = smrc.utils.get_file_list_in_directory(dir_path)

            histogram_file = os.path.join(self.data_root_dir, dir_name + '_hist.jpg')
            count_file = os.path.join(self.data_root_dir, dir_name + '_count.txt')

            results = []
            pbar = tqdm(enumerate(pkl_file_list))
            for k, pkl_file in pbar:
                # extract the img files
                significance_score, tokens_small_obj, scale_gt = smrc.utils.load_pkl_file(pkl_file)
                significance_score = significance_score.cpu()
                tokens_small_obj = tokens_small_obj.cpu()
                scale_gt = scale_gt.cpu()

                sig_values = scale_gt[tokens_small_obj].view(-1).numpy()
                results += list(sig_values)
                pbar.set_description(f'Processing {k}/{len(pkl_file_list)} ...')

            smrc.utils.save_1d_list_to_file(file_path=count_file, list_to_save=results)
            results_array = np.array(results)
            self.plot_histogram(data=results_array, plot_name=histogram_file,
                                      xlabel='significant_value', ylabel='count',
                                      num_bin=20)

    @staticmethod
    def plot_histogram(data, plot_name,
                       title: str = None, xlabel: str = None, ylabel: str = None, fontsize: int = 14,
                       num_bin: int = None,
                       ):
        if num_bin is not None:
            plt.hist(data, bins=num_bin)  # arguments are passed to np.histogram
        else:
            plt.hist(data, bins='auto')  # arguments are passed to np.histogram
        # plt.title("Histogram with 'auto' bins")
        # Text(0.5, 1.0, "Histogram with 'auto' bins")

        if title is not None:
            plt.title(title, fontsize=fontsize)

        if xlabel is not None:
            plt.xlabel(xlabel, fontsize=fontsize)

        if ylabel is not None:
            plt.ylabel(ylabel, fontsize=fontsize)
        plt.ylim([0, 650000])
        plt.savefig(plot_name)
        plt.close()

    def compare_split_tokens(self):
        batch_size = [1, 2, 4, 8]
        dir_list = [f'token_split_view_batchsize{k}' for k in batch_size]

        for suffix in ['pred_split_mask_overlap.jpg', 'pred_split_token_label_final_overlap.jpg']:
            dir_path = os.path.join(self.data_root_dir, dir_list[0])
            img_list = smrc.utils.get_file_list_in_directory(
                dir_path, ext_str=suffix,
                only_local_name=True
            )

            # 139_pred_split_mask_overlap.jpg
            # 139_pred_split_token_label_final_overlap.jpg
            # 139_pred_split_significance_score_vs_gt.jpg
            result_dir = os.path.join(self.data_root_dir, 'compare_view_' + suffix)
            smrc.utils.generate_dir_if_not_exist(result_dir)

            pbar = tqdm(enumerate(img_list))
            for k, img_name in pbar:
                img_paths_to_plot = [os.path.join(self.data_root_dir, dir_name, img_name) for dir_name in dir_list]
                img_paths_to_plot = [file_path for file_path in img_paths_to_plot if os.path.isfile(file_path)]

                if len(img_paths_to_plot) < 4:
                    continue

                result_img_path = os.path.join(result_dir, img_name)
                if os.path.isfile(result_img_path):
                    continue

                imgs = [cv2.imread(img_path) for img_path in img_paths_to_plot]
                top_img = smrc.utils.merging_two_images_side_by_side(imgs[0], imgs[1])
                bottom_img = smrc.utils.merging_two_images_side_by_side(imgs[2], imgs[3])
                # bottom_img = imgs[-1]
                merged_img = smrc.utils.merging_two_images_top_by_down(top_img, bottom_img)
                cv2.imwrite(result_img_path, merged_img)

                pbar.set_description(f'Processing {k}/{len(img_list)} ...')


if __name__ == "__main__":
    token_ana_tool = AnalyzeSplit()
    # token_ana_tool.count_statistics()
    token_ana_tool.compare_split_tokens()

