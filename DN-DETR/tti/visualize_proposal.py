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


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
         34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
         63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def coco90_class_to_80class():
    x = coco80_to_coco91_class()
    class_map = {c: k for k, c in enumerate(x)}
    return class_map


class TTICOCOVisualizer(COCOVisualizer):
    def visualize(self, img, tgt, caption=None, dpi=300, savedir=None, show_in_console=True):
        """
        img: tensor(3, H, W)
        tgt: make sure they are all on cpu.
            must have items: 'image_id', 'boxes', 'size'

            139.jpg JPEG 992x721 992x721+0+0 8-bit sRGB 36662B 0.000u 0:00.020
            (smrc) kai@kai:~/cnn/project/DN-DETR/visualize/view/view_det/coco_imgs$ identify 139
            139 (copy).jpg  139.jpg
            (smrc) kai@kai:~/cnn/project/DN-DETR/visualize/view/view_det/coco_imgs$ identify 139\ \(copy\).jpg
            139 (copy).jpg JPEG 1201x873 1201x873+0+0 8-bit sRGB 232620B 0.000u 0:00.020

        """
        plt.figure(dpi=dpi)
        plt.rcParams['font.size'] = '5'
        ax = plt.gca()
        img = renorm(img).permute(1, 2, 0)
        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()
        h, w = img.shape[:2]

        ax.imshow(img)

        self.addtgt(tgt)
        if show_in_console:
            plt.show()

        if savedir is not None:
            if caption is None:
                savename = '{}/{}.jpg'.format(savedir, int(tgt['image_id']))
            else:
                savename = '{}/{}-{}.jpg'.format(savedir, int(tgt['image_id']), caption)
            print("savename: {}".format(savename))
            os.makedirs(os.path.dirname(savename), exist_ok=True)
            # plt.tight_layout()
            plt.savefig(savename, bbox_inches='tight', pad_inches=0, dpi=200)
            image = cv2.imread(savename)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(savename, img=image)
        plt.close()


from models.sgdt.sgdt_module import SGDT_module
import torch



class ProposalViewBatch:
    def __init__(
            self,
            proposal_pkl_file,
    ):

        self.proposal_pkl_file = proposal_pkl_file
        # self.proposal_file = proposal_file
        # self.data_root_dir = data_root_dir
        # self.image_dir = os.path.join(self.data_root_dir, 'coco_imgs')
        # self.json_file_dir = os.path.join(self.data_root_dir, 'coco_det')
        # self.grid_search_result_dir = os.path.join(self.data_root_dir, 'grid_search_result_dir')

        self.coco_class_map = coco90_class_to_80class()




class TokenSplitRemoveAnalysis:
    def __init__(
            self,
            data_root_dir=os.path.join(VISUALIZATION_DIR, 'view'),
            proposal_gt_pkl_file_dir=os.path.join(VISUALIZATION_DIR, 'view/gt_proposal_pkl'),
            proposal_file=os.path.join(VISUALIZATION_DIR, 'view/proposals_original39.3v0.pth'),
    ):

        self.proposal_gt_pkl_file_dir = proposal_gt_pkl_file_dir
        self.proposal_file = proposal_file
        self.data_root_dir = data_root_dir
        self.image_dir = os.path.join(self.data_root_dir, 'coco_imgs')
        self.json_file_dir = os.path.join(self.data_root_dir, 'coco_det')
        self.grid_search_result_dir = os.path.join(self.data_root_dir, 'grid_search_result_dir')
        self.coco_class_map = coco90_class_to_80class()

    def load_data(self):
        """
        Out of memory.
        Returns:

        """
        # load_proposal(self):
        smrc.utils.assert_file_exist(self.proposal_file)
        # proposals_loaded = smrc.utils.load_pkl_file(self.proposal_file)

        det_files = smrc.utils.get_file_list_in_directory(self.proposal_gt_pkl_file_dir)
        results = {}
        for det_file in det_files:
            # extract the img files
            targets, sgdt_target_raw, sgdt_output, sgdt_targets = smrc.utils.load_pkl_file(det_file)

            for k in range(len(targets)):
                single_img_data = {}
                img_id = targets[k]['image_id'].detach().cpu().item()
                # extract the det files.
                single_img_data['target'] = targets[k]
                single_img_data['sgdt_target_raw'] = {key: value[k] for key, value in sgdt_target_raw.items()}
                single_img_data['sgdt_output'] = {key: value[k] for key, value in sgdt_target_raw.items()}
                single_img_data['sgdt_targets'] = {key: value[k] for key, value in sgdt_targets.items()}

                single_img_data['img'] = targets[-1]['input_imgs'][k]
                # single_img_data['proposals'] = proposals_loaded[img_id]
                # #  pred_boxes.append(pred['pred_boxes'])

                results[img_id] = single_img_data

        smrc.utils.generate_pkl_file(pkl_file_name=os.path.join(self.data_root_dir, 'result.pkl'), data=results)
        return results

    def save_img_data(self):
        """
        """
        # load_proposal(self):
        smrc.utils.assert_file_exist(self.proposal_file)
        # proposals_loaded = smrc.utils.load_pkl_file(self.proposal_file)

        det_files = smrc.utils.get_file_list_in_directory(self.proposal_gt_pkl_file_dir)

        image_dir = self.image_dir
        json_file_dir = self.json_file_dir
        smrc.utils.generate_dir_if_not_exist(image_dir)
        smrc.utils.generate_dir_if_not_exist(json_file_dir)

        json_file_path = os.path.join(json_file_dir, f'dn_detr0.394.json')
        cnt = 0
        for det_file in det_files:
            # extract the img files
            targets, sgdt_target_raw, sgdt_output, sgdt_targets = smrc.utils.load_pkl_file(det_file)

            for k in range(len(targets)):
                cnt += 1
                img_id = targets[k]['image_id'].detach().cpu().item()
                img_path = os.path.join(image_dir, f'{img_id}.jpg')
                # extract the det files.
                # single_img_data['target'] = targets[k]
                # single_img_data['sgdt_target_raw'] = {key: value[k] for key, value in sgdt_target_raw.items()}
                # single_img_data['sgdt_output'] = {key: value[k] for key, value in sgdt_target_raw.items()}
                # single_img_data['sgdt_targets'] = {key: value[k] for key, value in sgdt_targets.items()}

                img = targets[-1]['input_imgs'][k]
                # single_img_data['proposals'] = proposals_loaded[img_id]
                # #  pred_boxes.append(pred['pred_boxes'])

                #
                # h, w = list(res['sgdt_output']['feat_map_size'].detach().cpu().numpy())
                #
                # # new_img_size (H_new, W_new)
                # img, new_img_size = self.make_img_size_divisible(img, feat_map_size=(h, w))
                #
                ori_img = img.permute(1, 2, 0).detach().cpu().numpy() * 255
                # results[img_id] = single_img_data
                if not os.path.isfile(img_path):
                    cv2.imwrite(img_path, ori_img)
                print(f'Processing {cnt}th img done.')
        # smrc.utils.generate_pkl_file(pkl_file_name=os.path.join(self.data_root_dir, 'result.pkl'), data=results)
        # return results

    def save_detection(self):
        """
        """
        # load_proposal(self):
        smrc.utils.assert_file_exist(self.proposal_file)
        # proposals_loaded = smrc.utils.load_pkl_file(self.proposal_file)

        det_files = smrc.utils.get_file_list_in_directory(self.proposal_gt_pkl_file_dir)
        image_dir = self.image_dir
        json_file_dir = self.json_file_dir
        smrc.utils.generate_dir_if_not_exist(json_file_dir)
        json_file_path = os.path.join(json_file_dir, f'dn_detr0.394.json')
        cnt = 0
        det_list_v3 = []

        for det_file in det_files:
            # extract the img files
            targets, sgdt_target_raw, sgdt_output, sgdt_targets = smrc.utils.load_pkl_file(det_file)

            for k, img_target in enumerate(targets):
                cnt += 1
                img_id = targets[k]['image_id'].detach().cpu().item()
                img_path = os.path.join(image_dir, f'{img_id}.jpg')

                if 'proposal_boxes' in targets[0]:
                    """
                        targets_proposal[k]['boxes'] = targets[k]['proposal_boxes']
                        targets_proposal[k]['labels'] = targets[k]['proposal_labels']
                        targets_proposal[k]['scores'] = targets[k]['proposal_scores']
                        {"image_path": "test_data/test_images/2/000010.jpg",
                         "category_id": 1, "bbox": [1403, 614, 1635, 742],
                          "score": 0.986344},
                        
                    """
                    box_unnormalized = unnormalize_box(box_normalized=img_target['proposal_boxes'],
                                                       input_img_size=img_target['size'])

                    for box, score, label in zip(
                            list(box_unnormalized.cpu().numpy()),
                            list(targets[k]['proposal_scores'].cpu().numpy()),
                            list(targets[k]['proposal_labels'].cpu().numpy())
                    ):
                        x1, y1, x2, y2 = [int(x) for x in box]
                        det_list_v3.append(
                            {"image_path": img_path, "category_id": self.coco_class_map[int(label)],
                             "bbox": [x1, y1, x2, y2],
                             "score": float(str(score))}
                        )

                print(f'Processing [{cnt}/{len(det_files) * 2}]... ')
            del targets, sgdt_target_raw, sgdt_output, sgdt_targets
        with open(json_file_path, 'w') as fp:
            fp.write(
                '[\n' +
                ',\n'.join(json.dumps(one_det) for one_det in det_list_v3) +
                '\n]')

    def view_images(self):
        from smrc.utils.annotate.img_seq_viewer import ImageSequenceViewer
        viewer_tool = ImageSequenceViewer(
            image_dir=self.data_root_dir,  # image_dir
            # dir_list=['coco_imgs',]
        )

        viewer_tool.main_loop()

    def view_detection(self):
        from smrc.utils.annotate.visualize_detection import VisualizeDetection
        viewer_tool = VisualizeDetection(
            image_dir=os.path.join(self.data_root_dir, 'view_det'),  # image_dir
            detection_format='yolov3',
            class_list_file=os.path.join(self.data_root_dir, 'coco.names'),
            score_thd=0, nms_thd=100,
            auto_load_directory='image_dir',
            json_file_dir=os.path.join(self.data_root_dir, 'coco_det'),
            max_img_num=50,
            # dir_list=['coco_imgs',]
        )

        #  visualize/view -j  -c visualize/view/coco.names -a image_dir
        viewer_tool.main_loop()

    def save_gt_and_proposals(self):

        visualizer = TTICOCOVisualizer()

        smrc.utils.assert_file_exist(self.proposal_file)
        det_files = smrc.utils.get_file_list_in_directory(self.proposal_gt_pkl_file_dir)

        image_gt_dir = os.path.join(self.data_root_dir, 'coco_imgs_with_gt')
        smrc.utils.generate_dir_if_not_exist(image_gt_dir)

        image_proposal_dir = os.path.join(self.data_root_dir, 'coco_imgs_with_proposal')
        smrc.utils.generate_dir_if_not_exist(image_proposal_dir)

        # json_file_dir = self.json_file_dir
        # smrc.utils.generate_dir_if_not_exist(json_file_dir)
        # json_file_path = os.path.join(json_file_dir, f'dn_detr0.394.json')
        # files = smrc.utils.get_file_list_in_directory(
        #     image_proposal_dir, ext_str='.png'
        # )

        cnt = 0
        for det_file in det_files:
            # extract the img files
            targets, sgdt_target_raw, sgdt_output, sgdt_targets = smrc.utils.load_pkl_file(det_file)

            for k in range(len(targets)):
                cnt += 1

                img = targets[-1]['input_imgs'][k].cpu()
                tgt = targets[k]
                tgt = {k: v.detach().cpu() for k, v in tgt.items()}
                visualizer.visualize(img, tgt=tgt, caption=None, savedir=image_gt_dir, show_in_console=False)

                if 'proposal_boxes' in targets[0]:
                    # extract proposals as targets
                    targets_proposal = extract_proposals_as_targets(targets)
                    tgt = targets_proposal[k]
                    tgt = {k: v.detach().cpu() for k, v in tgt.items()}
                    visualizer.visualize(img, tgt=tgt, caption='proposal',
                                         savedir=image_proposal_dir, show_in_console=False)
            del targets, sgdt_target_raw, sgdt_output, sgdt_targets

    def save_split_tokens(self,
                          token_scoring_gt_criterion='significance_value',
                          pad_fg_pixel=16,
                          token_scoring_discard_split_criterion='gt_only_exp-reclaim_padded_region-no_bg_token_remove'
                          ):
        # load_proposal(self):
        smrc.utils.assert_file_exist(self.proposal_file)
        # proposals_loaded = smrc.utils.load_pkl_file(self.proposal_file)

        det_files = smrc.utils.get_file_list_in_directory(self.proposal_gt_pkl_file_dir)

        split_token_dir = os.path.join(self.data_root_dir, 'token_split_dir')
        smrc.utils.generate_dir_if_not_exist(split_token_dir)

        token_scoring_gt_generator = TokenScoringGTGenerator(
            token_scoring_gt_criterion=token_scoring_gt_criterion,
            pad_fg_pixel=pad_fg_pixel
        )

        cnt = 0
        for det_file in det_files:
            # extract the img files  targets, sgdt_target_raw, sgdt_output, sgdt_targets
            targets, _, sgdt_output_old, _ = smrc.utils.load_pkl_file(det_file, verbose=False)
            sgdt_target_raw = token_scoring_gt_generator.get_gt_raw(targets=targets)
            sgdt_targets = token_scoring_gt_generator.resize_sig_value_gt(
                sgdt_target_raw, feat_map_size=sgdt_output_old['feat_map_size'])

            # ------------------------
            # Token adaption module, shared across all layers
            device = sgdt_output_old['x'].device
            N, B, C = sgdt_output_old['x'].shape
            # (N, B) ->  (B, N)
            mask = ~(sgdt_output_old['valid_tokens'].bool()).permute(1, 0)  # torch.zeros((B, N), device=device).bool()
            x = torch.rand_like(sgdt_output_old['x'])

            sgdt_module = SGDT_module(embed_dim=C,
                                      token_scoring_discard_split_criterion=token_scoring_discard_split_criterion)

            sgdt_module.to(device)
            with torch.no_grad():
                #         x: dim: (N, B, C), where N is the number of tokens, B is the batch size,
                #         mask: (B, N), 0 valid locations, True padding locations.
                sgdt_output = \
                    sgdt_module(x=x,
                                mask=mask,
                                sgdt_targets=sgdt_targets,
                                feat_map_size=sgdt_output_old['feat_map_size'],
                                )
                sgdt_output.update(dict(feat_map_size=sgdt_output_old['feat_map_size']))
            # ----------------------

            vis_tool = VisualizeToken(targets=targets,
                                      sgdt_target_raw=sgdt_target_raw,
                                      sgdt_targets=sgdt_targets,
                                      sgdt_output=sgdt_output
                                      )
            # sub_dir =
            vis_tool.visualize_split_token(
                sub_dir=f'view/token_split_dir/{token_scoring_discard_split_criterion}----'
                        f'{token_scoring_gt_criterion}----pad{pad_fg_pixel}'
            )
            del targets, sgdt_target_raw, sgdt_output_old, sgdt_targets, sgdt_output
            print(f'Processing {cnt}th/{len(det_files)} file done.')

            cnt += 1

            if cnt == 100:
                break

    def ana_split(self):
        choices = ['significance_value',
                   # 'significance_value_bg_w_priority',
                   'significance_value_inverse_fg',
                   'fg_scale_class_small_medium_random',
                   'fg_scale_class_all_fg',
                   'fake_all_tokens_are_fg'
                   ]
        for token_scoring_gt_criterion in choices:
            self.save_split_tokens(
                token_scoring_gt_criterion=token_scoring_gt_criterion,
                pad_fg_pixel=16,
                token_scoring_discard_split_criterion='gt_only_exp-reclaim_padded_region-no_bg_token_remove'
            )

    def eval_batch_grid_search(self):
        token_scoring_gt_criterions = [
            'significance_value',
            # 'significance_value_bg_w_priority',
            # 'significance_value_inverse_fg',
            # 'fg_scale_class_small_medium_random',
            'fg_scale_class_all_fg',
            # 'fake_all_tokens_are_fg'
        ]

        # for sampling in ['-proposal_split_gt_filter', '']:  # ['-debug_gt_split_sampling_priority', '']
        #     for scoring in ['-without_scoring_proposal', '']:  #
        #         token_scoring_discard_split_criterion = \
        #             f'gt_only_exp-reclaim_padded_region-no_bg_token_remove' \
        #             f'-use_proposal{sampling}{scoring}'
        #
        #         for token_scoring_gt_criterion in token_scoring_gt_criterions:
        #             self.eval_split_grid_search(
        #                 token_scoring_gt_criterion=token_scoring_gt_criterion,
        #                 pad_fg_pixel=16,
        #                 token_scoring_discard_split_criterion=token_scoring_discard_split_criterion,
        #             )

        # token_scoring_discard_split_criterion = \
        #     f'gt_only_exp-reclaim_padded_region-no_bg_token_remove' \
        #     f'-use_proposal'

        token_scoring_discard_split_criterion = \
            f'gt_only_exp-no_split' \
            f'-use_proposal'
        for token_scoring_gt_criterion in token_scoring_gt_criterions:
            self.eval_split_remove_grid_search(
                token_scoring_gt_criterion=token_scoring_gt_criterion,
                pad_fg_pixel=16,
                token_scoring_discard_split_criterion=token_scoring_discard_split_criterion,
                split_or_remove='remove',
                visualize=True,  # True
            )

    def eval_split_remove_grid_search(
            self,
            token_scoring_gt_criterion='significance_value',
            pad_fg_pixel=16,
            token_scoring_discard_split_criterion='gt_only_exp-reclaim_padded_region-no_bg_token_remove',
            split_or_remove='split',
            visualize=False,
    ):

        assert split_or_remove in ['split', 'remove']
        # evaluate the precision, recall, F1 score for the hyperparamters to find the best setting.
        nms_thds = np.linspace(0.5, 1, 6)
        # min_split_scores = np.linspace(0, 0.5, 6)

        # nms_thds = [0.5]
        # min_split_scores = np.linspace(0.0, 0.9, 10)
        min_split_scores = [0.5]
        min_fg_scores = np.linspace(0.0, 0.9, 10)
        results = []
        result_file = os.path.join(
            self.grid_search_result_dir,
            f'{token_scoring_gt_criterion}-pad{pad_fg_pixel}-{token_scoring_discard_split_criterion}'
            f'-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )
        # f'-{datetime.now().strftime("%Y%m%d-%H%M%S"),}')
        # now.strftime("%m/%d/%Y, %H:%M:%S")
        smrc.utils.generate_dir_for_file_if_not_exist(result_file)

        for use_conf_score in [False]: # , True
            for nms_thd in nms_thds:
                for min_split_score in min_split_scores:
                    for min_fg_score in min_fg_scores:  # min_fg_scores
                        split_metric, remove_metric = self.eval_split_remove(
                            nms_thd=nms_thd,
                            min_split_score=min_split_score,
                            min_fg_score=min_fg_score,
                            token_scoring_gt_criterion=token_scoring_gt_criterion,
                            pad_fg_pixel=pad_fg_pixel,
                            token_scoring_discard_split_criterion=token_scoring_discard_split_criterion,
                            visualize=visualize,
                        )
                        if split_or_remove == 'split':
                            precision, recall, f1_score, accu, tp, fp, fn, tn = [x.cpu().item() for x in split_metric]
                        else:
                            precision, recall, f1_score, accu, tp, fp, fn, tn = [x.cpu().item() for x in remove_metric]
                        print(f'nms_thd={nms_thd}, min_split_score={min_split_score}, min_fg_score={min_fg_score},'
                              f'{precision, recall, f1_score, accu, tp, fp, fn, tn}')
                        results.append([nms_thd, min_split_score, use_conf_score,
                                        precision, recall, f1_score, accu, tp, fp, fn, tn])
        smrc.utils.save_multi_dimension_list_to_file(result_file, list_to_save=results)

    def visualize_remove(
            self,
            token_scoring_gt_criterion='significance_value',
            pad_fg_pixel=16,
            token_scoring_discard_split_criterion='gt_only_exp-reclaim_padded_region-no_bg_token_remove',
            split_or_remove='split',
            visualize=False,
    ):

        assert split_or_remove in ['split', 'remove']
        # evaluate the precision, recall, F1 score for the hyperparamters to find the best setting.
        # nms_thds = np.linspace(0.5, 1, 6)
        nms_thds = [1.0]
        min_split_scores = [0.5]
        min_fg_scores = np.linspace(0.0, 0.9, 10)
        results = []
        result_file = os.path.join(
            self.grid_search_result_dir,
            f'{token_scoring_gt_criterion}-pad{pad_fg_pixel}-{token_scoring_discard_split_criterion}'
            f'-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )
        # f'-{datetime.now().strftime("%Y%m%d-%H%M%S"),}')
        # now.strftime("%m/%d/%Y, %H:%M:%S")
        smrc.utils.generate_dir_for_file_if_not_exist(result_file)

        for use_conf_score in [False]:  # , True
            for nms_thd in nms_thds:
                for min_split_score in min_split_scores:
                    for min_fg_score in min_fg_scores:  # min_fg_scores
                        split_metric, remove_metric = self.eval_split_remove(
                            nms_thd=nms_thd,
                            min_split_score=min_split_score,
                            min_fg_score=min_fg_score,
                            token_scoring_gt_criterion=token_scoring_gt_criterion,
                            pad_fg_pixel=pad_fg_pixel,
                            token_scoring_discard_split_criterion=token_scoring_discard_split_criterion,
                            visualize=visualize,
                        )
                        if split_or_remove == 'split':
                            precision, recall, f1_score, accu, tp, fp, fn, tn = [x.cpu().item() for x in split_metric]
                        else:
                            precision, recall, f1_score, accu, tp, fp, fn, tn = [x.cpu().item() for x in remove_metric]
                        print(f'nms_thd={nms_thd}, min_split_score={min_split_score}, min_fg_score={min_fg_score},'
                              f'{precision, recall, f1_score, accu, tp, fp, fn, tn}')
                        results.append([nms_thd, min_split_score, use_conf_score,
                                        precision, recall, f1_score, accu, tp, fp, fn, tn])
        smrc.utils.save_multi_dimension_list_to_file(result_file, list_to_save=results)

    def eval_split_remove(self, nms_thd=None, min_split_score=None, min_fg_score=None, min_score=None,
                          token_scoring_gt_criterion='significance_value',
                          pad_fg_pixel=16,
                          token_scoring_discard_split_criterion='gt_only_exp-reclaim_padded_region-no_bg_token_remove',
                          debug=True,
                          visualize=False,
                          ):

        smrc.utils.assert_file_exist(self.proposal_file)
        det_files = smrc.utils.get_file_list_in_directory(self.proposal_gt_pkl_file_dir)

        split_token_dir = os.path.join(self.data_root_dir, 'token_split_dir')
        smrc.utils.generate_dir_if_not_exist(split_token_dir)

        token_scoring_gt_generator = TokenScoringGTGenerator(
            token_scoring_gt_criterion=token_scoring_gt_criterion,
            pad_fg_pixel=pad_fg_pixel
        )

        split_evaluator = TokenEval(split_or_remove='split')
        remove_evaluator = TokenEval(split_or_remove='remove')

        if debug:
            det_files = det_files[:500]  # 500

        num_image = len(det_files)
        cnt = 0
        with tqdm(total=num_image) as pbar:
            for det_file in det_files:
                # extract the img files  targets, sgdt_target_raw, sgdt_output, sgdt_targets
                targets, _, sgdt_output_old, _ = smrc.utils.load_pkl_file(det_file, verbose=False)
                sgdt_target_raw = token_scoring_gt_generator.get_gt_raw(
                    targets=targets,
                    nms_thd=nms_thd, min_score=min_score,
                    min_fg_score=min_fg_score,
                    min_split_score=min_split_score,
                )

                sgdt_targets = token_scoring_gt_generator.resize_sig_value_gt(
                    sgdt_target_raw, feat_map_size=sgdt_output_old['feat_map_size']
                )

                # ------------------------
                # Token adaption module, shared across all layers
                device = sgdt_output_old['x'].device
                N, B, C = sgdt_output_old['x'].shape
                # (N, B) ->  (B, N)
                mask = ~(sgdt_output_old['valid_tokens'].bool()).permute(1,
                                                                         0)  # torch.zeros((B, N), device=device).bool()
                x = torch.rand_like(sgdt_output_old['x'])

                sgdt_module = SGDT_module(embed_dim=C,
                                          token_scoring_discard_split_criterion=token_scoring_discard_split_criterion)

                sgdt_module.to(device)
                with torch.no_grad():
                    #         x: dim: (N, B, C), where N is the number of tokens, B is the batch size,
                    #         mask: (B, N), 0 valid locations, True padding locations.
                    sgdt_output = \
                        sgdt_module(x=x,
                                    mask=mask,
                                    sgdt_targets=sgdt_targets,
                                    feat_map_size=sgdt_output_old['feat_map_size'],
                                    )
                    sgdt_output.update(dict(feat_map_size=sgdt_output_old['feat_map_size']))
                # ----------------------
                if visualize:
                    vis_tool = VisualizeToken(targets=targets,
                                              sgdt_target_raw=sgdt_target_raw,
                                              sgdt_targets=sgdt_targets,
                                              sgdt_output=sgdt_output
                                              )
                    # # sub_dir =
                    vis_tool.visualize_split_token(
                        sub_dir=f'view/grid_search_result_dir/{token_scoring_discard_split_criterion}----'
                                f'{token_scoring_gt_criterion}----pad{pad_fg_pixel}'
                                f'nms_thd={nms_thd}, min_score={min_score}, '
                                f'min_fg_score={min_fg_score}, min_split_score={min_split_score}'
                    )
                # tokens_to_discard_original = sgdt_output['tokens_to_discard_original'].long()
                # tokens_to_split_original = sgdt_output['tokens_to_split_original'].long()

                tokens_small_obj = sgdt_output['tokens_small_obj']
                tokens_to_discard = sgdt_output['tokens_to_discard']
                fg_gt, scale_gt = sgdt_targets['fg_gt'], sgdt_targets['scale_gt']
                split_evaluator.eval_single(token_pred=tokens_small_obj, token_gt=scale_gt)
                remove_evaluator.eval_single(token_pred=tokens_to_discard, token_gt=fg_gt)

                del targets, sgdt_target_raw, sgdt_output_old, sgdt_targets, sgdt_output
                # print(f'Processing {cnt}th/{len(det_files)} file done.')

                pbar.set_description(f'| Processing {cnt}th/{len(det_files)} file done.')
                #             # f' avg_iou = {np.average(self.IoUs_)}'
                pbar.update(1)
                cnt += 1
                # if cnt == 500:
                #     break

            # precision_split, recall_split, f1_score_split = split_evaluator._wrap_result()
            # precision_remove, recall_remove, f1_score_remove = remove_evaluator._wrap_result()

        split_metirc = split_evaluator.wrap_result()
        remove_metric = remove_evaluator.wrap_result()
        return split_metirc, remove_metric


class ViewProposal:
    def __init__(self,
                 data_root_dir=os.path.join(VISUALIZATION_DIR, 'view'),
                 proposal_gt_pkl_file_dir=os.path.join(VISUALIZATION_DIR, 'view/gt_proposal_pkl'),
                 proposal_file=os.path.join(VISUALIZATION_DIR, 'view/proposals_original39.3v0.pth'),
                 ):
        self.proposal_gt_pkl_file_dir = proposal_gt_pkl_file_dir
        self.proposal_file = proposal_file
        self.data_root_dir = data_root_dir
        # self.proposal_file_dir = proposal_file_dir

    def extract_img_ids(self):
        files = smrc.utils.get_file_list_in_directory(self.proposal_gt_pkl_file_dir, only_local_name=True)
        img_ids = []
        for file_path in files:
            # _534605_534639.pkl ->  ['', '534605', '534639']
            tmp = smrc.utils.get_basename_prefix(file_path).split('_')[1:]
            img_ids += tmp
        return img_ids

    def load_data(self):
        # load_proposal(self):
        proposals_loaded = smrc.utils.load_pkl_file(self.proposal_file)

        det_files = smrc.utils.get_file_list_in_directory(self.proposal_gt_pkl_file_dir)
        results = {}
        for det_file in det_files:
            # extract the img files
            targets, sgdt_target_raw, sgdt_output, sgdt_targets = smrc.utils.load_pkl_file(det_file)

            for k in len(targets):
                single_img_data = {}
                img_id = targets[k]['image_id'].detach().cpu().item()
                # extract the det files.
                single_img_data['target'] = targets[k]
                single_img_data['sgdt_target_raw'] = {key: value[k] for key, value in sgdt_target_raw.items()}
                single_img_data['sgdt_output'] = {key: value[k] for key, value in sgdt_target_raw.items()}
                single_img_data['sgdt_targets'] = {key: value[k] for key, value in sgdt_targets.items()}

                single_img_data['img'] = targets[-1]['input_imgs'][k]
                single_img_data['proposals'] = proposals_loaded[img_id]
                #  pred_boxes.append(pred['pred_boxes'])

                results[img_id] = single_img_data

        smrc.utils.generate_pkl_file(pkl_file_name=os.path.join(self.data_root_dir, 'result.pkl'), data=results)
        return results


if __name__ == "__main__":
    token_ana_tool = TokenSplitRemoveAnalysis()
    token_ana_tool.save_img_data()
    # token_ana_tool.view_images()
    token_ana_tool.save_gt_and_proposals()
    # token_ana_tool.save_detection()
    # token_ana_tool.view_detection()

    # token_ana_tool.ana_split()
    # token_ana_tool.eval_split_grid_search()
    # token_ana_tool.eval_batch_grid_search()
    # token_ana_tool.visualize_remove(visualize=True)

    # token_ana_tool.save_gt_and_proposals()

