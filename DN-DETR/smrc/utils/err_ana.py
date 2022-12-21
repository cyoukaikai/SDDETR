import cv2

from . import load_image_list
from .eval import *
from .file_path import file_path_last_two_level, generate_dir_if_not_exist
from .load_save import load_1d_list_from_file
from .color import RED, BLACK, WHITE, GREEN, YELLOW


class ErrorAnalyze(Eval):
    def __init__(self, class_list_file=None):
        super().__init__()
        self.LINE_THICKNESS = 2

        # self.annotation_tool =
        self.CLASS_NAME_LIST = []
        if class_list_file is not None and os.path.isfile(class_list_file):
            self.CLASS_NAME_LIST = load_1d_list_from_file(class_list_file)
        print(f'self.CLASS_NAME_LIST = {self.CLASS_NAME_LIST}')

        self.pred_tp_color, self.pred_fp_color = WHITE, YELLOW  # black RED, BLUE, BLACK, WHITE, GREEN, YELLOW
        self.gt_fn_color, self.gt_tp_color = RED, GREEN
        self.pred_colors, self.gt_colors = [self.pred_fp_color, self.pred_tp_color], [self.gt_fn_color, self.gt_tp_color]
        self.fp_tp_text_list, self.fn_tp_text_list = ['FP', 'TP'], ['FN', 'TP']

    def _error_analyze(
            self, image_path_list, det_file_list, gtruth_file_list, result_root_dir,
            iou_thd=0.5):
        """
        Evaluate the precision, recall and F1 score.
        :param det_file_list: complete det file list
        :param gtruth_file_list: complete gtruth file list
        :param iou_thd:
        :return:
        """

        history = []
        self._init_metric()
        assert len(image_path_list) == len(gtruth_file_list) and len(image_path_list) == len(det_file_list)
        num_image = len(image_path_list)

        with tqdm(total=num_image) as pbar:
            for image_path, det_file, gtruth_file in \
                    zip(image_path_list, det_file_list, gtruth_file_list):
                preds = load_bbox_from_file(det_file)
                gts = load_bbox_from_file(gtruth_file)

                singe_image_metric = eval_single_image(preds=preds, gts=gts, iou_thd=iou_thd)
                self._update_metric(singe_image_metric)

                tp, fp, fn, IoUs = singe_image_metric

                # only handle the errors
                if fp > 0 or fn > 0:
                    pred_labels, gt_labels = classify_single_img_dets(
                        preds=preds, gts=gts, iou_thd=iou_thd
                    )
                    history.append(
                        [image_path, det_file, gtruth_file] + singe_image_metric
                    )
                    result_image_path = os.path.join(result_root_dir, file_path_last_two_level(image_path))

                    # draw the result on image

                    legend = f'{file_path_last_two_level(image_path)}\n' \
                                   f'TP = {tp}, FP = {fp}, FN = {fn}'
                    img = self.draw_errors(
                        img_path=image_path, preds=preds,
                        gts=gts, pred_labels=pred_labels, gt_labels=gt_labels,
                        legend=legend
                    )
                    generate_dir_if_not_exist(os.path.dirname(result_image_path))
                    cv2.imwrite(result_image_path, img)
                pbar.set_description(f'| tp = {self.TP_}, fp = {self.FP_}, fn = {self.FN_}')
                pbar.update(1)
        precision, recall, f1_score = self._wrap_result()

        return history, precision, recall, f1_score

    def draw_errors(self, img_path, preds, gts, pred_labels, gt_labels, legend):
        pred_colors, gt_colors = self.pred_colors, self.gt_colors
        fp_tp_text_list, fn_tp_text_list = self.fp_tp_text_list, self.fn_tp_text_list

        line_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX  # FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        margin = 10
        text_width, text_height = cv2.getTextSize(legend, font, font_scale, line_thickness)[0]

        tmp_img = cv2.imread(img_path)
        for bbox, correct_flag in zip(gts, gt_labels):
            _, xmin, ymin, xmax, ymax = bbox
            cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax),
                          gt_colors[correct_flag], line_thickness)
            if not correct_flag:
                pred_label_text = fn_tp_text_list[correct_flag]
                self.draw_class_name(tmp_img, (xmin, ymax + text_height + margin),
                                     pred_label_text, BLACK, gt_colors[correct_flag])  #

        for bbox, correct_flag in zip(preds, pred_labels):
            class_idx, xmin, ymin, xmax, ymax = bbox
            cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax),
                          pred_colors[correct_flag], line_thickness)
            if class_idx < len(self.CLASS_NAME_LIST):
                text = self.CLASS_NAME_LIST[class_idx]
            else:
                text = 'class ' + str(class_idx)

            if not correct_flag:
                self.draw_class_name(tmp_img, (xmin, ymin), text, YELLOW, BLACK)  #
                pred_label_text = fp_tp_text_list[correct_flag]
                self.draw_class_name(tmp_img, (xmin, ymax + text_height + margin),
                                     pred_label_text, BLACK, pred_colors[correct_flag])  #

        self.put_text_on_image(
            tmp_img, text_content=legend, thickness=4, font_scale=1.5
        )
        return tmp_img

    def put_text_on_image(
            self, tmp_img, text_content, x0=20, y0=60, dy=50,
            font_color=(0, 0, 255), thickness=2, font_scale=0.6
    ):
        # thickness = 4, font_scale = 1.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, text in enumerate(text_content.split('\n')):
            y = y0 + i * dy
            if i == 0:
                cv2.putText(tmp_img, text, (x0, y), font,
                            font_scale, color=font_color, thickness=thickness)
            if i == 1:  # second row
                text_width_total = x0
                sub_text_list = text.split(',')  # TP = {tp}, FP = {fp}, FN = {fn}
                for sub_text, color in zip(sub_text_list, [self.pred_tp_color, self.pred_fp_color, self.gt_fn_color]):
                    text_width, text_height = cv2.getTextSize(sub_text, font, font_scale, thickness)[0]
                    cv2.putText(tmp_img, sub_text, (text_width_total, y), font,
                                font_scale, color=color, thickness=thickness)
                    text_width_total += text_width

    def draw_class_name(self, tmp_img, location_to_draw, text_content, text_shadow_color, text_color):
        font = cv2.FONT_HERSHEY_SIMPLEX  # FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        margin = 3
        text_width, text_height = cv2.getTextSize(text_content, font, font_scale, self.LINE_THICKNESS)[0]

        xmin, ymin = int(location_to_draw[0]), int(location_to_draw[1])
        # cv2.rectangle(tmp_img, (xmin, ymin), (xmin + text_width + margin, ymin - text_height - margin),
        #               text_shadow_color, -1)

        cv2.putText(tmp_img, text_content, (xmin, ymin - 5), font, font_scale, text_color, self.LINE_THICKNESS,
                    int(cv2.LINE_AA))

    def error_analyze_img_root_dir(
            self, image_root_dir, det_root_dir, gt_root_dir,
            result_root_dir=None, dir_list=None, iou_thd=0.5):

        if dir_list is None:
            dir_list = get_dir_list_in_directory(image_root_dir)
        assert len(dir_list) > 0
        print(f'| Total {len(dir_list)} directories. ')
        test_image_path_list = load_image_list(image_root_dir, dir_list=dir_list)

        print(f'| Total {len(test_image_path_list)} image files. ')
        det_file_list, gtruth_file_list = get_det_gt_txt_file_list(
            test_image_path_list=test_image_path_list,
            image_root_dir=image_root_dir,
            det_root_dir=det_root_dir,
            gt_root_dir=gt_root_dir)
        self._error_analyze(
            image_path_list=test_image_path_list,
            det_file_list=det_file_list,
            gtruth_file_list=gtruth_file_list, result_root_dir=result_root_dir,
            iou_thd=iou_thd
        )

    def error_analyze_img_path_list(
            self, test_image_path_list, image_root_dir, det_root_dir, gt_root_dir,
            result_root_dir=None, iou_thd=0.5):

        assert len(test_image_path_list) > 0
        print(f'| Total {len(test_image_path_list)} image files. ')
        det_file_list, gtruth_file_list = get_det_gt_txt_file_list(
            test_image_path_list=test_image_path_list,
            image_root_dir=image_root_dir,
            det_root_dir=det_root_dir,
            gt_root_dir=gt_root_dir)
        self._error_analyze(
            image_path_list=test_image_path_list,
            det_file_list=det_file_list,
            gtruth_file_list=gtruth_file_list, result_root_dir=result_root_dir,
            iou_thd=iou_thd
        )
