from util.plot_utils import plot_logs
import os
from tti.tti_conf import LIB_ROOT_DIR, LIB_RESULT_DIR, LIB_OUTPUT_DIR
logs = [
    os.path.join(LIB_RESULT_DIR, 'sgdt_dn_dab_detr_lr0.5_x2gpus_e5l_sgdt1l_pred_score_sumbel_softmax')
    ]

plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt')