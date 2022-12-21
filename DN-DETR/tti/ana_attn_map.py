
from tti.tti_conf import LIB_RESULT_DIR
import smrc.utils
import os
import torch
import cv2
# from models.sgdt.sgdt_ import KL

data_root_dir = os.path.join(LIB_RESULT_DIR, 'e6-d6-gt_split_only/share_double_head_transformer_ShareV_out_proj_FFN')

checkpoint_attn_map_dirs = smrc.utils.get_dir_list_recursively(data_root_dir)
checkpoint_attn_map_dirs = [x for x in checkpoint_attn_map_dirs if x.find('output_fg_attn_epoch') > -1]
checkpoint_attn_map_dirs = sorted(checkpoint_attn_map_dirs)
output_dir = os.path.join(data_root_dir, 'view_attn_map')
smrc.utils.generate_dir_if_not_exist(output_dir)
for k in range(len(checkpoint_attn_map_dirs)):
    cur_dir = checkpoint_attn_map_dirs[k]
    pred_files = smrc.utils.get_file_list_in_directory(cur_dir, only_local_name=True, ext_str='pkl')

    epoch = os.path.basename(cur_dir).replace('output_fg_attn_epoch', '')
    for pred_file in pred_files[:5]:
        # 8, N, N
        img_id = smrc.utils.get_basename_prefix(pred_file)

        attn_map_cur = torch.load(os.path.join(cur_dir, pred_file))
        num_heads = attn_map_cur.shape[0]

        for h in range(num_heads):
            # plot_name = os.path.join(output_dir, f'{img_id}_{h}_{epoch}.png')
            # head_attn_map = attn_map_cur[h].cpu().numpy()
            # smrc.utils.plot_matrix(plot_name=plot_name, matrix=head_attn_map)

            plot_name = os.path.join(output_dir, f'{img_id}_{h}_{epoch}.jpg')
            N = attn_map_cur[h].shape[1]

            # img[:, :, 0] = ((1 - attn_map_cur[h]).view(N, N, 1).expand(-1, -1, 3) * 255).numpy().astype(int)
            # img = smrc.utils.generate_blank_image(height=N, width=N)
            img = ((1 - attn_map_cur[h]).view(N, N, 1).expand(-1, -1, 3) * 255).numpy().astype(int)
            img[img < 245] = 0
            cv2.imwrite(filename=plot_name, img=img)
            # from PIL import Image
            #
            # im = Image.fromarray(A)
            # im.save("your_file.jpeg")


# checkpoint_attn_map_dirs = sorted(checkpoint_attn_map_dirs)
# output_dir = os.path.join(data_root_dir, 'view_attn_map')
# for k in range(1, len(checkpoint_attn_map_dirs)):
#     pre_dir = checkpoint_attn_map_dirs[k - 1]
#     cur_dir = checkpoint_attn_map_dirs[k]
#     pred_files = smrc.utils.get_dir_list_in_directory(pre_dir, only_local_name=True)
#
#     kl_loss_sum = 0
#     for pred_file in pred_files[:5]:
#         # 8, N, N
#         attn_map_pre = torch.load(os.path.join(pre_dir, pred_file))
#         attn_map_cur = torch.load(os.path.join(cur_dir, pred_file))
#
#         num_heads = attn_map_pre.shape[0]

    # kl_loss = KL(input=attn_map_pre)