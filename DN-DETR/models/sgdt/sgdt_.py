import torch
from torch import nn as nn, nn
import torch.nn.functional as F

from models.sgdt.nms import ProposalProcess
from models.sgdt.scoring_gt import ProposalScoringParser, TokenScoringGTGenerator
from models.sgdt.scoring_loss import TokenScoringLoss
from models.sgdt.sgdt_components import extract_thd, src_key_padding_mask2valid_token_mask
from models.sgdt.sgdt_module import SGDT_module
from models.distillation.losses import FeatureLoss

SIGMA = 0.05


class GTRatioOrSigma:
    def __init__(self, gt_decay_criterion,  # default=None
                 data_size, total_epoch, decay_sigma=False
                 ):
        # for gt proposal fusion
        self.gt_decay_criterion = gt_decay_criterion
        self.data_size = data_size
        self.total_epoch = total_epoch
        self.decay_start_epoch = None
        self.decay_end_epoch = None
        self.total_steps = None
        self.gt_ratio_updater_ready = False
        self.gt_ratio = 1.0

        if self.gt_decay_criterion is not None:
            self.decay_start_epoch = 0
            self.decay_end_epoch = total_epoch

            if gt_decay_criterion != '':
                decay_start_epoch = extract_thd(gt_decay_criterion, 'start_epoch')
                if decay_start_epoch is not None:
                    assert isinstance(decay_start_epoch, (int, float)) and decay_start_epoch >= 0
                    self.decay_start_epoch = decay_start_epoch

                decay_end_epoch = extract_thd(gt_decay_criterion, 'end_epoch')
                if decay_end_epoch is not None:
                    assert isinstance(decay_end_epoch, (int, float)) and decay_end_epoch > 0
                    self.decay_end_epoch = decay_end_epoch
            assert self.decay_start_epoch < self.decay_end_epoch < self.total_epoch

            self.total_steps = self.data_size * (self.decay_end_epoch - self.decay_start_epoch)
            self.gt_ratio_updater_ready = True
        print(f'self.gt_ratio_updater_ready = {self.gt_ratio_updater_ready}')

        self.sigma_max = SIGMA
        self.sigma = SIGMA
        self.decay_sigma = decay_sigma

    def update_gt_ratio(self, cur_epoch, cur_iter):
        if self.gt_decay_criterion is not None:
            if cur_epoch < self.decay_start_epoch:
                self.gt_ratio = 1.0
            elif cur_epoch >= self.decay_end_epoch:
                self.gt_ratio = 0
            else:
                cur_step = (cur_epoch - self.decay_start_epoch) * self.data_size + cur_iter
                self.gt_ratio = 1 - cur_step / self.total_steps

    def update_sigma(self, cur_epoch, cur_iter):
        if self.decay_sigma:
            total_steps = self.data_size * self.total_epoch
            cur_step = cur_epoch * self.data_size + cur_iter
            process = cur_step / total_steps
            sigma_multiplier = 1 - process
            self.sigma = SIGMA * sigma_multiplier

    def update(self, cur_epoch, cur_iter):
        self.update_gt_ratio(cur_epoch=cur_epoch, cur_iter=cur_iter)
        self.update_sigma(cur_epoch=cur_epoch, cur_iter=cur_iter)


def init_proposal_processor(proposal_scoring):
    if proposal_scoring is not None:
        proposal_scoring_parser = ProposalScoringParser(proposal_scoring_config=proposal_scoring)
        proposal_filtering_param = proposal_scoring_parser.extract_box_filtering_parameter()
        proposal_processor = ProposalProcess(**proposal_filtering_param)
    else:
        proposal_processor = ProposalProcess()

    return proposal_processor


def KL(input, target, valid_tokens_float=None, top_k=None):
    """https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss
    F.softmax(attn_weights, dim=-1)  in self-attention, so dim should be -1
    Args:
        input:
        target:
                torch.Size([616, 2]) N, B
            a = valid_tokens_float[:, 0].bool()
            b = a.sum()
            c = valid_mask[0, a].sum()
            d = 513 * 513
    Returns:

    """
    input = input.float()  # torch.Size([2, 8, 888, 888])
    target = target.float()  # torch.Size([2, 8, 888, 888])
    # loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32),
    #                 F.softmax(target, dim=-1, dtype=torch.float32))
    bsz, num_heads, src_len = input.shape[:3]
    if valid_tokens_float is not None:  # N, B
        valid_mask = torch.bmm(valid_tokens_float.transpose(1, 0).view(bsz, src_len, 1),
                               valid_tokens_float.transpose(1, 0).view(bsz, 1, src_len),
                               )  # B, N, N
        valid_mask = valid_mask.view(bsz, 1, src_len, src_len).expand(-1, num_heads, -1, -1)

        # B, N_Head, N, N  torch.Size([2, 8, 725, 725])
        # valid_mask = valid_tokens_float.transpose(1, 0).view(bsz, 1, src_len, 1).expand(-1, num_heads, -1, src_len)

        if top_k is not None and top_k > 0:
            src_mask = torch.ones_like(valid_mask)
            input_topk_indices = torch.topk(input, k=top_k, dim=-1).indices
            input_mask = torch.zeros_like(valid_mask).scatter_(-1, index=input_topk_indices, src=src_mask)
            target_topk_indices = torch.topk(target, k=top_k, dim=-1).indices
            target_mask = torch.zeros_like(valid_mask).scatter_(-1, index=target_topk_indices, src=src_mask)
            # invalid token may also included in topk
            final_topk_mask = torch.logical_or(input_mask.bool(), target_mask.bool())
            valid_mask = valid_mask * final_topk_mask.float()

        weight = valid_tokens_float.sum() / torch.ones_like(valid_tokens_float).sum()
        loss = (F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32),
                         F.softmax(target, dim=-1, dtype=torch.float32),
                         reduction='none') * valid_mask).sum() / (bsz * num_heads * src_len * weight)
    else:
        raise NotImplementedError
        # # loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32),
        # #                 F.softmax(target, dim=-1, dtype=torch.float32),
        # #                 reduction='mean')
        # loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32),
        #                 F.softmax(target, dim=-1, dtype=torch.float32),
        #                 reduction='sum') / (bsz * num_heads * src_len)

    return loss


def MSE(input, target, valid_tokens_float=None):
    """https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss

    Returns:

    """
    input = input.float()  # N, B, C
    target = target.float()  # N, B, C
    loss_func = nn.MSELoss(reduction='none')

    N, B, C = input.shape
    if valid_tokens_float is not None:  # N, B; torch.Size([713, 2])
        valid_mask = valid_tokens_float.view(N, B, 1).expand(-1, -1, C)
        weight = valid_tokens_float.sum() / torch.ones_like(valid_tokens_float).sum()
        loss = (loss_func(input, target) * valid_mask).sum() / (B * N * weight)
        # tensor(57.0677, device='cuda:0', grad_fn=<DivBackward0>)
    else:
        raise NotImplementedError
    return loss


class TokenClassifier(nn.Module):

    def __init__(self, embed_dim, num_class=2):  # channel dim, also is the feature dimension for each token
        super().__init__()
        assert isinstance(embed_dim, int)

        hidden_dim = embed_dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_class),  # 3 classes.
            # nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, feat_map_size=None, with_global_feat=False, mask=None):
        """ Significant value prediction, < 0.5, bg, > 0.5 fg (smaller object, large significance).
        Args:
            x: dim: N, B, C, where N is the number of tokens, B is the batch size,
            C is the channel dimension.
        Returns:
        """
        # torch.Size([750, 2, 256])
        return self.mlp(x)  # torch.Size([650, 2, 3])


class FgBgClassifier(nn.Module):
    """ This is the DAB-DETR module that performs object detection """

    def __init__(self, embed_dim, ):
        super().__init__()

        self.token_classifier = TokenClassifier(embed_dim=embed_dim)

    def forward(self, teacher_encoder_output_list, ):
        classification_logits = []
        for k, teacher_encoder_output in enumerate(teacher_encoder_output_list):
            x = teacher_encoder_output['feat']
            pred = self.token_classifier(x)
            classification_logits.append(pred)  # torch.Size([750, 2, 2])

        out = {'sgdt_token_classification_output_list': classification_logits}

        # ================ Visualization purpose
        softmax = nn.Softmax(dim=-1)
        prob = softmax(classification_logits[-1])[:, -1]
        tokens_to_split = classification_logits[-1].max(dim=-1).indices
        tokens_to_discard = 1 - tokens_to_split
        vis_data = {
            'tokens_small_obj': tokens_to_split,
            'tokens_to_discard': tokens_to_discard,
            # 'valid_tokens': valid_tokens_float,  # TODO: change this to bool()

            # valid_tokens in the original size, this is only used for loss calculation.
            # 'valid_tokens_original': valid_tokens_original_float,
            'tokens_to_discard_original': tokens_to_discard,
            'tokens_to_split_original': tokens_to_split,
            'fg_score': prob,
            'small_scale_score': prob,
        }
        out.update(dict(sgdt_vis_data=vis_data))
        return out


class SGDT(nn.Module):
    # from fgd setting
    # tensor(0.0542, device='cuda:1', grad_fn=<AddBackward0>) loss in the beginning
    # temp = 0.5
    # alpha_fgd = 0.00005
    # beta_fgd = 0.000025
    # gamma_fgd = 0.00005
    # lambda_fgd = 0.0000005

    # tensor(0.7360, device='cuda:1', grad_fn=<AddBackward0>) loss in the beginning
    # temp = 0.5
    # alpha_fgd = 0.001
    # beta_fgd = 0.0005
    # gamma_fgd = 0.001
    # lambda_fgd = 0.000005

    def __init__(self, args):
        super(SGDT, self).__init__()

        # ----------------
        self.token_scoring_gt_criterion = args.token_scoring_gt_criterion
        self.token_scoring_gt_generator = TokenScoringGTGenerator(
            token_scoring_gt_criterion=args.token_scoring_gt_criterion,
            pad_fg_pixel=args.pad_fg_pixel,
            proposal_scoring=args.proposal_scoring,
            proposal_token_scoring_gt_criterion=args.proposal_token_scoring_gt_criterion,
        )

        self.targets = None
        self.sgdt_target_raw = None
        self.sgdt_targets = None
        self.feat_map_size = None

        # self.sgdt_loss_weight = sgdt_loss_weight
        # assert isinstance(self.sgdt_loss_weight, (int, float))
        self.disable_fg_scale_supervision = args.disable_fg_scale_supervision
        self.token_scoring_loss = None
        if args.token_scoring_loss_criterion is not None and args.token_scoring_loss_criterion:
            self.token_scoring_loss = TokenScoringLoss(
                token_scoring_loss_criterion=args.token_scoring_loss_criterion
            )
        self.token_adaption_visualization = args.token_adaption_visualization
        self.visualization_out_sub_dir = args.visualization_out_sub_dir

        self.attention_map_evaluation = args.attention_map_evaluation

        self.feature_distiller_loss = None

        self.distiller = args.feature_attn_distillation
        self.feature_distillation_teacher_feat_with_grad = args.feature_distillation_teacher_feat_with_grad
        # self.teacher_encoder_output_list = None  # for loss calculation
        if args.feature_attn_distillation:
            temp = 0.5  # official setting is 0.8
            alpha_fgd = 0.0016
            beta_fgd = 0.0008
            gamma_fgd = 0.0008
            lambda_fgd = 0.000008
            if args.feature_attn_distillation == 'separate_trained_model':
                temp = 0.5
                alpha_fgd = 0.001
                beta_fgd = 0.0005
                gamma_fgd = 0.001
                lambda_fgd = 0.000005

                # temp = 0.5
                # alpha_fgd = 0.00005
                # beta_fgd = 0.000025
                # gamma_fgd = 0.00005
                # lambda_fgd = 0.0000005
            # else:
            #     # tensor(0.7360, device='cuda:1', grad_fn=<AddBackward0>) loss in the beginning
            #     temp = 0.5
            #     alpha_fgd = 0.001
            #     beta_fgd = 0.0005
            #     gamma_fgd = 0.001
            #     lambda_fgd = 0.000005

            self.feature_distiller_loss = FeatureLoss(
                student_channels=256,  # TODO: adapt this later
                teacher_channels=256,  # TODO: adapt this later
                name='feature_loss',
                temp=temp,
                alpha_fgd=alpha_fgd,
                beta_fgd=beta_fgd,
                gamma_fgd=gamma_fgd,
                lambda_fgd=lambda_fgd,
            )

        self.with_sgdt_attention_loss = args.with_sgdt_attention_loss
        self.sgdt_attention_loss = KL
        self.with_sgdt_transformer_feature_distill_loss = args.with_sgdt_transformer_feature_distill_loss
        self.sgdt_transformer_feature_distill_loss = MSE
        # proposal_processor is always set.
        self.proposal_processor = init_proposal_processor(args.proposal_scoring)

        # the parameters of SGDT.
        self.sgdt_module = None
        if args.token_scoring_discard_split_criterion is not None and args.token_scoring_discard_split_criterion != '':
            self.sgdt_module = SGDT_module(
                # embed_dim=d_model,
                embed_dim=args.hidden_dim,
                token_scoring_discard_split_criterion=args.token_scoring_discard_split_criterion,
            )

        # for gt proposal fusion
        self.gt_decay_criterion = args.gt_decay_criterion  # start_epoch8-end_epoch11
        # self.data_size = None
        # self.total_epoch = None
        # self.decay_start_epoch = None
        # self.decay_end_epoch = None
        # self.total_steps = None
        # self.gt_ratio_updater_ready = False
        # self.gt_ratio = 1.0
        self.gt_ratio_or_sigma = None
        self.encoder_layer_config = args.encoder_layer_config
        self.src_key_padding_mask = None
        # self.with_teacher_model
        # self.src_key_padding_mask = src_key_padding_mask  # B, H, W; to change to B, N use .flatten(1)

        self.marking_encoder_layer_fg1_bg0 = args.marking_encoder_layer_fg1_bg0

        self.token_fg_bg_classier = None
        if args.auxiliary_fg_bg_cls_encoder_layer_ids is None:
            self.auxiliary_fg_bg_cls_encoder_layer_ids = []
        elif args.auxiliary_fg_bg_cls_encoder_layer_ids and \
                not isinstance(args.auxiliary_fg_bg_cls_encoder_layer_ids, list):
            self.auxiliary_fg_bg_cls_encoder_layer_ids = [args.auxiliary_fg_bg_cls_encoder_layer_ids]
        else:
            self.auxiliary_fg_bg_cls_encoder_layer_ids = args.auxiliary_fg_bg_cls_encoder_layer_ids

        if len(self.auxiliary_fg_bg_cls_encoder_layer_ids) > 0:
            self.token_fg_bg_classier = FgBgClassifier(embed_dim=args.hidden_dim, )

        # used to freeze the w_q, w_k, w_q_teacher, w_k_teacher in Transformer.Encoder block.
        self.freeze_online_encoder_distillation = args.freeze_online_encoder_distillation
        self.attn_distillation_teacher_with_grad = args.attn_distillation_teacher_with_grad
        self.freeze_attn_online_encoder_distillation = args.freeze_attn_online_encoder_distillation
        # self.double_head_transformer = args.double_head_transformer

        self.training_only_distill_student_attn = args.training_only_distill_student_attn \
                                                  or args.training_only_distill_student_attn_not_free_backbone

        self.attention_loss_top_100_token = args.attention_loss_top_100_token
        self.debug_st_attn_sweep_n_attn_heads = args.debug_st_attn_sweep_n_attn_heads
        self.debug_st_attn_sweep_fg_attn = args.debug_st_attn_sweep_fg_attn
        self.debug_st_attn_sweep_bg_attn = args.debug_st_attn_sweep_bg_attn
        self.args = args

    def set_sgdt_targets(self, targets, feat_map_size):
        # feat_map_size =(h, w)
        self.feat_map_size = feat_map_size
        self.targets = targets

        if self.token_scoring_gt_criterion:
            self.sgdt_target_raw = self.token_scoring_gt_generator.get_gt_raw(targets=targets)
            self.sgdt_targets = self.token_scoring_gt_generator.resize_sig_value_gt(
                self.sgdt_target_raw, feat_map_size)

    def update_proposal_gt(self, selected_proposals):
        assert self.sgdt_target_raw is not None, 'self.sgdt_target_raw should have been generated already.'

        self.targets, self.sgdt_target_raw = self.token_scoring_gt_generator.update_proposal_gt_raw(
            targets=self.targets, selected_proposals=selected_proposals,
            sgdt_target_raw=self.sgdt_target_raw)
        self.sgdt_targets = self.token_scoring_gt_generator.resize_sig_value_gt(
            self.sgdt_target_raw, self.feat_map_size)

    def get_input_img_sizes(self):
        return torch.stack([t['size'] for t in self.targets], dim=0)

    def set_gt_ratio_or_sigma(self, gt_ratio_or_sigma: GTRatioOrSigma):
        self.gt_ratio_or_sigma = gt_ratio_or_sigma

    @property
    def gt_ratio(self):
        if self.gt_ratio_or_sigma is not None and self.gt_ratio_or_sigma.gt_ratio_updater_ready:
            return self.gt_ratio_or_sigma.gt_ratio
        else:
            return None

    @property
    def valid_tokens_float(self):

        valid_tokens = src_key_padding_mask2valid_token_mask(self.src_key_padding_mask)
        return valid_tokens.float()

    @property
    def sigma(self):
        if self.gt_ratio_or_sigma is not None:
            return self.gt_ratio_or_sigma.sigma
        else:
            return None

    def forward(self, x, mask):

        return self.sgdt_module(
            x=x, mask=mask,
            sgdt_targets=self.sgdt_targets,
            feat_map_size=self.feat_map_size,  # feat_map_size = (h, w)
            sigma=self.sigma,
            gt_ratio=self.gt_ratio
            # reclaim_padded_region=self.reclaim_padded_region
        )  # torch.Size([630, 2, 256]),torch.Size([2, 630])
    # -------------------------------
    # def update_sigma(self, cur_step, total_steps):
    #     process = cur_step / total_steps
    #     sigma_multiplier = 1 - process
    #     self.sigma = self.sigma_max * sigma_multiplier


def build_sgdt(args):
    return SGDT(args)
