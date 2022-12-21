import torch
import torch.nn as nn

from models.sgdt.sgdt_components import SGDTConfigParse
from models.sgdt.token_scoring import TokenFGSmallScaleScoring, TokenFGScoringSigmoid, TokenFGScoringSoftmax, \
    TokenFGScoringWGF, TokenScoringWGFDynamicViT, TokenScoringConv, TokenScoringPredictorLG

# FG_SIGNIFICANCE_THD = 0.6
# BG_SIGNIFICANCE_THD = 0.3  # only used in this file

FG_SIGNIFICANCE_THD = 0.0
BG_SIGNIFICANCE_THD = 1.0
EPSILON = 1e-5  # 1e-8


class TokenSplit(nn.Module):
    """ Importance Score (foreground score), Object scale Predictor

    How about changing this to a bottleneck structure?
            self.fg_scoring = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )
    """

    def __init__(self, embed_dim, expand=1):  # channel dim, also is the feature dimension for each token
        super().__init__()
        assert isinstance(embed_dim, int)

        self.expand = expand
        assert self.expand in [1, 2], 'expand not in [1, 2] can not be handled yet.'

        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expand),  # C -> 2C
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """

        Args:
            x: dim: N, B, C, where N is the number of tokens, B is the batch size,
            C is the channel dimension.

        Returns:

        """

        if self.expand == 1:
            z = self.linear(x)
            return z  # torch.split(z, C, dim=-1)
        else:
            C = x.shape[-1]
            z = self.linear(x)
            # z[:,:, :C], z[:,:, C:]
            # Splits the tensor into chunks. Each chunk is a view of the original tensor.
            return torch.split(z, C, dim=-1)


def get_valid_token_mask(x, mask=None):
    """

    Args:
        x: dim: (N, B, C), where N is the number of tokens, B is the batch size,
        mask: (B, N), 0 valid locations, True padding locations.

    Returns: bool()

    """
    N, B, C = x.shape
    if mask is None:
        valid_tokens = torch.ones(N, B).to(x.device).bool()
    else:
        valid_tokens = ~mask.permute(1, 0)  # (B, N) -> (N, B)
    return valid_tokens


class TokenScoringConfigParser(SGDTConfigParse):
    def __init__(self, token_scoring_discard_split_criterion):
        super().__init__(config_str=token_scoring_discard_split_criterion)

    @property
    def reclaim_padded_region(self):  # padded tokens, default: not claim
        return self.str_exist('reclaim_padded_region')

    @property
    def no_split(self):  # fg tokens, default: split
        return self.str_exist('no_split')

    @property
    def no_bg_token_remove(self):
        return self.str_exist('no_bg_token_remove')

    @property
    def predict_significance(self):
        # 'pred_significance_all_fg_w_priority':
        return self.str_exist('pred_significance')

    @property
    def with_global_feat(self):
        return self.str_exist('w_gf')

    @property
    def inverse_remove_thd(self):
        return self.str_exist('inverse_remove_thd')

    @property
    def inverse_split_thd(self):
        return self.str_exist('inverse_split_thd')

    @property
    def filter_false_remove(self):
        return self.str_exist('filter_false_remove')

    @property
    def filter_false_split(self):
        return self.str_exist('filter_false_split')

    @property
    def bg_sig_thd(self):
        discard_split_configs = self.config_str.split('-')
        bg_significance_thd = None
        for v in discard_split_configs:
            if v.find('bg_sig_thd') > -1:
                bg_significance_thd = float(v.split('bg_sig_thd')[-1])
                break
        return bg_significance_thd

    @property
    def split_sig_thd(self):
        discard_split_configs = self.config_str.split('-')
        split_significance_thd = None
        for v in discard_split_configs:
            if v.find('split_sig_thd') > -1:
                split_significance_thd = float(v.split('split_sig_thd')[-1])
                break
        return split_significance_thd

    @property
    def pred_score(self):
        """
        Making predictions or not
        if  token_scoring_discard_split_criterion in ['v0_with_gt',
                                 'v0_with_gt_and_reclaim_padded_region',
                                 'v0_with_gt_only_reclaim_padded_region'
                                          ]):
             return True

        Returns:


        """
        if self.config_str.find('gt_only_exp') > -1:
            return False
        else:
            return True

    def pred_fg_and_small_scale(self):
        """
        Making predictions or not
        if  token_scoring_discard_split_criterion in ['v0_with_gt',
                                 'v0_with_gt_and_reclaim_padded_region',
                                 'v0_with_gt_only_reclaim_padded_region'
                                          ]):
             return True

        Returns:


        """
        if self.config_str in [
            'v1_selection_differentiable',  # 'test_but_scoring_grad_for_loss_only'
        ]:
            return True
        else:
            return False

    def token_split_(self):
        """
        Making predictions or not
        if  token_scoring_discard_split_criterion in ['v0_with_gt',
                                 'v0_with_gt_and_reclaim_padded_region',
                                 'v0_with_gt_only_reclaim_padded_region'
                                          ]):
             return True

        Returns:


        """
        if self.config_str in [
            'v0_with_gt_only_remove', 'pred_significance_all_fg_w_priority_only_remove',
        ]:
            return False
        elif self.no_split:
            return False
        else:
            return True


class SGDT_module(nn.Module):
    """
    Token adaption module, input: a set of n tokens
    output: a set of n tokens

    1. token scoring
    2. scoring based token merging, split, removal
    """

    # img_size=224, tokens_type='performer', in_chans=3, , token_dim=64
    def __init__(self, embed_dim, max_split_token_num=10000,  # 10000  300
                 max_split_token_num_inference=10000,  # 300
                 token_scoring_discard_split_criterion=None,
                 ):
        super().__init__()
        assert token_scoring_discard_split_criterion is not None

        self.token_scoring_discard_split_criterion = token_scoring_discard_split_criterion
        self.token_scoring_config_parser = TokenScoringConfigParser(
            token_scoring_discard_split_criterion)

        # TODO: clean the code in this section.
        if self.token_scoring_config_parser.pred_score:
            if self.token_scoring_config_parser.str_exist('pred_token_fg_dynamic_vit'):
                self.token_scoring = TokenScoringWGFDynamicViT(embed_dim=embed_dim)
            elif self.token_scoring_config_parser.str_exist('pred_token_fg_conv'):
                self.token_scoring = TokenScoringConv(embed_dim=embed_dim)
            elif self.token_scoring_config_parser.str_exist('pred_token_fg_wgf'):
                self.token_scoring = TokenFGScoringWGF(embed_dim=embed_dim)
            elif self.token_scoring_config_parser.str_exist('pred_token_fg'):
                self.token_scoring = TokenFGScoringSoftmax(embed_dim=embed_dim)
            elif self.token_scoring_config_parser.pred_fg_and_small_scale():
                self.token_scoring = TokenFGSmallScaleScoring(embed_dim=embed_dim)
            elif self.token_scoring_config_parser.predict_significance:
                self.token_scoring = TokenFGScoringSigmoid(embed_dim=embed_dim)
            elif self.token_scoring_config_parser.str_exist('topk_token_selection_differentiable'):
                # out_channels=1 for sigmoid use
                self.token_scoring = TokenScoringPredictorLG(embed_dim=embed_dim, out_channels=1)
            elif self.token_scoring_config_parser.str_exist('gumbel_softmax_token_selection_differentiable'):
                # out_channels=2 for softmax use
                self.token_scoring = TokenScoringPredictorLG(embed_dim=embed_dim, out_channels=2)
            else:
                raise NotImplementedError
        else:
            self.token_scoring = None

        if self.token_scoring_config_parser.token_split_():
            self.token_split_conv = TokenSplit(embed_dim=embed_dim, expand=1)
        else:
            self.token_split_conv = None

        self.max_split_token_num = max_split_token_num
        self.max_split_token_num_inference = max_split_token_num_inference
        self.num_samples = 100  # for top k token selection

    @staticmethod
    def _generate_src_key_padding_mask(invalid_tokens, tokens_to_discard):
        """ no matter we reclaim padded tokens or not.
        Args:
            tokens_to_discard:  (N, B), bool, True means locations to be removed, False, locations to keep.
            invalid_tokens: (N, B), bool
        Returns:
        """
        assert invalid_tokens.dtype == torch.bool and tokens_to_discard.dtype == torch.bool
        # assert ~(torch.logical_and(invalid_tokens, tokens_to_discard).any()), \
        #     'No overlap should occur for tokens to remove and tokens invalid (padded tokens). '

        N, B = tokens_to_discard.shape
        final_invalid_tokens = torch.logical_or(invalid_tokens, tokens_to_discard)

        src_mask_reclaimed = torch.full((N, B), False, device=invalid_tokens.device)
        src_mask_reclaimed[final_invalid_tokens] = True
        src_mask_reclaimed = src_mask_reclaimed.permute(1, 0)  # N, B ->  B, N

        # We cannot mask all tokens as invalid.
        for k, src_mask in enumerate(src_mask_reclaimed):  # B, N
            assert not src_mask.all()
            # if src_mask.all():
            #     inds = removed_tokens_mask[:, k].nonzero(as_tuple=True)[0]
            #     # set the first item to be False, to set this location to be valid.
            #     src_mask_reclaimed[k, inds[0]] = False
        return src_mask_reclaimed

    def _token_filtering_and_adaption(
            self, x,
            fg_score, small_scale_score,
            mask=None,
            reclaim_padded_region=False,  # token split, remove
            no_bg_token_remove=False,
            no_token_split=False,
            **kwargs,
    ):
        """
            sgdt_targets=None,  # using gt or not, only for debug purpose
            inverse_bg_thd=kwargs.pop('inverse_bg_thd', False),
            debug=kwargs.pop('debug', False)
        Returns:
,
        """

        valid_tokens_float = get_valid_token_mask(x, mask).float()

        bg_sig_thd = kwargs.get('bg_sig_thd', None)
        if bg_sig_thd is None: bg_sig_thd = BG_SIGNIFICANCE_THD

        split_sig_thd = kwargs.get('split_sig_thd', None)
        if split_sig_thd is None: split_sig_thd = FG_SIGNIFICANCE_THD

        tokens_to_discard_original = (fg_score < bg_sig_thd).float() * valid_tokens_float
        tokens_to_split_original = (small_scale_score >= split_sig_thd).float() * valid_tokens_float

        # inverse for debugging only,
        if kwargs.get('inverse_remove_thd', False):
            tokens_to_discard_original = (fg_score > bg_sig_thd).float() * valid_tokens_float

        if kwargs.get('inverse_split_thd', False):
            tokens_to_split_original = (small_scale_score < split_sig_thd).float() * valid_tokens_float

        # filter out false bg predictions (tokens are fg but predicted bg)
        sgdt_targets = kwargs.get('sgdt_targets', None)
        if kwargs.get('filter_false_remove', False):
            assert sgdt_targets is not None
            fg_gt = sgdt_targets['fg_gt'].bool()
            # (not discard) or (fg_gt)
            tokens_to_discard_original[fg_gt] = 0.0  # fg tokens will not be discard.

        if kwargs.get('filter_false_split', False):
            assert sgdt_targets is not None
            scale_gt = torch.where(sgdt_targets['scale_gt'] > 0, 1, 0)  # torch.int64
            # and operation
            tokens_to_split_original *= scale_gt  # only real split tokens will be to split.

        return self._pred_processing(
            x=x, tokens_to_split_original=tokens_to_split_original,
            tokens_to_discard_original=tokens_to_discard_original,
            fg_score=fg_score,
            small_scale_score=small_scale_score,
            mask=mask,
            reclaim_padded_region=reclaim_padded_region,
            no_bg_token_remove=no_bg_token_remove,
            no_token_split=no_token_split,
        )

    def _pred_processing(self,
                         x, tokens_to_discard_original, tokens_to_split_original,
                         fg_score,
                         small_scale_score,
                         mask=None,
                         reclaim_padded_region=False,
                         no_bg_token_remove=False,
                         no_token_split=False,
                         reclaim_token_high_priority=True,
                         **kwargs,
                         ):
        assert ~no_token_split or (no_token_split and ~no_bg_token_remove), \
            'valid setting: 1) with token split 2) no token split but with bg token remove'

        valid_tokens_original_float = get_valid_token_mask(x, mask).float()
        valid_tokens_float = get_valid_token_mask(x, mask).float()
        invalid_tokens = ~get_valid_token_mask(x, mask)  # bool()

        if no_token_split:  # no_token_split, split_conv will never be called.
            # No token to split
            tokens_to_split_original = torch.full_like(invalid_tokens, False)
        elif no_bg_token_remove:  # only reclaim padded region
            # no need to update the score
            tokens_to_discard_original = torch.full_like(invalid_tokens, False)  # .detach().clone()

        # set at least one token to keep if all tokens are to be removed
        all_remove_check = torch.logical_or(invalid_tokens, tokens_to_discard_original.bool()).all(dim=0)
        for k, all_remove in enumerate(all_remove_check):
            if all_remove:
                # set the first tokens to 'not remove'
                tokens_to_discard_original[0, k] *= 0.0  # use multiply to make it differentiable

        assert (tokens_to_discard_original * tokens_to_split_original).sum() == 0, \
            'There should be no overlap for tokens_to_discard_original and tokens_to_split'

        # TODO: do not set the values of invalid tokens to zeros. (child class when reclaim padded regions)
        x_new, tokens_to_discard, tokens_to_split = self._extract_adapted_token(
            x=x, tokens_to_discard_original=tokens_to_discard_original,
            tokens_to_split_original=tokens_to_split_original,
            fg_score=fg_score,
            small_scale_score=small_scale_score,
        )

        src_mask_reclaimed = self._generate_src_key_padding_mask(
            invalid_tokens=invalid_tokens, tokens_to_discard=tokens_to_discard)

        return {
            'x': x_new,
            'src_mask_reclaimed': src_mask_reclaimed,
            'tokens_small_obj': tokens_to_split,
            'tokens_to_discard': tokens_to_discard,
            'valid_tokens': valid_tokens_float,  # TODO: change this to bool()
            # valid_tokens in the original size, this is only used for loss calculation.
            'valid_tokens_original': valid_tokens_original_float,
            'increase_resolution': False,
            # for visualization only
            'tokens_to_discard_original': tokens_to_discard_original,
            'tokens_to_split_original': tokens_to_split_original,
            # 'significance_score': small_scale_score,  # small_scale_score  fg_score
            'fg_score': fg_score,
            'small_scale_score': small_scale_score,
        }  # output_dict

    def _extract_adapted_token(self, x, tokens_to_discard_original,
                               tokens_to_split_original, fg_score,
                               small_scale_score,
                               ):
        """
        the number of tokens to split and remove can be controlled by tokens_to_discard_original and
        tokens_to_split_original, the priority of sampling can be decided by the significance_score or
        small_scale_score.

        Args:
            x:  N, B, C = x.shape
            tokens_to_discard_original: float
            tokens_to_split_original: float
            fg_score: lower score (less important bg tokens), higher priority of being sampled.
            small_scale_score: higher score, higher priority of being sampled.
        Returns:
        """

        # foreground and small objects, (N, B)
        tokens_to_discard = tokens_to_discard_original.clone().detach().bool()
        tokens_small_obj = tokens_to_split_original.clone().detach().bool()

        # Make sure there is no any token to be both removed and split
        assert not torch.any(torch.logical_and(tokens_small_obj, tokens_to_discard))

        x_new = self._reassemble_tokens(
            x=x, tokens_to_discard_original=tokens_to_discard_original,
            tokens_to_split_original=tokens_to_split_original,
            tokens_small_obj=tokens_small_obj, tokens_to_discard=tokens_to_discard)

        return x_new, tokens_to_discard, tokens_small_obj

    def _reassemble_tokens(self, x, tokens_to_discard_original, tokens_to_split_original,
                           tokens_small_obj, tokens_to_discard):
        """ Conduct split conv and update x.
        tokens_to_discard_original and tokens_to_split_original are included here so to make
        the adaption process differentiable (in case their requires_grad = True)
        Args:
            x:
            tokens_to_discard_original: float()
            tokens_to_split_original:  float()
            tokens_small_obj: bool(), (N, B)
            tokens_to_discard: bool()
        Returns:
        """
        N, B, C = x.shape
        x_small = torch.zeros_like(x)

        if self.token_split_conv is not None:  # no split if None
            # different image has different number of tokens to split or merge, thus
            # batch processing is not possible.

            # # do not use the following way, as if the num_split = 0, it will cause backpropogation error
            # # due to the skipped split_conv .
            # batch_ids = (tokens_small_obj.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
            # if there is no split, then self.token_split_conv will not be called.
            # for k in batch_ids:  # #

            for k in range(B):
                img_small_obj_ids = tokens_small_obj[:, k]
                # img_discard_token_ids = tokens_to_discard[:, k]
                x_k = x[:, k, :][img_small_obj_ids]  # M, C, where M is the number of tokens to split
                tokens_small_obj_new = self.token_split_conv(x_k)
                # # x[img_small_obj_ids, k, :] = tokens_small_obj_new[0]
                # # x[img_discard_token_ids, k, :] = tokens_small_obj_new[1]
                x_small[img_small_obj_ids, k, :] += tokens_small_obj_new

        keep_mask = 1 - (
                tokens_to_discard_original * tokens_to_discard + tokens_to_split_original * tokens_small_obj)
        x_new = x.clone() * keep_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]) + x_small  # t

        return x_new

    @staticmethod
    def disable_selected_tokens(tokens_to_split_original, small_scale_score,
                                debug_gt_split_ratio, sampling_method='random_order'
                                ):
        # random_order, priority_order, priority_order_inverse
        assert sampling_method in ['random_order', 'priority_order', 'priority_order_inverse']
        assert 0 <= debug_gt_split_ratio <= 1

        tokens_small_obj = tokens_to_split_original.detach().clone().bool()
        num_valids = torch.sum(tokens_to_split_original, dim=0)

        for k, num in enumerate(num_valids):

            sorted_ids = torch.argsort(small_scale_score[tokens_small_obj[:, k], k], descending=True)  #
            if sampling_method == 'random_order':
                inds = torch.randperm(sorted_ids.shape[0])[:int(num.item())].to(device=sorted_ids.device)
                sorted_ids = sorted_ids[inds]
            elif sampling_method == 'priority_order_inverse':
                sorted_ids = torch.argsort(small_scale_score[tokens_small_obj[:, k], k], descending=False)  #

            keep_num = int(num * debug_gt_split_ratio)
            stop_ids = sorted_ids[keep_num:]
            tokens_to_stop_split = torch.nonzero(tokens_small_obj[:, k]).squeeze(-1)[stop_ids]
            tokens_small_obj[tokens_to_stop_split, k] = 0.0
        return tokens_small_obj.float()

    def discard_split_with_gt_token_score(self, x, mask=None,
                                          sgdt_targets=None,
                                          reclaim_padded_region=False,
                                          no_token_split=False,
                                          no_bg_token_remove=False,
                                          # reclaim_token_high_priority=None,
                                          **kwargs,
                                          ):
        fg_gt, scale_gt = sgdt_targets['fg_gt'], sgdt_targets['scale_gt']

        split_score = sgdt_targets['scale_gt'] + \
                      EPSILON * torch.rand_like(sgdt_targets['scale_gt'])
        fg_score = sgdt_targets['fg_gt'] + \
                   EPSILON * torch.rand_like(sgdt_targets['fg_gt'])

        valid_tokens_float = get_valid_token_mask(x, mask).float()

        tokens_to_split_original = (scale_gt > 0).float() * valid_tokens_float
        tokens_to_discard_original = (1 - fg_gt.float()) * valid_tokens_float

        # # due to interpolation, there may be some tokens both not in fg_gt but in scale_gt
        if (tokens_to_discard_original * tokens_to_split_original).sum() > 0:
            tokens_overlap = (tokens_to_discard_original * tokens_to_split_original).bool()
            # assign the overlapping tokens to split_tokens
            tokens_to_discard_original[tokens_overlap] = 0.0
            print(f'{tokens_overlap.sum()} tokens both in tokens_to_discard_original and tokens_to_split_original.')
        assert (tokens_to_discard_original * tokens_to_split_original).sum() == 0

        output_dict = self._pred_processing(
            x=x, tokens_to_split_original=tokens_to_split_original,
            tokens_to_discard_original=tokens_to_discard_original,
            fg_score=fg_score,
            small_scale_score=split_score,
            mask=mask,
            reclaim_padded_region=reclaim_padded_region,
            no_bg_token_remove=no_bg_token_remove,
            no_token_split=no_token_split,
            **kwargs,
            # reclaim_token_high_priority=reclaim_token_high_priority
        )

        output_dict.update({
            'fg_score': fg_gt,
            'small_scale_score': scale_gt,
            'significance_score': scale_gt,
        })
        return output_dict

    def forward(self, x, mask=None,
                sgdt_targets=None,  # only for debugging
                feat_map_size=None,
                sigma=None,
                gt_ratio=None,
                ):

        if not self.training:
            self.max_split_token_num = self.max_split_token_num_inference

        # return self.discard_split(x, mask)

        reclaim_padded_region = self.token_scoring_config_parser.reclaim_padded_region
        no_bg_token_remove = self.token_scoring_config_parser.no_bg_token_remove
        no_token_split = self.token_scoring_config_parser.no_split

        # return self.discard_split_with_gt_token_score(x, mask, sgdt_targets=sgdt_targets)
        if self.token_scoring_discard_split_criterion.find('gt_only_exp') > -1:
            return self.discard_split_with_gt_token_score(
                x, mask, reclaim_padded_region=reclaim_padded_region,
                sgdt_targets=sgdt_targets,
                no_bg_token_remove=no_bg_token_remove,
                no_token_split=no_token_split,
                gt_ratio=gt_ratio,
            )
        else:
            raise NotImplementedError
