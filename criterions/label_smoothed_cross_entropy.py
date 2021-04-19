# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion
import torch


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # , g_mask_loss, d_mask_loss, encoder_nll_loss
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']


        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            #'g_mask_loss': utils.item(g_mask_loss.data) if reduce else g_mask_loss.data,
            #'encoder_nll_loss': utils.item(encoder_nll_loss.data) if reduce else encoder_nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs1 = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)
        target1 = target.view(-1, 1)
        non_pad_mask = target1.ne(self.padding_idx)
        
        nll_loss = -lprobs1.gather(dim=-1, index=target1)[non_pad_mask]
        smooth_loss = -lprobs1.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs1.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss 
        return loss, nll_loss ###, encoder_nll_loss ,g_mask_loss
        
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        g_mask_loss = float(sum(log.get('g_mask_loss', 0) for log in logging_outputs) / sample_size)
        #d_mask_loss = float(sum(log.get('d_mask_loss', 0) for log in logging_outputs) / sample_size)
        encoder_nll_loss = float(sum(log.get('encoder_nll_loss', 0) for log in logging_outputs) / sample_size) / math.log(2)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'g_mask_loss': g_mask_loss,
            #'d_mask_loss': d_mask_loss,
            'encoder_nll_loss': encoder_nll_loss,
        }
'''         
        encoder_x = model.encoder.src_tokens
        encoder_x = encoder_x.view(-1, 1) # [batch*src_len, ]                
        encoder_vocab_output = model.encoder.encoder_vocab_output
        encoder_vocab_output = encoder_vocab_output.transpose(0,1)
        lprobs_encoder = torch.nn.functional.log_softmax(encoder_vocab_output, dim=-1)
        lprobs_encoder = model.get_normalized_probs((encoder_vocab_output, None), log_probs=True)
        lprobs_encoder = lprobs_encoder.view(-1, lprobs_encoder.size(-1))

        encoder_non_pad_mask = encoder_x.ne(self.padding_idx)
        
        encoder_nll_loss = -lprobs_encoder.gather(dim=-1, index=encoder_x)[encoder_non_pad_mask] #  [batch*src_len, 
        #encoder_nll_loss = encoder_nll_loss *mask.view(-1).unsqueeze(-1)[encoder_non_pad_mask]
 
        

        ####        
        cross_entropy_loss = loss
        
        # encoder_nll_loss = encoder_nll_loss.sum()
        encoder_nll_loss =  encoder_nll_loss
        # non_pad_mask_batch : [batch, length, 1]
        #non_pad_mask_batch = encoder_x.unsqueeze(-1).ne(self.padding_idx)

        # lprobs: [batch, length, class_num]
        # target: [batch, length, 1]
        # nll_loss_batch : [batch, length, 1]
        # nll_loss_batch = -lprobs_encoder.gather(dim=-1, index=encoder_x)
        # nll_loss_batch = nll_loss_batch * non_pad_mask_batch
        # nll_loss_batch = nll_loss_batch.sum(dim=(1,2)).unsqueeze(-1)
        
        mask = model.encoder.p_mask.bool() # mask is 0, no mask is 1
        unmask = ~mask

        p_mask = model.encoder.mask_output

        #model.encoder.backwards += 1

        g_mask_loss = -encoder_nll_loss.detach()*((model.encoder.mask_output * unmask).transpose(0,1).reshape(-1,1)[encoder_non_pad_mask]+ 1e-6).log()
        # d_mask_loss = -nll_loss_batch *(1 - model.encoder.mask_output.detach()+ 1e-6).log()

        # if torch.isnan((model.encoder.unmask_output*mask + model.encoder.mask_output*unmask + 1e-6).log().mean()):
        #     print("!", model.encoder.unmask_output*mask + model.encoder.mask_output*unmask + 1e-6)

        # if torch.isnan((model.encoder.unmask_output.detach()*unmask + model.encoder.mask_output.detach()*mask + 1e-6).log().mean()):
        #     print("!!", model.encoder.unmask_output.detach()*unmask + model.encoder.mask_output.detach()*mask + 1e-6)


        g_mask_loss = g_mask_loss.sum()
        
        encoder_nll_loss = encoder_nll_loss.sum()
        # d_mask_loss = d_mask_loss.sum()
        
        
        loss =  cross_entropy_loss  + encoder_nll_loss  + g_mask_loss
'''
        ### return loss, nll_loss ###, encoder_nll_loss ,g_mask_loss
    
    
