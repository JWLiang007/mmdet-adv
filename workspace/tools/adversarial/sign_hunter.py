import torch
import torch.nn as nn

from .attack import Attack
from collections import Iterable
import numpy as np 

class SIGN_HUNTER(Attack):

    def __init__(self, model, args):
        super().__init__("SIGN_HUNTER", model)
        self.eps = args.eps
        self.steps = args.steps
        self.supported_mode = ['default', 'targeted']

    def forward(self, data):
        r"""
        Overridden.
        """
        images = data['img'][0].data[0].clone().detach().to(self.device)
        ub,lb = torch.max(images.view(3,-1),dim=1).values,torch.min(images.view(3,-1),dim=1).values
        eps = self.eps * torch.max(ub - lb )

        adv_images = images.clone().detach()
        adv_images.requires_grad = False
        new_data = {}
        new_data['img_metas'] = data['img_metas'][0].data[0]
        test_data = {}
        
        test_data['img_metas'] = data['img_metas'][0].data[0]
        
        # param 
        qps = 100
        steps = self.steps *  qps 

        new_xs=adv_images.clone()
        self.is_new_batch = True
        for i in range(steps):
        # i=0
        # while True:
            if i % qps == 0:
                test_data['img'] = new_xs
                print('loss_',i," : ", self.model.module._parse_losses(self.model(**test_data, return_loss=True,gt_bboxes=data['gt_bboxes'][0].data[0],gt_labels=data['gt_labels'][0].data[0])))

 
            _shape = list(new_xs.shape)
            dim = np.prod(_shape[1:])
            # additional queries at the start
            add_queries = 0
            if self.is_new_batch:
                self.xo_t = new_xs.clone()
                self.h = 0
                self.i = 0
            if self.i == 0 and self.h == 0:
                self.sgn_t = torch.sign(torch.ones(_shape[0], dim))
                fxs_t = self.xo_t+ eps * torch.sign(self.sgn_t.view(_shape)).cuda()
                # fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
                bxs_t = self.xo_t  
                est_deriv = (self.loss_fct(fxs_t,data) - self.loss_fct(bxs_t,data)) / eps
                self.best_est_deriv = est_deriv
            chunk_len = np.ceil(dim / (2 ** self.h)).astype(int)
            istart = self.i * chunk_len
            iend = min(dim, (self.i + 1) * chunk_len)
            self.sgn_t[:, istart:iend] *= - 1.
            fxs_t = self.xo_t + eps * torch.sign(self.sgn_t.view(_shape)).cuda()
            # fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
            bxs_t = self.xo_t
            est_deriv = (self.loss_fct(fxs_t,data) - self.loss_fct(bxs_t,data)) / eps
            
            ### sign here
            s_set = []
            if est_deriv < self.best_est_deriv:
                s_set.append(0)
            self.sgn_t[s_set, istart: iend] *= -1.
            # self.sgn_t[[i for i, val in enumerate(est_deriv < self.best_est_deriv) if val], istart: iend] *= -1.
            

            self.best_est_deriv = (est_deriv >= self.best_est_deriv) * est_deriv + (
                    est_deriv < self.best_est_deriv) * self.best_est_deriv
            # perform the step
            new_xs = self.xo_t + eps * torch.sign(self.sgn_t.view(_shape)).cuda()
            # new_xs = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
            # update i and h for next iteration
            self.i += 1
            if self.i == 2 ** self.h or iend == dim:
                self.h += 1
                self.i = 0
                # if h is exhausted, set xo_t to be xs_t
                if self.h == np.ceil(np.log2(dim)).astype(int) + 1:
                    self.xo_t = adv_images.clone()
                    self.h = 0
                    print("new change")
                    break
                    
                
            
            self.is_new_batch = False
            # i+=1
            
        adv_images = new_xs.detach()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)

        for chn in range(adv_images.shape[1]):
            adv_images[:,chn:chn+1,:,:] = torch.clamp(images[:,chn:chn+1,:,:] + delta[:,chn:chn+1,:,:], min=lb[chn], max=ub[chn]).detach()
        data['img'][0].data[0] = adv_images
        return adv_images

    def loss_fct(self,img,data):
        with torch.no_grad():
            new_data  = {}
            new_data['img'] = img
            new_data['img_metas'] = data['img_metas'][0].data[0]
            losses = self.model(**new_data, return_loss=True,gt_bboxes=data['gt_bboxes'][0].data[0],
                            gt_labels=data['gt_labels'][0].data[0])
            loss_cls ,_ = self.model.module._parse_losses(losses)
            return loss_cls