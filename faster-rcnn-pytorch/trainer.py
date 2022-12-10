from collections import namedtuple

import torch as torch
from torch import nn
from torch.nn import functional as F
import math 


from utils.utils import AnchorTargetCreator, ProposalTargetCreator
cifar10_mean = (0.0, 0.0, 0.0)
cifar10_std = (1.0, 1.0, 1.0)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)
epsilon = (4 / 255.) / std
alpha = (2 / 255.) / std
steps = 3
from attacker import *
attacker = One_Layer_Attacker_01(eps=(2 / 225.) / std, input_channel=6).cuda()
optimizer_att = torch.optim.SGD(attacker.parameters(), lr=0.01, momentum=0.9,
                              weight_decay=5e-4)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)
LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])

class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn,optimizer):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = 1
        self.roi_sigma = 1

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = [0, 0, 0, 0]
        self.loc_normalize_std = [0.1, 0.1, 0.2, 0.2]

        self.optimizer = optimizer

    def forward(self, imgs, bboxes, labels, scale):
        n = imgs.shape[0]
        img_size = imgs.shape[2:]
        
        # 获取公用特征层
        base_feature = self.faster_rcnn.extractor(imgs)

        # 利用rpn网络获得先验框的得分与调整参数
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(base_feature, img_size, scale)

        rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all = 0, 0, 0, 0
        for i in range(n):
            bbox = bboxes[i]
            label = labels[i]
            rpn_loc = rpn_locs[i]
            rpn_score = rpn_scores[i]
            roi = rois[roi_indices==i]
            feature = base_feature[i]

            # -------------------------------------------------- #
            #   利用真实框和先验框获得建议框网络应该有的预测结果
            #   给每个先验框都打上标签
            #   gt_rpn_loc      [num_anchors, 4]
            #   gt_rpn_label    [num_anchors, ]
            # -------------------------------------------------- #
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchor, img_size)
            gt_rpn_loc = torch.Tensor(gt_rpn_loc)
            gt_rpn_label = torch.Tensor(gt_rpn_label).long()

            if rpn_loc.is_cuda:
                gt_rpn_loc = gt_rpn_loc.cuda()
                gt_rpn_label = gt_rpn_label.cuda()

            # -------------------------------------------------- #
            #   分别计算建议框网络的回归损失和分类损失
            # -------------------------------------------------- #
            rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)
  
            # ------------------------------------------------------ #
            #   利用真实框和建议框获得classifier网络应该有的预测结果
            #   获得三个变量，分别是sample_roi, gt_roi_loc, gt_roi_label
            #   sample_roi      [n_sample, ]
            #   gt_roi_loc      [n_sample, 4]
            #   gt_roi_label    [n_sample, ]
            # ------------------------------------------------------ #
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label, self.loc_normalize_mean, self.loc_normalize_std)
            sample_roi = torch.Tensor(sample_roi)
            gt_roi_loc = torch.Tensor(gt_roi_loc)
            gt_roi_label = torch.Tensor(gt_roi_label).long()
            sample_roi_index = torch.zeros(len(sample_roi))
            
            if feature.is_cuda:
                sample_roi = sample_roi.cuda()
                sample_roi_index = sample_roi_index.cuda()
                gt_roi_loc = gt_roi_loc.cuda()
                gt_roi_label = gt_roi_label.cuda()

            roi_cls_loc, roi_score = self.faster_rcnn.head(torch.unsqueeze(feature, 0), sample_roi, sample_roi_index, img_size)

            # ------------------------------------------------------ #
            #   根据建议框的种类，取出对应的回归预测结果
            # ------------------------------------------------------ #
            n_sample = roi_cls_loc.size()[1]
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample), gt_roi_label]

            # -------------------------------------------------- #
            #   分别计算Classifier网络的回归损失和分类损失
            # -------------------------------------------------- #
            roi_loc_loss = _fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data, self.roi_sigma)
            roi_cls_loss = nn.CrossEntropyLoss()(roi_score[0], gt_roi_label)

            rpn_loc_loss_all += rpn_loc_loss
            rpn_cls_loss_all += rpn_cls_loss
            roi_loc_loss_all += roi_loc_loss
            roi_cls_loss_all += roi_cls_loss
            
        losses = [rpn_loc_loss_all/n, rpn_cls_loss_all/n, roi_loc_loss_all/n, roi_cls_loss_all/n]
        losses = losses + [sum(losses)]
        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        return losses
    def train_FGSM_step(self, imgs, bboxes, labels, scale,return_img = False,targeted=False):
        delta = torch.zeros_like(imgs).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - imgs, upper_limit - imgs)
        delta.requires_grad = True
        losses = self.forward(imgs+delta, bboxes, labels, scale)
        losses.total_loss.backward()
        grad = delta.grad.detach()
        d = delta
        g = grad
        if targeted:
            d = clamp(d - epsilon * torch.sign(g), -epsilon, epsilon)
        else:
            d = clamp(d + epsilon * torch.sign(g), -epsilon, epsilon)
        d = clamp(d, lower_limit - imgs, upper_limit - imgs)
        delta.data = d
        delta.grad.zero_()
        if return_img:
            return imgs + delta
        self.optimizer.zero_grad()
        losses = self.forward(imgs+delta, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        return losses
    def train_PGD_step(self, imgs, bboxes, labels, scale,return_img=False,targeted=False):
        delta = torch.zeros_like(imgs).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - imgs, upper_limit - imgs)
        delta.requires_grad = True
        for _ in range(steps):
            losses = self.forward(imgs + delta, bboxes, labels, scale)
            losses.total_loss.backward()
            grad = delta.grad.detach()
            d = delta
            g = grad
            if targeted:
                d = clamp(d - alpha * torch.sign(g), -epsilon, epsilon)
            else:
                d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - imgs, upper_limit - imgs)
            delta.data = d
            delta.grad.zero_()
        if return_img:
            return imgs + delta
        self.optimizer.zero_grad()
        losses = self.forward(imgs+delta, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        return losses
    
    
    # only support batch size == 1
    def train_SquareAttack_step(self, imgs, bboxes, labels, scale,return_img = False,targeted=False):
        def random_choice(shape):
            t = 2 * torch.rand(shape) - 1
            return torch.sign(t).cuda()
        def p_selection( it,p_init=0.05):
            """ schedule to decrease the parameter p """
            if 10 < it <= 50:
                p = p_init / 2
            elif 50 < it <= 200:
                p = p_init / 4
            elif 200 < it <= 500:
                p = p_init / 8
            elif 500 < it <= 1000:
                p = p_init / 16
            elif 1000 < it <= 2000:
                p = p_init / 32
            elif 2000 < it <= 4000:
                p = p_init / 64
            elif 4000 < it <= 6000:
                p = p_init / 128
            elif 6000 < it <= 8000:
                p = p_init / 256
            elif 8000 < it:
                p = p_init / 512
            else:
                p = p_init
            return p
        def random_int(low=0, high=1, shape=[1]):
            t = low + (high - low) * torch.rand(shape)
            return t.long().cuda()
        with torch.no_grad():
            n_query = 500
            x = imgs.clone().detach()
            c, h, w = x.shape[1:]
            n_features = c * h * w
            x_best = x.clone().detach() 
            for i in range(len(epsilon)):    
                x_best[:, i, :, :] = x_best[:, i, :, :] + epsilon[i][0][0].item() * random_choice([x_best.shape[0], 1, 1, w])
            x_best.data = clamp(x_best, lower_limit , upper_limit )
            loss_min = self.forward(x_best,bboxes,labels,scale)[-1]
            for  iter in range(n_query):
                x_cur = x_best.clone().detach()
                loss_min_curr = loss_min
                p = p_selection(iter)
                s = max(int(round(math.sqrt(p * n_features / c))), 1)
                vh = random_int(0, h - s)
                vw = random_int(0, w - s)
                new_deltas = torch.zeros([x_cur.shape[0],c, h, w]).cuda()
                for i in range(len(epsilon)):   
                    new_deltas[:,i, vh:vh + s, vw:vw + s] = 2. * epsilon[i][0][0].item() * random_choice([1, 1, 1])
                    x_cur[:,i,:,:] += new_deltas[:,i,:,:]
                    x_cur[:,i,:,:] = clamp(x_cur[:,i,:,:],x[:,i,:,:]-epsilon[i][0][0].item(),x[:,i,:,:]+epsilon[i][0][0].item())
                x_cur.data = clamp(x_cur, lower_limit , upper_limit )
                loss_cur = self.forward(x_cur,bboxes,labels,scale)[-1]
                idx_improved = (loss_cur > loss_min_curr).float()
                loss_min = idx_improved * loss_cur + (
                    1. - idx_improved) * loss_min_curr
                idx_improved = idx_improved.reshape([-1,
                    *[1]*len(x_cur.shape[:-1])])
                x_best = idx_improved * x_cur + (
                    1. - idx_improved) * x_best
                if iter % (n_query//5) == 0:
                    print('cur_loss: ',loss_min)
        if return_img:
            return x_best
        
    def train_FGSM_SDI_step(self, imgs, bboxes, labels, scale):
        delta = torch.zeros_like(imgs).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - imgs, upper_limit - imgs)
        delta.requires_grad = True

        losses = self.forward(imgs + delta, bboxes, labels, scale)
        losses.total_loss.backward(retain_graph=True)
        grad = delta.grad.detach()
        advinput = torch.cat([imgs, 1.0 * (torch.sign(grad))], 1).detach()
        perturbation = attacker(advinput)

        x_adv = imgs + perturbation
        x_adv.requires_grad_()
        #########################################
        with torch.enable_grad():
            losses = self.forward(x_adv, bboxes, labels, scale)
            losses.total_loss.backward(retain_graph=True)
            grad_adv = torch.autograd.grad(losses, [x_adv])[0]
            perturbation_1 = clamp(epsilon * torch.sign(grad_adv), -epsilon, epsilon)
        perturbation_total = perturbation + perturbation_1
        perturbation_total = clamp(perturbation_total, -epsilon, epsilon)

        optimizer_att.zero_grad()
        losses = self.forward(imgs+perturbation_total, bboxes, labels, scale)
        total_loss=-losses.total_loss
        total_loss.backward(retain_graph=True)
        optimizer_att.step()
        #################################
        perturbation = attacker(advinput)
        x_adv = imgs + perturbation
        x_adv.requires_grad_()
        #########################################
        with torch.enable_grad():
            losses = self.forward(x_adv, bboxes, labels, scale)
            losses.total_loss.backward(retain_graph=True)
            grad_adv = torch.autograd.grad(losses, [x_adv])[0]
            perturbation_1 = clamp(epsilon * torch.sign(grad_adv), -epsilon, epsilon)
        perturbation_total = perturbation + perturbation_1
        perturbation_total = clamp(perturbation_total, -epsilon, epsilon)
        self.optimizer.zero_grad()
        losses = self.forward(imgs+perturbation_total, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        return losses

    def train_FGSM_SDI_01_step(self, imgs, bboxes, labels, scale):
        delta = torch.zeros_like(imgs).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - imgs, upper_limit - imgs)
        delta.requires_grad = True

        losses = self.forward(imgs + delta, bboxes, labels, scale)
        losses.total_loss.backward(retain_graph=True)
        grad = delta.grad.detach()
        advinput = torch.cat([imgs, 1.0 * (torch.sign(grad))], 1).detach()
        perturbation = attacker(advinput)

        x_adv = imgs + perturbation
        x_adv.requires_grad_()
        #########################################
        with torch.enable_grad():
            losses = self.forward(x_adv, bboxes, labels, scale)
            losses.total_loss.backward(retain_graph=True)
            grad_adv = torch.autograd.grad(losses, [x_adv])[0]
            perturbation_1 = clamp(epsilon * torch.sign(grad_adv), -epsilon, epsilon)
        perturbation_total = perturbation + perturbation_1
        perturbation_total = clamp(perturbation_total, -epsilon, epsilon)
        perturbation_total_01=perturbation_total.clone().detach()
        optimizer_att.zero_grad()
        losses = self.forward(imgs+perturbation_total, bboxes, labels, scale)
        total_loss=-losses.total_loss
        total_loss.backward(retain_graph=True)
        optimizer_att.step()
        #################################

        self.optimizer.zero_grad()
        losses = self.forward(imgs+perturbation_total_01, bboxes, labels, scale)

        losses.total_loss.backward()
        self.optimizer.step()
        return losses
def _smooth_l1_loss(x, t, sigma):
    sigma_squared = sigma ** 2
    regression_diff = (x - t)
    regression_diff = regression_diff.abs()
    regression_loss = torch.where(
            regression_diff < (1. / sigma_squared),
            0.5 * sigma_squared * regression_diff ** 2,
            regression_diff - 0.5 / sigma_squared
        )
    return regression_loss.sum()

def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    pred_loc = pred_loc[gt_label>0]
    gt_loc = gt_loc[gt_label>0]

    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, sigma)
    num_pos = (gt_label > 0).sum().float()
    loc_loss /= torch.max(num_pos, torch.ones_like(num_pos))
    return loc_loss
