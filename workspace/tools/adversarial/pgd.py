import torch
import torch.nn as nn

from .attack import Attack
from collections import Iterable
from .util import mmdet_clamp

class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, args):
        super().__init__("PGD", model)
        self.eps = args.eps
        self.alpha = args.alpha
        self.steps = args.steps
        self.random_start = args.random_start
        self.supported_mode = ['default', 'targeted']

    def forward(self, data):
        r"""
        Overridden.
        """
        images = data['img'][0].data[0].clone().detach().to(self.device)
        ub,lb = torch.max(images.view(3,-1),dim=1).values,torch.min(images.view(3,-1),dim=1).values
        eps = self.eps * torch.max(ub - lb )
        alpha = self.alpha * torch.max(ub - lb)

        adv_images = images.clone().detach()
        new_data = {}
        new_data['img_metas'] = data['img_metas'][0].data[0]

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = mmdet_clamp(adv_images,lb,ub)
            # for chn in range(adv_images.shape[1]):
            #     adv_images[:,chn:chn+1,:,:] = torch.clamp(adv_images[:,chn:chn+1,:,:], min=lb[chn], max=ub[chn]).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            new_data['img'] = adv_images

            if 'gt_masks' in data.keys():
                losses = self.model(**new_data, return_loss=True,gt_bboxes=data['gt_bboxes'][0].data[0],
                                gt_labels=data['gt_labels'][0].data[0], gt_masks=  data['gt_masks'][0].data[0])
                loss_cls = sum(losses[_loss].mean() for _loss in losses.keys() if 'cls' in _loss and isinstance(losses[_loss],torch.Tensor))
            else:
                losses = self.model(**new_data, return_loss=True,gt_bboxes=data['gt_bboxes'][0].data[0],
                                gt_labels=data['gt_labels'][0].data[0])
                loss_cls ,_ = self.model.module._parse_losses(losses)
            self.model.zero_grad()
            loss_cls.backward()
            grad = adv_images.grad.data

            adv_images = adv_images.detach() + alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)

            # for chn in range(adv_images.shape[1]):
            #     adv_images[:,chn:chn+1,:,:] = torch.clamp(images[:,chn:chn+1,:,:] + delta[:,chn:chn+1,:,:], min=lb[chn], max=ub[chn]).detach()
            adv_images = mmdet_clamp(images+delta,lb,ub)
            data['img'][0].data[0] = adv_images
        return adv_images
