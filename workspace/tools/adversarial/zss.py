import torch
import torch.nn as nn

from .attack import Attack
from collections import Iterable


class ZSS(Attack):
    r"""
    BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

    .. note:: If steps set to 0, steps will be automatically decided following the paper.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=10)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, args):
        super().__init__("BIM", model)
        self.eps = args.eps
        self.steps = args.steps
        self.alpha = args.alpha
        self.supported_mode = ['default', 'targeted']

    def forward(self, data):
        r"""
        Overridden.
        """
        images = data['img'][0].data[0].clone().detach().to(self.device)
        ub,lb = torch.max(images.view(3,-1),dim=1).values,torch.min(images.view(3,-1),dim=1).values
        eps = self.eps * torch.max(ub - lb )
        alpha = self.alpha * torch.max(ub - lb)
        # labels = labels.clone().detach().to(self.device)

        # if self._targeted:
        #     target_labels = self._get_target_label(images, labels)

        # loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        adv_images.requires_grad = False
        new_data = {}
        new_data['img_metas'] = data['img_metas'][0].data[0]
        
        # test_data = {}
        # test_data['img_metas'] = data['img_metas'][0].data[0]
        qps = 100
        self.steps = self.steps * 10 
        alpha = alpha/ 10 
        for i in range(self.steps):
            # test_data['img'] = adv_images
            # print('loss_',i," : ", self.model.module._parse_losses(self.model(**test_data, return_loss=True,gt_bboxes=data['gt_bboxes'][0].data[0],gt_labels=data['gt_labels'][0].data[0])))
            grad = 0
            # adv_images.requires_grad = True
            for j in range(qps):
                noise = torch.randn_like(adv_images) 
                noise = noise * alpha / (torch.max(noise)*(qps/2))
                f_img = adv_images + noise
                b_img = adv_images
                with torch.no_grad():
                    new_data['img'] = f_img
                    # if 'gt_masks' in data.keys():
                    #     losses = self.model(**new_data, return_loss=True,gt_bboxes=data['gt_bboxes'][0].data[0],
                    #                     gt_labels=data['gt_labels'][0].data[0], gt_masks=  data['gt_masks'][0].data[0])
                    #     loss_cls = sum(losses[_loss].mean() for _loss in losses.keys() if 'cls' in _loss and isinstance(losses[_loss],torch.Tensor))
                    # else:
                    losses = self.model(**new_data, return_loss=True,gt_bboxes=data['gt_bboxes'][0].data[0],
                                    gt_labels=data['gt_labels'][0].data[0])
                    loss_cls_f ,_ = self.model.module._parse_losses(losses)
  
                    new_data['img'] = b_img
                    losses = self.model(**new_data, return_loss=True,gt_bboxes=data['gt_bboxes'][0].data[0],
                                    gt_labels=data['gt_labels'][0].data[0])
                    loss_cls_b ,_ = self.model.module._parse_losses(losses)
                    grad += (loss_cls_f - loss_cls_b) * noise
   
            # grad = adv_images.grad.data

            adv_images = adv_images.detach() + alpha * grad.detach().sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)

            for chn in range(adv_images.shape[1]):
                adv_images[:,chn:chn+1,:,:] = torch.clamp(images[:,chn:chn+1,:,:] + delta[:,chn:chn+1,:,:], min=lb[chn], max=ub[chn]).detach()

        data['img'][0].data[0] = adv_images
        return adv_images
