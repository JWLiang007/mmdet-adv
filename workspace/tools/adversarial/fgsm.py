import torch
import torch.nn as nn

from .attack import Attack
from collections import Iterable


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.007)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=0.007)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model,args):
        super().__init__("FGSM", model)
        self.eps = args.eps
        self._supported_mode = ['default', 'targeted']

    def forward(self, data):
        r"""
        Overridden.
        """
        images = data['img'][0].data[0].clone().detach().to(self.device)
        ub,lb = torch.max(images.view(3,-1),dim=1).values,torch.min(images.view(3,-1),dim=1).values
        eps = self.eps * torch.max(ub - lb )
        # labels = labels.clone().detach().to(self.device)

        # if self._targeted:
        #     target_labels = self._get_target_label(images, labels)

        # loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        new_data = {}
        new_data['img_metas'] = data['img_metas'][0].data[0]
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
            # loss_cls = torch.zeros((1)).cuda()
            # for k in losses.keys():
            #     if 'loss_cls' in k:
            #         _loss_cls = sum(_loss.mean() for _loss in losses[k])  if isinstance(losses[k],list) else losses[k]
            #         loss_cls+=_loss_cls
            # # if 'loss_cls' in losses.keys():
            # #     loss_cls = sum(_loss.mean() for _loss in losses['loss_cls'])  if isinstance(losses['loss_cls'],list) else losses['loss_cls']
            #     elif 'det_loss' in k:
            #         loss_cls = sum(_loss.mean() for _loss in losses[k])  if isinstance(losses[k],list) else losses[k]


        self.model.zero_grad()
        loss_cls= loss_cls
        loss_cls.backward()
        grad = adv_images.grad.data


        adv_images = images + eps*grad.sign()
        for chn in range(adv_images.shape[1]):
            adv_images[:,chn:chn+1,:,:] = torch.clamp(adv_images[:,chn:chn+1,:,:] , min=lb[chn], max=ub[chn]).detach()
        data['img'][0].data[0] = adv_images
        
        return adv_images
