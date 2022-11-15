import torch
import torch.nn as nn

from .attack import Attack
from .util import mmdet_clamp


class DeepFool(Attack):
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (Default: 50)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, args):
        super().__init__("DeepFool", model)
        self.steps = args.steps * 10
        self.overshoot = args.overshoot
        self.supported_mode = ['default']

    def forward(self, data):
        r"""
        Overridden.
        """
        images = data['img'][0].data[0].clone().detach().to(self.device)
        ub,lb = torch.max(images.view(3,-1),dim=1).values,torch.min(images.view(3,-1),dim=1).values
        adv_images = images.clone().detach()
        new_data = {}
        new_data['img_metas'] = data['img_metas'][0].data[0]


        for i in range(self.steps):
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
            delta = grad / (torch.norm(grad, p=2)**2)
            adv_images = adv_images.detach() + (1+self.overshoot)*delta
        adv_images = mmdet_clamp(adv_images,lb,ub)
        data['img'][0].data[0] = adv_images
        return adv_images
