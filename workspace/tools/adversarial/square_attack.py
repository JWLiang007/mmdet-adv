import torch
import torch.nn as nn

from .attack import Attack
from collections import Iterable
import numpy as np

class SquareAttack(Attack):
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
        self.p_init = 0.05

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
        
        # test_data = {}
        # test_data['img_metas'] = data['img_metas'][0].data[0]
        qps = 100
        self.steps = self.steps * 10 
        alpha = alpha/ 10 
        
        self.is_new_batch = True
        xs = adv_images.clone()
        c, h, w = adv_images.shape[1:]
        n_features = c*h*w
        n_queries = torch.zeros(adv_images.shape[0])
        for i in range(self.steps):
            # test_data['img'] = adv_images
            # print('loss_',i," : ", self.model.module._parse_losses(self.model(**test_data, return_loss=True,gt_bboxes=data['gt_bboxes'][0].data[0],gt_labels=data['gt_labels'][0].data[0])))
            grad = 0

            if self.is_new_batch:
                self.x = xs.clone()
                init_delta = torch.Tensor(np.random.choice([-eps.clone().cpu(), eps.clone().cpu()], size=[xs.clone().cpu().shape[0], c, 1, w])).cuda()
                # xs = torch.clamp(xs + init_delta, self.lb, self.ub)
                for chn in range(xs.shape[1]):
                    xs[:,chn:chn+1,:,:] = torch.clamp(xs[:,chn:chn+1,:,:] + init_delta[:,chn:chn+1,:,:], min=lb[chn], max=ub[chn]).detach()
                self.best_loss = self.loss_fct(xs,data)
                n_queries += torch.ones(xs.shape[0])
                self.i = 0

            deltas = xs - self.x
            p = self.p_selection(self.p_init, self.i, 10000)
            for i_img in range(xs.shape[0]):
                s = int(round(np.sqrt(p * n_features / c)))
                s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
                center_h = np.random.randint(0, h - s)
                center_w = np.random.randint(0, w - s)

                x_window = self.x[i_img, :, center_h:center_h+s, center_w:center_w+s]
                x_best_window = xs[i_img, :, center_h:center_h+s, center_w:center_w+s]
                # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
                _win = x_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s]
                for chn in range(xs.shape[1]):
                    _win[chn:chn+1,:,:] = torch.clamp(_win[chn:chn+1,:,:], min=lb[chn], max=ub[chn]).detach()
                while torch.sum(torch.abs(_win - x_best_window) < 10**-7) == c*s*s:
                    deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = torch.Tensor(np.random.choice([-eps.clone().cpu(), eps.clone().cpu()], size=[c, 1, 1])).cuda()
                    _win = x_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s]
                    for chn in range(xs.shape[1]):
                        _win[chn:chn+1,:,:] = torch.clamp(_win[chn:chn+1,:,:], min=lb[chn], max=ub[chn]).detach()
                    
            # x_new = torch.clamp(self.x + deltas, self.lb, self.ub).permute(0,2,3,1)
            x_new = self.x.clone()
            for chn in range(xs.shape[1]):
                x_new[:,chn:chn+1,:,:] = torch.clamp(self.x[:,chn:chn+1,:,:] + deltas[:,chn:chn+1,:,:], min=lb[chn], max=ub[chn]).detach()
            new_loss = self.loss_fct(x_new,data)
            n_queries += torch.ones(xs.shape[0])
            idx_improved = new_loss > self.best_loss
            self.best_loss = idx_improved * new_loss + ~idx_improved * self.best_loss
            # xs = xs.permute(0,2,3,1)
            idx_improved = torch.reshape(idx_improved, [-1, *[1]*len(x_new.shape[:-1])])
            x_new = idx_improved * x_new + ~idx_improved * xs
            self.i += 1
            
            self.is_new_batch = False
            
            # grad = adv_images.grad.data

        adv_images = x_new
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

    def p_selection(self, p_init, it, n_iters):
        """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
        it = int(it / n_iters * 10000)

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
        elif 8000 < it <= 10000:
            p = p_init / 512
        else:
            p = p_init

        return p

    def pseudo_gaussian_pert_rectangles(self, x, y):
        delta = torch.zeros([x, y])
        x_c, y_c = x // 2 + 1, y // 2 + 1

        counter2 = [x_c - 1, y_c - 1]
        for counter in range(0, max(x_c, y_c)):
            delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
                max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

            counter2[0] -= 1
            counter2[1] -= 1

        delta /= torch.sqrt(torch.sum(delta ** 2, dim=1, keepdim=True))
        return delta


    def meta_pseudo_gaussian_pert(self, s):
        delta = torch.zeros([s, s])
        n_subsquares = 2
        if n_subsquares == 2:
            delta[:s // 2] = self.pseudo_gaussian_pert_rectangles(s // 2, s)
            delta[s // 2:] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
            delta /= torch.sqrt(torch.sum(delta ** 2, dim=1, keepdim=True))
            if np.random.rand(1) > 0.5: delta = torch.transpose(delta, 0, 1)

        elif n_subsquares == 4:
            delta[:s // 2, :s // 2] = self.pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
            delta[s // 2:, :s // 2] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
            delta[:s // 2, s // 2:] = self.pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
            delta[s // 2:, s // 2:] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
            delta /= torch.sqrt(torch.sum(delta ** 2, dim=1, keepdim=True))

        return delta