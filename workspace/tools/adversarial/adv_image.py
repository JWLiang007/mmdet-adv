import os.path as osp
import pickle
import shutil
import tempfile
import time
import os

import cv2
import mmcv
import torch
import torch.distributed as dist
from .util import get_gt_bboxes_scores_and_labels
from .difgsm import DIFGSM
from .tifgsm import  TIFGSM
from .mifgsm import  MIFGSM
from .vmifgsm import VMIFGSM
from .fgsm import FGSM
from .bim import BIM
from .nes import NES
from .zss import ZSS
from .sign_hunter import SIGN_HUNTER
from .square_attack import SquareAttack
from .pgd import PGD
from .nifgsm import NIFGSM
from .deepfool import DeepFool
import numpy as np
from mmcv.parallel.data_container import DataContainer
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
ta_factory = {
    'difgsm': DIFGSM,
    'tifgsm': TIFGSM,
    'mifgsm': MIFGSM,
    'vmifgsm': VMIFGSM,
    'fgsm':FGSM,
    'bim':BIM,
    'nes':NES,
    'zss':ZSS,
    'sign_hunter':SIGN_HUNTER,
    'square_attack':SquareAttack,
    'pgd':PGD,
    'nifgsm':NIFGSM,
    'deepfool':DeepFool
}


def single_gpu_adv(model,
                    data,
                   args):
    model.eval()


    attack = ta_factory[args.method](model, args)
    

    # test_res = dataset._det2gt(test_res)
    # test_res=[i for i in test_res if i['score'] > 0.3]
    new_data = data
    if not args.with_gt:
        new_data = det2gt(data,model,args.score_thr)
    adv = attack(new_data)

    batch_size = adv.shape[0]
    if args.show_dir:
        img_tensor = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
        imgs = tensor2imgs(img_tensor.detach().clone(), **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
            h, w, _ = img_meta['img_shape']
            if 'border' in img_meta.keys():
                h_s , h_t, w_s, w_t = img_meta['border'].astype(np.uint32)
                img_show = img[h_s:h_t,w_s:w_t,:]
            else:
                img_show = img[:h, :w, :]

            ori_h, ori_w = img_meta['ori_shape'][:-1]
            img_show = mmcv.imresize(img_show, (ori_w, ori_h))
            if 'ori_filename' in img_meta.keys():
                img_basename = os.path.basename(img_meta['ori_filename']) 
            elif 'filename' in img_meta.keys() :
                img_basename = os.path.basename(img_meta['filename'])
            out_file = osp.join(args.show_dir, img_basename)

            mmcv.imwrite( img_show,out_file)

    return model(return_loss=False, rescale=True, **data)[0]



def det2gt(data,model,score_thr):
    new_data = {'img':data['img'],'img_metas':data['img_metas']}
    # if torch.is_tensor(new_data['img'][0]):
    #     new_data['img'][0] = DataContainer([new_data['img'][0] ])
    #     new_data['img_metas'][0] = DataContainer([new_data['img_metas'][0] ])
    gt_labels = []
    gt_bboxes = []
    test_res = model(**new_data,return_loss=False)
    for res in test_res:
        bboxes = np.vstack(res)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(res)
        ]
        labels = np.concatenate(labels)
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :4]
        labels = labels[inds]
        gt_labels.append(DataContainer([[torch.Tensor(labels).long()]])) 
        gt_bboxes.append(DataContainer([[torch.Tensor(bboxes)]])) 
        bad_idx = torch.unique(torch.where(gt_bboxes[-1].data[0][0]<0)[0])
        if bad_idx.shape[0] != 0:
            good_idx = torch.zeros(gt_bboxes[-1].data[0][0].size(0)) == 0
            good_idx[bad_idx] = False
            gt_bboxes[-1].data[0][0] = gt_bboxes[-1].data[0][0][good_idx]
            gt_labels[-1].data[0][0] = gt_labels[-1].data[0][0][good_idx]
    new_data['gt_labels']=gt_labels
    new_data['gt_bboxes']=gt_bboxes
    return new_data
    