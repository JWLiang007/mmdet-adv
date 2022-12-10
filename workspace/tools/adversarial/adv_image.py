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
from .pgd import PGD
from .nifgsm import NIFGSM
from .deepfool import DeepFool
from .square import Square
from .prfa import PRFA
from .jitter import Jitter
from .zosignsgd import ZOsignSGD
from .util import det2gt
import numpy as np

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
    'square_attack':Square,
    'pgd':PGD,
    'nifgsm':NIFGSM,
    'deepfool':DeepFool,
    'prfa':PRFA,
    'jitter': Jitter,
    'ZOsignSGD':ZOsignSGD
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

    # batch_size = adv.shape[0]
    # if args.show_dir:
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
            # if 'ori_filename' in img_meta.keys():
            #     img_basename = os.path.basename(img_meta['ori_filename']) 
            # elif 'filename' in img_meta.keys() :
            #     img_basename = os.path.basename(img_meta['filename'])
            # out_file = osp.join(args.show_dir, img_basename)

            # mmcv.imwrite( img_show,out_file)

    return model(return_loss=False, rescale=True, **data)[0], img_show




    