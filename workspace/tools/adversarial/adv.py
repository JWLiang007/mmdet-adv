import os.path as osp
import time
import os

import mmcv

from .difgsm import DIFGSM
from .tifgsm import  TIFGSM
from .mifgsm import  MIFGSM
from .vmifgsm import VMIFGSM
from .fgsm import FGSM
from .bim import BIM
from .nes import NES
from .zss import ZSS
from .sign_hunter import SIGN_HUNTER
# from .square_attack import SquareAttack
from .pgd import PGD
from .nifgsm import NIFGSM
from .deepfool import DeepFool
from .square import Square
from .prfa import PRFA
from .util import det2gt
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
    'square_attack':Square,
    'pgd':PGD,
    'nifgsm':NIFGSM,
    'deepfool':DeepFool,
    'prfa':PRFA
}


def single_gpu_adv(model,
                    data_loader,
                   args):
    model.eval()

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    attack = ta_factory[args.method](model, args)
    for i, data in enumerate(data_loader):

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
                    img_basename = img_meta['ori_filename'] 
                elif 'filename' in img_meta.keys() :
                    img_basename = os.path.basename(img_meta['filename'])
                out_file = osp.join(args.show_dir, img_basename)

                mmcv.imwrite( img_show,out_file)


        for _ in range(batch_size):
            prog_bar.update()


def multi_gpu_adv(model, data_loader, args):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    attack = ta_factory[args.method](model, args)
    for i, data in enumerate(data_loader):

        adv = attack(data)

        batch_size = adv.shape[0]
        if args.show_dir:
            img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor.detach().clone(), **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                out_file = osp.join(args.show_dir, img_meta['ori_filename'])

                mmcv.imwrite(img_show, out_file)

        if rank == 0:

            for _ in range(batch_size * world_size):
                prog_bar.update()

