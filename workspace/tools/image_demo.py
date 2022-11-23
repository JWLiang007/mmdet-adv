# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
from mmcv import Config
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from mmcv.utils.config import ConfigDict
from mmdet.datasets import replace_ImageToTensor
from adversarial.adv_image import  single_gpu_adv
import os 
from copy import deepcopy

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img',default=None, help='Image file')
    parser.add_argument('--config', default=None,help='Config file')
    parser.add_argument('--checkpoint',default=None, help='Checkpoint file')
    parser.add_argument('--out-dir', default=None, help='Path to output dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--show-score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--test-cfg',type=str, default=None, help='config of test model ')
    parser.add_argument(
        '--test-checkpoint',type=str, default=None, help='checkpoint of test model ')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='pseudo label score threshold (default: 0.0)')
    parser.add_argument('--p-init', type=float, default=0.05, help='initial p for square attack')
    parser.add_argument('--with-gt', action='store_true', help='attack with ground truth')
    parser.add_argument('--method', type=str, default='difgsm', help='attack method')
    parser.add_argument('--eps', type=float, default=15, help='maximum perturbation')
    parser.add_argument('--alpha', type=float, default=4, help='step size')
    parser.add_argument('--steps', type=int, default=5, help='step size')
    parser.add_argument('--decay', type=float, default=1.0, help='momentum factor')
    parser.add_argument('--resize_rate', type=float, default=0.9, help='resize factor used in input diversity')
    parser.add_argument('--diversity_prob', type=float, default=0.5,
                        help='the probability of applying input diversity of difgsm/tifgsm')
    parser.add_argument('--random_start', type=bool,default=True, help='using random initialization of delta')
    parser.add_argument('--kernel_name', type=str, default='gaussian',help='kernel name of tifgsm')
    parser.add_argument('--len_kernel', type=int, default=15, help='kernel length of tifgsm')
    parser.add_argument('--nsig', type=int, default=3, help=' radius of gaussian kernel of tisgsm')
    parser.add_argument('--beta', type=float, default=1.5, help=' the upper bound of neighborhood of vmifgsm')
    parser.add_argument('--N', type=int, default=20, help=' the number of sampled examples in the neighborhood')
    parser.add_argument('--overshoot', type=float, default=0.02, help='overshoot (float): parameter for enhancing the noise. (Default: 0.02)')
    parser.add_argument('--norm', type=str, default='Linf', help='Lp-norm of the attack.')
    parser.add_argument('--n_queries', type=int, default=500, help='max number of queries (each restart). (Default: 500)')
    parser.add_argument('--n_restarts', type=int, default=1, help='number of random restarts. ')
    parser.add_argument('--defense', type=bool, default=False, help='preprocessing defense')

    args = parser.parse_args()
    args.eps = args.eps / 255.0
    args.alpha = args.alpha / 255.0
    return args

def _det2json(result,classes,score_thr):
    """Convert detection results to COCO json style."""
    json_results = []
    for label in range(len(result)):
        bboxes = result[label]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['score'] = float(bboxes[i][4])
            if data['score'] < score_thr:
                continue
            data['bbox'] = bboxes[i]
            data['category_id'] = label
            data['category'] = classes[label]
            json_results.append(data)
    return json_results


def main(args=parse_args()):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device,args=args)
    # test a single image
    img_meta = inference_detector(model, args.img,return_meta=True)
    ori_result = model(return_loss=False, rescale=True, **img_meta)[0]
    show_result_pyplot(
        model,
        args.img,
        ori_result,
        palette=args.palette,
        score_thr=args.show_score_thr,
        out_file=os.path.join(args.out_dir,'ori.png'))
    defense_result = None 
    if args.defense:
        defense_result = single_gpu_adv(model,img_meta,args)
        show_result_pyplot(
            model,
            args.img,
            defense_result,
            palette=args.palette,
            score_thr=args.show_score_thr,
            out_file=os.path.join(args.out_dir,'defense.png'))
        args.defense=False
    adv_result = single_gpu_adv(model,img_meta,args)
    if args.test_cfg is not None and args.test_checkpoint is not None:
        model = init_detector(args.test_cfg, args.test_checkpoint, device=args.device,args=args)
    show_result_pyplot(
        model,
        args.img,
        adv_result,
        palette=args.palette,
        score_thr=args.show_score_thr,
        out_file=os.path.join(args.out_dir,'adv.png'))
    # show the results
    ori_info = _det2json(ori_result,model.CLASSES,args.show_score_thr)
    adv_info = _det2json(adv_result,model.CLASSES,args.show_score_thr)
    defense_info = None 
    if defense_result is not None :
        defense_info = _det2json(defense_result,model.CLASSES,args.show_score_thr)
    return ori_info,adv_info,defense_info



async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device,args=args)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
