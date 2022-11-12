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

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
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
        '--score-thr',
        type=float,
        default=0.0,
        help='pseudo label score threshold (default: 0.0)')
    parser.add_argument('--p-init', type=float, default=0.9, help='initial p for square attack')
    parser.add_argument('--with-gt', action='store_true', help='attack with ground truth')
    parser.add_argument('--method', type=str, default='difgsm', help='attack method')
    parser.add_argument('--eps', type=float, default=15, help='maximum perturbation')
    parser.add_argument('--alpha', type=float, default=4, help='step size')
    parser.add_argument('--steps', type=int, default=5, help='step size')
    parser.add_argument('--decay', type=float, default=1.0, help='momentum factor')
    parser.add_argument('--resize_rate', type=float, default=0.9, help='resize factor used in input diversity')
    parser.add_argument('--diversity_prob', type=float, default=0.5,
                        help='the probability of applying input diversity of difgsm/tifgsm')
    parser.add_argument('--random_start', action='store_true', help='using random initialization of delta')
    parser.add_argument('--kernel_name', type=str, default='gaussian',help='kernel name of tifgsm')
    parser.add_argument('--len_kernel', type=int, default=15, help='kernel length of tifgsm')
    parser.add_argument('--nsig', type=int, default=3, help=' radius of gaussian kernel of tisgsm')
    parser.add_argument('--beta', type=float, default=1.5, help=' the upper bound of neighborhood of vmifgsm')
    parser.add_argument('--N', type=int, default=20, help=' the number of sampled examples in the neighborhood')
    args = parser.parse_args()
    args.eps = args.eps / 255.0
    args.alpha = args.alpha / 255.0
    return args




def main(args):
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
    adv_result = single_gpu_adv(model,img_meta,args)
    show_result_pyplot(
        model,
        args.img,
        adv_result,
        palette=args.palette,
        score_thr=args.show_score_thr,
        out_file=os.path.join(args.out_dir,'adv.png'))
    # show the results



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
