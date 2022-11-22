import os

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim

from mmdet.apis import init_detector,show_result_pyplot
from mmdet.datasets.pipelines import Compose,LoadImageFromWebcam,ToTensor
from mmcv.parallel import collate, scatter
from mmcv.ops.point_sample import bilinear_grid_sample
from PIL import Image
from renderer import Renderer as Renderer
import numpy as np
from mmcv.parallel.data_container import DataContainer
from mmdet.datasets import replace_ImageToTensor
from argparse import ArgumentParser
import random 

parser = ArgumentParser()
parser.add_argument(
    '--device', default='cuda:0', help='Device used for inference')
parser.add_argument(
    '--gpu-id',
    type=int,
    default=0,
    help='id of gpu to use '
    '(only applicable to non-distributed testing)')
args = parser.parse_args()

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

def set_backgroud(x, mask, color):
    """Set background color according to a boolean mask
    Args:
        x: A 4-D tensor with shape [batch_size, 3, height, width]
        mask: boolean mask with shape [batch_size, 1, height, width]
        color: background color with shape [batch_size, 3, 1, 1]
    """
    mask = torch.tile(mask, [1, 3, 1, 1])
    inv = torch.logical_not(mask)

    return mask.type(torch.float32) * x + inv.type(torch.float32) * color

def normalize(x, y):
    x_t = x.clone().detach()
    y_t = y.clone().detach()
    minimum = torch.minimum(torch.amin(input=x_t, dim=[2, 3], keepdim=True), torch.amin(input=y_t, dim=[1, 2, 3], keepdim=True))
    maximum = torch.maximum(torch.amax(input=x_t, dim=[2, 3], keepdim=True), torch.amax(input=y_t, dim=[1, 2, 3], keepdim=True))

    minimum = torch.minimum(minimum, torch.zeros([1]).cuda())
    maximum = torch.maximum(maximum, torch.ones([1]).cuda())

    return (x - minimum) / (maximum - minimum), (y - minimum) / (maximum - minimum)

def transform(x, a, b):
    """Apply transform a * x + b element-wise
    """
    return torch.add(torch.mul(a, x), b)

def mmnormalize_(img, mean, std):
    """Inplace normalize an image with mean and std.

    Args:
        img (Tensor): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    # assert img.dtype != np.uint8
    mean = torch.reshape(torch.from_numpy(np.float64(mean.reshape(1, -1))),(1,3,1,1)).cuda()
    stdinv = torch.reshape(torch.from_numpy(1 / np.float64(std.reshape(1, -1))),(1,3,1,1)).cuda()
    img  = img - mean 
    img = img * stdinv
    # cv2.subtract(img, mean, img)  # inplace
    # cv2.multiply(img, stdinv, img)  # inplace
    return img


# config
config_file = '../workspace/config/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '../workspace/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
texture_file  = '3dmodel/body.png'
texture_mask_file = '3dmodel/body_mask.png'
obj_file = '3dmodel/DJS2022.obj'
device = args.device
out_dir = 'res/'
batch_size = 1
lr = 1
l2_weight = 0.0
photo_error = False
print_error = False
render_size = (1333, 800)
target_cls = (2,7) # 2 for car , 7 for truck
misled_cls = 0
alpha = 1


# renderer config
camera_distance_min = 700
camera_distance_max = 1000
x_translation_min = -5
x_translation_max = 5
y_translation_min = -5
y_translation_max = 5
background_min = 0.1
background_max = 1.0
channel_mult_min=0.7
channel_mult_max=1.3
channel_add_min=-0.15
channel_add_max=0.15
light_mult_min=0.5
light_mult_max=2.0
light_add_min=-0.15
light_add_max=0.15

os.makedirs(out_dir,exist_ok=True)

model = init_detector(config_file, checkpoint_file, device='cuda:0',args=args)
# model.test_cfg['rcnn'].score_thr = 1e-5
# model.test_cfg['rcnn'].max_per_img = 300
model.eval()
test_pipeline = model.cfg._cfg_dict['test_pipeline']
test_pipeline = replace_ImageToTensor(test_pipeline)
test_pipeline[0]['type'] = 'LoadImageFromWebcam'
test_pipeline[0]['channel_order'] = 'rgb'
for trans in test_pipeline[1].transforms:
    if trans['type'] =='Pad':
        trans['size_divisor'] = 1
    if trans['type'] == 'Normalize':
        trans['to_rgb'] = False
        mean = trans['mean']
        std = trans['std']

test_pipeline = Compose(test_pipeline)

nor = transforms.Normalize(mean,std)

img_metas = {}

texture = Image.open(texture_file)
texture_mask = Image.open(texture_mask_file)
height, width = texture.size

# renderer = Renderer(obj_file,(299, 299))
renderer = Renderer(render_size)
renderer.load_obj(obj_file)

renderer.set_parameters(
    camera_distance=(camera_distance_min, camera_distance_max),
    x_translation=(x_translation_min, x_translation_max),
    y_translation=(y_translation_min, y_translation_max)
)

texture = np.asarray(texture).astype(np.float32)[..., :3] 
texture_mask = np.asarray(texture_mask).astype(np.float32)[..., :3] / 255

# convert hwc to chw
texture_mask = torch.from_numpy(texture_mask.transpose(2,0,1)).cuda()

std_texture = torch.from_numpy(texture.transpose(2,0,1)).cuda()

adv_texture = std_texture.clone().detach()
# adv_texture.requires_grad = True
# optimizer = optim.SGD([adv_texture], lr=lr)
find_ori = 0
find_adv = 0
loss_summary = 0
for i in range(30000):
    diff = (adv_texture - std_texture) * texture_mask
    # diff = torch.clamp(diff,0,127)
    _std_texture = torch.unsqueeze(std_texture,dim = 0)
    _std_texture.requires_grad = False
    # _std_texture = torch.tanh(torch.tile(torch.unsqueeze(std_texture,dim = 0),[batch_size,1,1,1]))
    _adv_texture = torch.unsqueeze(std_texture + diff ,dim = 0).clone().detach()
    _adv_texture.requires_grad = True
    # _adv_texture = torch.tanh(torch.tile(torch.unsqueeze(adv_texture + diff ,dim = 0), [batch_size,1,1,1]))

    if print_error:
        multiplier = torch.distributions.uniform.Uniform(channel_mult_min,channel_mult_max).rsample([batch_size,3, 1, 1]).cuda()
        addend = torch.distributions.uniform.Uniform(channel_add_min,channel_add_max).rsample([batch_size,3, 1, 1]).cuda()
        _std_texture = transform(_std_texture,multiplier,addend)
        _adv_texture = transform(_adv_texture,multiplier,addend)

    uv = renderer.render(batch_size) * \
             np.asarray([width - 1, height - 1], dtype=np.float32)

    # grid_sample requires to map location to (-1,1)
    uv[...,0] = (uv[...,0] - width/2.) / (width/2.)
    uv[...,1] = (uv[...,1] - height/2.) / (height/2.)

    _std_images = F.grid_sample(_std_texture,torch.from_numpy(uv).cuda())
    _adv_images = F.grid_sample(_adv_texture,torch.from_numpy(uv).cuda())
    # if i % 10 == 0:
    #     img = np.transpose(_std_images.clone().detach().squeeze(0).cpu().numpy(), [1, 2, 0]) * 255
    #     out = Image.fromarray(img.astype(np.uint8))
    #     out.save('{}/render_{}.png'.format('render_img', i))

    # mask = torch.all(torch.not_equal(torch.from_numpy(uv.transpose([0,3,1,2])), 0.0), dim=1, keepdim=True).cuda()
    # color = torch.distributions.uniform.Uniform(background_min,background_max).rsample([batch_size,3, 1, 1]).cuda()

    # _std_images = set_backgroud(_std_images, mask, color)
    # _adv_images = set_backgroud(_adv_images, mask, color)

    if photo_error:
        multiplier = torch.distributions.uniform.Uniform(light_mult_min,light_mult_max).rsample([batch_size,1, 1, 1]).cuda()
        addend = torch.distributions.uniform.Uniform(light_add_min,light_add_max).rsample([batch_size,1, 1, 1]).cuda()
        _std_images = transform(_std_images, multiplier, addend)
        _adv_images = transform(_adv_images, multiplier, addend)
        # add gaussian noise with avg=0 and stdv=0.1
        gaussian_noise = (0.1**0.5)*torch.randn(_std_images.shape).cuda()
        _std_images += gaussian_noise
        _adv_images += gaussian_noise


    # _std_images, _adv_images = normalize(_std_images, _adv_images)

    # _std_images = 2.0 * _std_images - 1.0
    # _adv_images = 2.0 * _adv_images - 1.0
    std_images = _std_images
    # std_images = (_std_images + 1) * 127.5
    adv_images  = _adv_images 
    # adv_images  = ( _adv_images + 1 ) * 127.5


    if len(img_metas) == 0:

        std_images_np = std_images.clone().detach()[0, ...].cpu().numpy().transpose(1, 2, 0)

        img_metas['img'] = std_images_np
        img_metas = test_pipeline(img_metas)
        img_metas = collate([img_metas], samples_per_gpu=1)
        # img_metas = scatter(img_metas, [device])[0]
    std_images = nor(std_images)
    adv_images = nor(adv_images)
    img_metas['img'][0].data[0] =  std_images
    # img_metas['img'][0] = nor(adv_images.squeeze(0)).unsqueeze(0).cuda()
    # std_images = nor(std_images.squeeze(0)).unsqueeze(0).cuda()
    # det_res = model(return_loss=False, rescale=True,img=img_metas['img'], img_metas=img_metas['img_metas'])
    data = det2gt(img_metas,model,0.0)
    # scores = model(return_loss=False, rescale=True, img=img_metas['img'], img_metas=img_metas['img_metas'],return_score=True)
    # feat = model.extract_feat(img_metas['img'][0].unsqueeze(0).cuda())
    # proposals = model.rpn_head.simple_test(feat,[img_metas['img_metas'][0].data])[0]
    # scores = model.roi_head.simple_test(feat,[proposals],[img_metas['img_metas'][0].data],return_score=True)

    # loss_cls = model.roi_head.bbox_head.loss_cls(scores,torch.zeros(size=(scores.shape[0],),dtype=torch.long).to(device),torch.ones(size=(scores.shape[0],)).to(device),scores.shape[0])
    # loss_diff = torch.sum(torch.square(torch.sub(std_images,img_metas['img'][0])))
    # loss = loss_cls + l2_weight * loss_diff
    gt_labels = data['gt_labels'][0].data[0][0]
    for j in range(len(gt_labels)):
        if int(gt_labels[j].item()) in (target_cls):
            # gt_labels[j] = misled_cls
            find_ori+=1
        else:
            gt_labels[j] = random.choice(target_cls)
    new_data = {}
    new_data['img_metas'] = data['img_metas'][0].data[0]
    new_data['img'] = adv_images

    # new_data['img'] = data['img'][0].data[0]
    losses = model(**new_data, return_loss=True,gt_bboxes=data['gt_bboxes'][0].data[0],
                    gt_labels=data['gt_labels'][0].data[0])
    loss ,_ = model.module._parse_losses(losses)

    model.zero_grad()
    # optimizer.zero_grad()
    loss.backward()
    # optimizer.step()
    grad = torch.sign(_adv_texture.grad)
    adv_texture = torch.clamp((_adv_texture + alpha * grad).squeeze(0),0,255)
    loss_summary += loss.clone().detach().cpu().numpy()
    
    img_metas['img'][0].data[0] = adv_images
    data = det2gt(img_metas,model,0.0)
    gt_labels = data['gt_labels'][0].data[0][0]
    for j in range(len(gt_labels)):
        if int(gt_labels[j].item()) in (target_cls):
            # gt_labels[j] = misled_cls
            find_adv+=1

    if i % 100 == 0:
        print('loss_cls:  ', loss_summary,
              # '   loss_diff:  ',loss_diff.clone().detach().cpu().numpy(),
              # '  loss_all:  ', loss.clone().detach().cpu().numpy(),
              '  find_ori : ', find_ori,'  find_adv : ', find_adv)
        find_ori = 0
        find_adv = 0
        loss_summary = 0
        out = Image.fromarray((adv_texture.clone().detach().cpu().numpy().transpose(1,2,0)).astype(np.uint8))
        out.save(os.path.join(out_dir,'adv_{}.png'.format(i)))
        
        adv_result  = model(**img_metas,return_loss=False)[0]
        show_img  =  cv2.cvtColor(_adv_images.squeeze(0).clone().detach().cpu().numpy().transpose(1,2,0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        show_result_pyplot(
            model,
            show_img,
            adv_result,
            palette='coco',
            score_thr=0.3,
            out_file=os.path.join('det',str(i)+'_det.png'))
    pass

print('finish...')


