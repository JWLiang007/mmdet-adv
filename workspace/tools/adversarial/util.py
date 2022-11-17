
import numpy as np
import torch 
from mmcv.parallel.data_container import DataContainer
def get_gt_bboxes_scores_and_labels(Anns, cat2label, img_name, scale_factor, ncls, scale_flag=None):
    bboxes = []
    scores = []
    labels = []

    if '/' in img_name:
        img_name = img_name.split('/')[-1]
    # modified
    for i in Anns.keys():
        if type(i) == int :
            img_id = int(img_name.split('.')[0])
        else:
            img_id = img_name.split('.')[0]
        break

    img2bboxes = [Anns[img_id][i]['bbox'] for i in range(len(Anns[img_id]))]
    img2labels = [cat2label[Anns[img_id][i]['category_id']]  for i in range(len(Anns[img_id]))]
    if scale_flag:
        img2bboxes = np.array(img2bboxes*scale_factor)
    img2bboxes = np.array(img2bboxes)
    xs_left, ys_left, ws, hs = img2bboxes[:, 0], img2bboxes[:, 1], img2bboxes[:, 2], img2bboxes[:, 3]
    bboxes = np.column_stack((xs_left, ys_left, xs_left+ws, ys_left+hs))
    labels = np.array(img2labels)
    scores = np.zeros((len(labels), ncls))
    for i in range(len(labels)):
        scores[i, labels[i]] = 1.0
    return bboxes, scores, labels


def mmdet_clamp(img,lb,ub):
    assert img.shape[1] == lb.shape[0] and img.shape[1] ==  ub.shape[0]
    for chn in range(img.shape[1]):
        img[:,chn:chn+1,:,:] = torch.clamp(img[:,chn:chn+1,:,:], min=lb[chn], max=ub[chn]).detach()
    return img

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