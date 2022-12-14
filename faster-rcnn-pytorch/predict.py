#----------------------------------------------------#
#   对视频中的predict.py进行了修改，
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import time
import os 
import cv2
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from frcnn import FRCNN

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img',default=None, help='Image file')
    parser.add_argument('--method',default='fgsm', help='adv attack method')
    parser.add_argument('--mode',default='adv', help='run mode')
    args = parser.parse_args()
    return args

def main(args=parse_args()):
    args = parse_args()
    frcnn = FRCNN()
    if args.method == 'fgsm':
        advt_frcnn = FRCNN(model_path='/workspace/mmdetection/faster-rcnn-pytorch/voc/FGSM-AT.pth') 
        advt_method =  advt_frcnn.train_util.train_FGSM_step
        adv_method = frcnn.train_util.train_FGSM_step
    elif args.method == 'pgd':
        advt_frcnn = FRCNN(model_path='/workspace/mmdetection/faster-rcnn-pytorch/voc/PGD-AT.pth') 
        advt_method =  advt_frcnn.train_util.train_PGD_step
        adv_method = frcnn.train_util.train_FGSM_step
    elif args.method == 'square':
        advt_frcnn = frcnn
        advt_method =  advt_frcnn.train_util.train_SquareAttack_step
    #-------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测
    #   'video'表示视频检测
    #   'fps'表示测试fps
    #-------------------------------------------------------------------------#
    mode = args.mode
    #-------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出才会完成完整的保存步骤，不可直接结束程序。
    #-------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0

    if mode == "predict":
        '''
        1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
        具体流程可以参考get_dr_txt.py，在get_dr_txt.py即实现了遍历还实现了目标信息的保存。
        2、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        3、如果想要获得预测框的坐标，可以进入frcnn.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        4、如果想要利用预测框截取下目标，可以进入frcnn.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        5、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入frcnn.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = frcnn.detect_image(image)
                # os.makedirs('res',exist_ok=True)
                # r_image.save('res/det.png')
                # r_image.show()

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref,frame=capture.read()
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(frcnn.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        test_interval = 100
        img = Image.open('img/street.jpg')
        tact_time = frcnn.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
    
    elif mode == 'adv':
        # while True:
        img = args.img
        # img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            # continue
        else:
            ori_res, images, bbox, label, conf ,ori_det_infos= frcnn.detect_image(image,return_det=True)
            bbox = np.expand_dims(bbox,0)
            label = np.expand_dims(label,0)
            # _label = np.zeros_like(label)
            # _label[label==0] = 1
            # label = _label
            new_images = adv_method(images, bbox,label,1,return_img = True)
            new_images = Image.fromarray((new_images*255).clone().detach().squeeze(0).cpu().numpy().transpose(1,2,0).astype(np.uint8)).resize(image.size)
            adv_res,adv_det_infos = frcnn.detect_image(new_images,return_det=False)
            
            new_images = advt_method(images, bbox,label,1,return_img = True)
            new_images = Image.fromarray((new_images*255).clone().detach().squeeze(0).cpu().numpy().transpose(1,2,0).astype(np.uint8)).resize(image.size)
            advt_res,advt_det_infos = advt_frcnn.detect_image(new_images,return_det=False)
            base_dir = os.path.join('res',args.method)
            os.makedirs(base_dir,exist_ok=True)
            ori_res.save(os.path.join(base_dir,'ori_det.png'))
            adv_res.save(os.path.join(base_dir,'adv_det.png'))
            advt_res.save(os.path.join(base_dir,'advt_det.png'))
            # os.makedirs('res',exist_ok=True)
            # r_image.save('res/det.png')
            # r_image.show()
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video' or 'fps'.")

    return ori_det_infos,adv_det_infos,advt_det_infos

if __name__ == "__main__":
    args = parse_args()
    main(args)