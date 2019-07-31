from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
#from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer
import time
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import shutil

parser = argparse.ArgumentParser(description='FaceBoxes')

parser.add_argument('-m', '--trained_model', default='weights/FaceBoxes.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset', default='PASCAL', type=str, choices=['AFW', 'PASCAL', 'FDDB'], help='dataset')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--show_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()

im_height = 720
im_width = 1280
resize = 1

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class FrameDataset(Dataset):
    def __init__(self, data_folder, limit=0, transform=None):
        self.data_folder = data_folder
        self.frames = []
        self.labels = []
        self.transform = transform
        for root, _, files in os.walk(self.data_folder):
            for file_name in files:
                if file_name[0] == '.':
                    continue
                img = cv2.imread(root + '/' + file_name, cv2.IMREAD_COLOR)
                img = np.float32(img)
                img -= (104, 117, 123)
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img)
                img = img.to(device)

                self.frames.append(img)
                self.labels.append(file_name)
                if limit > 0 and len(self.frames) >= limit:
                    break
            break 
    
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):         
        return self.frames[index], self.labels[index]
        


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # net and model
    net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    # print(net)

    src_folder = 'frames'
    dst_folder = 'test_result'
    batch_size = 20
    
    if dst_folder is not None:
        if os.path.isdir(dst_folder):
            shutil.rmtree(dst_folder)
        os.makedirs(dst_folder)

    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda:2")
    net = net.to(device)
    _t = {'forward_pass': Timer(), 'misc': Timer()}    
    dataset = FrameDataset(data_folder=src_folder, limit=0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
 
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data

    scale = torch.Tensor([im_width, im_height, im_width, im_height])
    scale = scale.to(device)
    
    for i, data in enumerate(loader, 0):  
        samples, labels = data
        _t['forward_pass'].tic()
        loc, conf = net(samples)
        _t['forward_pass'].toc()
        
        
        _t['misc'].tic()
        task_count = len(samples)
        whole_scores = conf.data.cpu().numpy()[:, 1]
        for j in range(task_count):
            boxes = decode(loc[j].data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = whole_scores[j*19620:j*19620 + 19620]
            
            # ignore low scores
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            #keep = py_cpu_nms(dets, args.nms_threshold)
            keep = nms(dets, args.nms_threshold, force_cpu=True)
            dets = dets[keep, :]

            # keep top-K faster NMS
            dets = dets[:args.keep_top_k, :]
            
            # debug
            # img = samples[j].cpu().numpy()
            # img = img.transpose(1, 2, 0)
            # img += (104, 117, 123)

            # for b in dets:
            #     # if b[4] < args.vis_thres:
            #     #     continue
            #     # img = img.astype('uint8')

            #     text = "{:.4f}".format(b[4])
            #     b = list(map(int, b))
            #     img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            #     cx = b[0]
            #     cy = b[1] + 12
            #     img = cv2.putText(img, text, (cx, cy),
            #                 cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            
            # cv2.imwrite(dst_folder + '/' + labels[j], img)
        
        _t['misc'].toc()
        
        print('{:d} forward: {:.4f}s misc: {:.4f}s'.format(i, _t['forward_pass'].diff, _t['misc'].diff))

                

        
        
        
    # # iter = iter(loader)
    # # images = iter.next()
    # # print (iter)
    



    # # # save file
    # # if not os.path.exists(args.save_folder):
    # #     os.makedirs(args.save_folder)
    # # fw = open(os.path.join(args.save_folder, args.dataset + '_dets.txt'), 'w')

    # # testing dataset


    # # resize = 1
    # # testing begin
    # # i = 0
    # for root, dirs, files in os.walk(testset_folder):
    #     # num_images = 3
    #     for file_name in files:
    #         if file_name[0] == '.':
    #             continue
    # # num_images = 10
    # # for i in range(num_images):

    #         image_path = testset_folder + '/' + file_name
    #         # image_path = 'test_images/oren_1.jpg'
    #         img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #         img = np.float32(img_raw)
    #         if resize != 1:
    #             # img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
    #             img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    #         # im_height, im_width, _ = img.shape
    #         scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    #         img -= (104, 117, 123)
    #         img = img.transpose(2, 0, 1)
    #         # img = torch.from_numpy(img).unsqueeze(0)
    #         # img = img.to(device)
    #         # scale = scale.to(device)

            
    #         _t['forward_pass'].tic()
    #         loc, conf = net(img)  # forward pass
    #         _t['forward_pass'].toc()
    #         _t['misc'].tic()
    #         priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    #         priors = priorbox.forward()
    #         priors = priors.to(device)
    #         prior_data = priors.data
    #         boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    #         boxes = boxes * scale / resize
    #         boxes = boxes.cpu().numpy()
    #         scores = conf.data.cpu().numpy()[:, 1]
            

    #         # ignore low scores
    #         inds = np.where(scores > args.confidence_threshold)[0]
    #         boxes = boxes[inds]
    #         scores = scores[inds]

    #         # keep top-K before NMS
    #         order = scores.argsort()[::-1][:args.top_k]
    #         boxes = boxes[order]
    #         scores = scores[order]

    #         # do NMS
    #         dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    #         #keep = py_cpu_nms(dets, args.nms_threshold)
    #         keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    #         dets = dets[keep, :]

    #         # keep top-K faster NMS
    #         dets = dets[:args.keep_top_k, :]
    #         _t['misc'].toc()
    #         print('forward_pass_time: {:.4f}s misc: {:.4f}s {:s}'.format(_t['forward_pass'].diff, _t['misc'].diff, file_name))

    #         # show image
    #         # if args.show_image:
    #         for b in dets:
    #             if b[4] < args.vis_thres:
    #                 continue
    #             text = "{:.4f}".format(b[4])
    #             b = list(map(int, b))
    #             cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    #             cx = b[0]
    #             cy = b[1] + 12
    #             cv2.putText(img_raw, text, (cx, cy),
    #                         cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    #             cv2.imwrite('test_result/' + file_name, img_raw)
    #             # cv2.imwrite('test_result/test.jpg', img_raw)
    #         # i += 1

    # fw.close()
