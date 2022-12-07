import os
import sys
import tqdm
import gdown
import torch
import warnings
import pyvirtualcam
import onnxruntime

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from packaging import version

filepath = os.path.abspath(__file__)
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from transparent_background.InSPyReNet import InSPyReNet_SwinB
from transparent_background.utils import *

warnings.filterwarnings("ignore")

CONFIG = {
'base': {'url': "https://drive.google.com/file/d/13oBl5MTVcWER3YU4fSxW3ATlVfueFQPY/view?usp=share_link",
         'md5': "576ebd735e78bd9d864bd0392c74d2eb",
         'base_size': [1024, 1024],
         'threshold': None,
         'ckpt_name': "ckpt_base.pth",
         'tfs': [dynamic_resize(L=1280),
                 tonumpy(),
                 normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                 transpose(),
                 totensor()]
         },
'base_onnx': {'url': "https://drive.google.com/file/d/1k3JfBndJ4rgU3HRCq98efRVj1CgczlSQ/view?usp=share_link",
              'md5': "d3508c6fbcb8968710ca80b710da179f",
              'base_size': [1024, 1024],
              'threshold': None,
              'ckpt_name': "ckpt_base.onnx",
              'tfs': [static_resize((1024, 1024)),
                      tonumpy(),
                      normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
                      transpose()]
         },
'fast': {'url': "https://drive.google.com/file/d/1iRX-0MVbUjvAVns5MtVdng6CQlGOIo3m/view?usp=share_link",
         'md5': "9efdbfbcc49b79ef0f7891c83d2fd52f",
         'base_size': [384, 384],
         'threshold': None,
         'ckpt_name': "ckpt_fast.pth",
         'tfs': [static_resize(size=[384, 384]),
                 tonumpy(),
                 normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                 transpose(),
                 totensor()]
         },
# 'fast_onnx': {'url': "https://drive.google.com/file/d/13oBl5MTVcWER3YU4fSxW3ATlVfueFQPY/view?usp=share_link",
#               'md5': "1bbed865c233daff825e387cec039caa",
#               'base_size': [384, 384],
#               'threshold': None,
#               'ckpt_name': "ckpt_fast.onnx",
#               'tfs': [static_resize((384, 384)),
#                       tonumpy(),
#                       normalize(mean=[0.485, 0.456, 0.406], 
#                                 std=[0.229, 0.224, 0.225]),
#                       transpose()]
#          },
}

class ONNXWrapper:
    def __init__(self):
        self.session = None
        
    def load(self, ckpt):
        self.session = onnxruntime.InferenceSession(ckpt)
        
    def __call__(self, img):
        return self.session.run(None, {'img': img})[0]

class Remover:
    def __init__(self, fast=False, onnx=False):
        key = 'fast' if fast else 'base'
        key = key + '_onnx' if onnx else key
        
        self.meta = CONFIG[key]
        
        checkpoint_dir = os.path.expanduser(os.path.join('~', '.transparent-background'))
        if os.path.isdir(checkpoint_dir) is False:
            os.makedirs(checkpoint_dir, exist_ok=True)
    
        download = False
        ckpt_name = self.meta['ckpt_name']        
        if not os.path.isfile(os.path.join(checkpoint_dir, ckpt_name)):
            download = True
        elif self.meta['md5'] != hashlib.md5(open(os.path.join(checkpoint_dir, ckpt_name), 'rb').read()).hexdigest():
            download = True
        
        if download:
            gdown.download(self.meta['url'], os.path.join(checkpoint_dir, ckpt_name), fuzzy=True)
            
        if onnx:
            self.model = ONNXWrapper()
            self.backend = 'onnx'            
            self.model.load(os.path.join(checkpoint_dir, ckpt_name))            
        else:
            self.model = InSPyReNet_SwinB(depth=64, pretrained=False, **self.meta)
            self.model.eval()
            
            self.backend = "cpu"
            if torch.cuda.is_available():
                self.backend = "cuda:0"
            elif version.parse(torch.__version__) > version.parse("1.13") and torch.backends.mps.is_available():
                self.backend = "mps:0"

            self.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, ckpt_name), map_location='cpu'), strict=True)
            self.model = self.model.to(self.backend)

        self.transform = transforms.Compose(self.meta['tfs'])        
        self.background = None
    
    def process(self, img, type='rgba'):
        shape = img.size[::-1]            
        x = self.transform(img)
        
        if self.backend != 'onnx':
            x = x.unsqueeze(0)
            x = x.to(self.backend)
        else:
            x = x[np.newaxis]
            
        with torch.no_grad():
            pred = self.model(x)

        if self.backend != 'onnx':
            pred = F.interpolate(pred, shape, mode='bilinear', align_corners=True)
            pred = pred.data.cpu()
            pred = pred.numpy().squeeze()   
        else:
            pred = cv2.resize(pred.squeeze(), shape[::-1])
            
        img = np.array(img)
        
        if type == 'map':
            img = (np.stack([pred] * 3, axis=-1) * 255).astype(np.uint8)
        elif type == 'rgba':
            r, g, b = cv2.split(img)
            pred = (pred * 255).astype(np.uint8)
            img = cv2.merge([r, g, b, pred])
        elif type == 'green':
            bg = np.stack([np.ones_like(pred)] * 3, axis=-1) * [120, 255, 155]
            img = img * pred[..., np.newaxis] + bg * (1 - pred[..., np.newaxis])
        elif type == 'blur':
            img = img * pred[..., np.newaxis] + cv2.GaussianBlur(img, (0, 0), 15) * (1 - pred[..., np.newaxis])
        elif type == 'overlay':
            bg = (np.stack([np.ones_like(pred)] * 3, axis=-1) * [120, 255, 155] + img) // 2
            img = bg * pred[..., np.newaxis] + img * (1 - pred[..., np.newaxis])
            border = cv2.Canny(((pred > .5) * 255).astype(np.uint8), 50, 100)
            img[border != 0] = [120, 255, 155]
        elif type.lower().endswith(('.jpg', '.jpeg', '.png')):
            if self.background is None:
                self.background = cv2.cvtColor(cv2.imread(type), cv2.COLOR_BGR2RGB)
                self.background = cv2.resize(self.background, img.shape[:2][::-1])
            img = img * pred[..., np.newaxis] + self.background * (1 - pred[..., np.newaxis])
            
        return img.astype(np.uint8) 

def console():
    args = parse_args()
    remover = Remover(fast=args.fast, onnx=args.onnx)

    if args.source.isnumeric() is True:
        save_dir = None
        _format = 'Webcam'
        try:
            vcam = pyvirtualcam.Camera(width=640, height=480, fps=30)
            print(f'Using virtual camera: {vcam.device}\n')
        except:
            vcam = None
            print('virtual camera not available, visualzing instead.')

    elif os.path.isdir(args.source):
        save_dir = os.path.join(os.getcwd(), args.source.split(os.sep)[-1])
        _format = get_format(os.listdir(args.source))

    elif os.path.isfile(args.source):
        save_dir = os.getcwd()
        _format = get_format([args.source])
        
    else:
        raise FileNotFoundError('File or directory {} is invalid'.format(args.source))
    
    if args.type == 'rgba' and _format == 'Video':
        raise AttributeError('type rgba cannot be applied to video input')
        
    if args.dest is not None:
        save_dir = args.dest
        
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    sample_list = eval(_format + 'Loader')(args.source)
    samples = tqdm.tqdm(sample_list, desc='Transparent Background', total=len(sample_list), position=0, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
        
    writer = None

    for img, name in samples:
        if args.type.lower().endswith(('.jpg', '.jpeg', '.png')):
            outname = '{}_{}'.format(name, os.path.splitext(os.path.split(args.type)[-1])[0])
        else:
            outname = '{}_{}'.format(name, args.type) 
        
        if _format == 'Video' and writer is None:
            writer = cv2.VideoWriter(os.path.join(save_dir, '{}.mp4'.format(outname)), cv2.VideoWriter_fourcc(*'mp4v'), sample_list.fps, img.size)
            samples.total += int(sample_list.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if _format == 'Video' and img is None:
            if writer is not None:
                writer.release()
            writer = None
            continue
        
        out = remover.process(img, type=args.type)
                                        
        if _format == 'Image':
            Image.fromarray(out).save(os.path.join(save_dir, '{}.png'.format(outname)))
        elif _format == 'Video' and writer is not None:
            writer.write(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        elif _format == 'Webcam':
            if vcam is not None:
                vcam.send(out)
                vcam.sleep_until_next_frame()
            else:
                cv2.imshow('transparent-background', cv2.cvtColor(out, cv2.COLOR_BGR2RGB))