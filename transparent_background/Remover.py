import os
import sys
import tqdm
import gdown
import torch
import warnings
import pyvirtualcam

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
         'md5': "3bb068bd44574f0a0c39a8da900b1cf9",
         'base_size': [1024, 1024],
         'threshold': None,
         'ckpt_name': "ckpt_base.pth",
         'resize': dynamic_resize(L=1280)},
'fast': {'url': "https://drive.google.com/file/d/1iRX-0MVbUjvAVns5MtVdng6CQlGOIo3m/view?usp=share_link",
         'md5': "735a2fe8519bc12290f86bf7b8b395ff",
         'base_size': [384, 384],
         'threshold': 512,
         'ckpt_name': "ckpt_fast.pth",
         'resize': static_resize(size=[384, 384])}
}

class Remover:
    def __init__(self, fast=False):
        self.meta = CONFIG['fast' if fast else 'base']
        self.model = InSPyReNet_SwinB(depth=64, pretrained=False, **self.meta)
        self.model.eval()

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

        self.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, ckpt_name), map_location='cpu'), strict=True)

        self.backend = "cpu"
        if torch.cuda.is_available():
            self.backend = "cuda:0"
        elif version.parse(torch.__version__) > version.parse("1.13") and torch.backends.mps.is_available():
            self.backend = "mps:0"

        self.model = self.model.to(self.backend)
    
        self.transform = transforms.Compose([self.meta['resize'],
                                            tonumpy(),
                                            normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225]),
                                            totensor()])

        self.background = None
    
    def process(self, img, type='rgba'):
        shape = img.size[::-1]            
        x = self.transform(img)
        x = x.unsqueeze(0)
        x = x.to(self.backend)
            
        with torch.no_grad():
            pred = self.model(x)

        pred = F.interpolate(pred, shape, mode='bilinear', align_corners=True)
        pred = pred.data.cpu()
        pred = pred.numpy().squeeze()   
        
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
    remover = Remover(fast=args.fast)

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
                cv2.imshow('transparent-background', out)