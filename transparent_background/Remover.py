import os
import sys
import tqdm
import gdown
import torch
import warnings

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image

filepath = os.path.abspath(__file__)
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from transparent_background.InSPyReNet import InSPyReNet_SwinB
from transparent_background.utils import *

warnings.filterwarnings("ignore")

URL = "https://drive.google.com/file/d/13oBl5MTVcWER3YU4fSxW3ATlVfueFQPY/view?usp=share_link"
MD5 = "3bb068bd44574f0a0c39a8da900b1cf9"

class Remover:
    def __init__(self, base_size=[1024, 1024], threshold=None, enable_pb=False, jit=False):
        if jit & enable_pb:
            raise AssertionError('jit and enable pyramid blending cannot be used together')
        
        self.model = InSPyReNet_SwinB(depth=64, pretrained=False, base_size=base_size, threshold=threshold)
        self.model.eval()

        checkpoint_dir = os.path.expanduser(os.path.join('~', '.transparent-background'))
        if os.path.isdir(checkpoint_dir) is False:
            os.makedirs(checkpoint_dir, exist_ok=True)
    
        download = False
        if not os.path.isfile(os.path.join(checkpoint_dir, 'latest.pth')):
            download = True
        elif MD5 != hashlib.md5(open(os.path.join(checkpoint_dir, 'latest.pth'), 'rb').read()).hexdigest():
            download = True
        
        if download:
            gdown.download(URL, os.path.join(checkpoint_dir, 'latest.pth'), fuzzy=True)

        self.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'latest.pth'), map_location='cpu'), strict=True)
        
        self.use_gpu = False
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.use_gpu = True
            
        if jit:
            self.model = torch.jit.trace(self.model, torch.rand(1, 3, *base_size).cuda(), strict=False)
            torch.jit.save(self.model, os.path.join(checkpoint_dir, 'jit.pt'))
            
        self.transform = transforms.Compose([dynamic_resize(L=1280),
                                            tonumpy(),
                                            normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225]),
                                            totensor()])

        self.background = None
    
    def process(self, img, type='rgba'):
        shape = img.size[::-1]            
        x = self.transform(img)
        x = x.unsqueeze(0)
        
        if self.use_gpu:
            x = x.cuda()
            
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
    remover = Remover(jit=args.jit)

    if os.path.isdir(args.source):
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