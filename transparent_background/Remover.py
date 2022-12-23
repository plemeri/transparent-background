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
'base':     {'url': "https://drive.google.com/file/d/13oBl5MTVcWER3YU4fSxW3ATlVfueFQPY/view?usp=share_link",
             'md5': "bf17ee133697bcb91c72437f55e5c659",
             'base_size': [1024, 1024],
             'threshold': None,
             'ckpt_name': "ckpt_base.pth",
             'resize': dynamic_resize(L=1280)},
'fast':     {'url': "https://drive.google.com/file/d/1iRX-0MVbUjvAVns5MtVdng6CQlGOIo3m/view?usp=share_link",
             'md5': "9efdbfbcc49b79ef0f7891c83d2fd52f",
             'base_size': [384, 384],
             'threshold': 512,
             'ckpt_name': "ckpt_fast.pth",
             'resize': static_resize(size=[384, 384])}
}

class Remover:
    def __init__(self, fast=False, jit=False, device=None, ckpt=None):
        key = "fast" if fast else "base"
        self.meta = CONFIG[key]
    
        if device is not None:
            self.device=device
        else:
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda:0"
            elif version.parse(torch.__version__) > version.parse("1.13") and torch.backends.mps.is_available():
                self.device = "mps:0"
        
        download = False
        if ckpt is None:
            ckpt_dir = os.path.expanduser(os.path.join('~', '.transparent-background'))
            if os.path.isdir(ckpt_dir) is False:
                os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_name = self.meta['ckpt_name']
        
            if not os.path.isfile(os.path.join(ckpt_dir, ckpt_name)):
                download = True
            elif self.meta['md5'] != hashlib.md5(open(os.path.join(ckpt_dir, ckpt_name), 'rb').read()).hexdigest():
                download = True
            
            if download:
                gdown.download(self.meta['url'], os.path.join(ckpt_dir, ckpt_name), fuzzy=True)
        else:
            ckpt_dir, ckpt_name = os.path.split(os.path.abspath(ckpt))
        
        self.model = InSPyReNet_SwinB(depth=64, pretrained=False, **self.meta)
        self.model.eval()
        self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, ckpt_name), map_location='cpu'), strict=True)
        self.model = self.model.to(self.device)
        
        if jit:
            ckpt_name = self.meta['ckpt_name'].replace('.pth', '_{}.pt'.format(self.device))
            try:
                traced_model = torch.jit.load(os.path.join(ckpt_dir, ckpt_name), map_location=self.device)
                del self.model
                self.model = traced_model
            except:
                traced_model = torch.jit.trace(self.model, torch.rand(1, 3, *self.meta['base_size']).to(self.device), strict=True)
                del self.model
                self.model = traced_model
                torch.jit.save(self.model, os.path.join(ckpt_dir, ckpt_name))
    
        self.transform = transforms.Compose([static_resize(self.meta['base_size']) if jit else self.meta['resize'],
                                            tonumpy(),
                                            normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225]),
                                            totensor()])

        self.background = None
        desc = 'Mode={}, Device={}, Torchscript={}'.format(key, self.device, 'enabled' if jit else 'disabled')
        # print('=' * (len(desc) + 2) + '\n', desc, '\n' + '=' * (len(desc) + 2))
        print('Settings -> {}'.format(desc))
    
    def process(self, img, type='rgba'):
        shape = img.size[::-1]            
        x = self.transform(img)
        x = x.unsqueeze(0)
        x = x.to(self.device)
            
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
    remover = Remover(fast=args.fast, jit=args.jit, device=args.device, ckpt=args.ckpt)

    if args.source.isnumeric() is True:
        save_dir = None
        _format = 'Webcam'
        try:
            vcam = pyvirtualcam.Camera(width=640, height=480, fps=30)
        except:
            vcam = None

    elif os.path.isdir(args.source):
        save_dir = os.path.join(os.getcwd(), args.source.split(os.sep)[-1])
        _format = get_format(os.listdir(args.source))

    elif os.path.isfile(args.source):
        save_dir = os.getcwd()
        _format = get_format([args.source])
        
    else:
        raise FileNotFoundError('File or directory {} is invalid.'.format(args.source))
    
    if args.type == 'rgba' and _format == 'Video':
        raise AttributeError('type \'rgba\' cannot be applied to video input.')
        
    if args.dest is not None:
        save_dir = args.dest
        
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    loader = eval(_format + 'Loader')(args.source)
    frame_progress = tqdm.tqdm(total=len(loader), position=1 if (_format == 'Video' and len(loader) > 1) else 0, leave=False, bar_format='{desc:<15}{percentage:3.0f}%|{bar:50}{r_bar}')
    sample_progress = tqdm.tqdm(total=len(loader), desc='Total:', position=0, bar_format='{desc:<15}{percentage:3.0f}%|{bar:50}{r_bar}') if (_format == 'Video' and len(loader) > 1) else None
    writer = None

    for img, name in loader:
        frame_progress.set_description('{}'.format(name))
        if args.type.lower().endswith(('.jpg', '.jpeg', '.png')):
            outname = '{}_{}'.format(os.path.splitext(name)[0], os.path.splitext(os.path.split(args.type)[-1])[0])
        else:
            outname = '{}_{}'.format(os.path.splitext(name)[0], args.type) 
        
        if _format == 'Video' and writer is None:
            writer = cv2.VideoWriter(os.path.join(save_dir, '{}.mp4'.format(outname)), cv2.VideoWriter_fourcc(*'mp4v'), loader.fps, img.size)
            frame_progress.refresh()
            frame_progress.reset()
            frame_progress.total = int(loader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if sample_progress is not None:
                sample_progress.update()
            
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
        frame_progress.update()
        
    print('\nDone. Results are saved in {}'.format(os.path.abspath(save_dir)))
