import os
import sys
import tqdm
import wget
import gdown
import torch
import shutil
import base64
import warnings
import importlib

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import albumentations as A
import albumentations.pytorch as AP

from PIL import Image
from io import BytesIO
from packaging import version

filepath = os.path.abspath(__file__)
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from transparent_background.InSPyReNet import InSPyReNet_SwinB
from transparent_background.utils import *

class Remover:
    def __init__(self, mode="base", jit=False, device=None, ckpt=None, resize='static'):
        """
        Args:
            mode   (str): Choose among below options
                                   base -> slow & large gpu memory required, high quality results
                                   fast -> resize input into small size for fast computation
                                   base-nightly -> nightly release for base mode
            jit    (bool): use TorchScript for fast computation
            device (str, optional): specifying device for computation. find available GPU resource if not specified.
            ckpt   (str, optional): specifying model checkpoint. find downloaded checkpoint or try download if not specified.
            fast   (bool, optional, DEPRECATED): replaced by mode argument. use fast mode if True.
        """
        cfg_path = os.environ.get('TRANSPARENT_BACKGROUND_FILE_PATH', os.path.abspath(os.path.expanduser('~')))
        home_dir = os.path.join(cfg_path, ".transparent-background")
        os.makedirs(home_dir, exist_ok=True)

        if not os.path.isfile(os.path.join(home_dir, "config.yaml")):
            shutil.copy(os.path.join(repopath, "config.yaml"), os.path.join(home_dir, "config.yaml"))
        self.meta = load_config(os.path.join(home_dir, "config.yaml"))[mode]

        if device is not None:
            self.device = device
        else:
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda:0"
            elif (
                version.parse(torch.__version__) >= version.parse("1.13")
                and torch.backends.mps.is_available()
            ):
                self.device = "mps:0"

        download = False
        if ckpt is None:
            ckpt_dir = home_dir
            ckpt_name = self.meta.ckpt_name

            if not os.path.isfile(os.path.join(ckpt_dir, ckpt_name)):
                download = True
            elif (
                self.meta.md5
                != hashlib.md5(
                    open(os.path.join(ckpt_dir, ckpt_name), "rb").read()
                ).hexdigest()
            ):
                if self.meta.md5 is not None:
                    download = True

            if download:
                if 'drive.google.com' in self.meta.url:
                    gdown.download(self.meta.url, os.path.join(ckpt_dir, ckpt_name), fuzzy=True, proxy=self.meta.http_proxy)
                elif 'github.com' in self.meta.url:
                    wget.download(self.meta.url, os.path.join(ckpt_dir, ckpt_name))
                else:
                    raise NotImplementedError('Please use valid URL')
        else:
            ckpt_dir, ckpt_name = os.path.split(os.path.abspath(ckpt))

        self.model = InSPyReNet_SwinB(depth=64, pretrained=False, threshold=None, **self.meta)
        self.model.eval()
        self.model.load_state_dict(
            torch.load(os.path.join(ckpt_dir, ckpt_name), map_location="cpu", weights_only=True),
            strict=True,
        )
        self.model = self.model.to(self.device)

        if jit:
            ckpt_name = self.meta.ckpt_name.replace(
                ".pth", "_{}.pt".format(self.device)
            )
            try:
                traced_model = torch.jit.load(
                    os.path.join(ckpt_dir, ckpt_name), map_location=self.device
                )
                del self.model
                self.model = traced_model
            except:
                traced_model = torch.jit.trace(
                    self.model,
                    torch.rand(1, 3, *self.meta.base_size).to(self.device),
                    strict=True,
                )
                del self.model
                self.model = traced_model
                torch.jit.save(self.model, os.path.join(ckpt_dir, ckpt_name))
            if resize != 'static':
                warnings.warn('Resizing method for TorchScript mode only supports static resize. Fallback to static.')
                resize = 'static'

        resize_tf = None
        resize_fn = None
        if resize == 'static':
            resize_tf = static_resize(self.meta.base_size)
            resize_fn = A.Resize(*self.meta.base_size)
        elif resize == 'dynamic':
            if 'base' not in mode:
                warnings.warn('Dynamic resizing only supports base and base-nightly mode. It will cause severe performance degradation with fast mode.')
            resize_tf = dynamic_resize(L=1280)
            resize_fn = dynamic_resize_a(L=1280)
        else:
            raise AttributeError(f'Unsupported resizing method {resize}')

        self.transform = transforms.Compose(
            [
                resize_tf,
                tonumpy(),
                normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                totensor(),
            ]
        )

        self.cv2_transform = A.Compose(
            [
                resize_fn,
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                AP.ToTensorV2(),
            ]
        )

        self.background = {'img': None, 'name': None, 'shape': None}
        desc = "Mode={}, Device={}, Torchscript={}".format(
            mode, self.device, "enabled" if jit else "disabled"
        )
        print("Settings -> {}".format(desc))

    def process(self, img, type="rgba", threshold=None, reverse=False):
        """
        Args:
            img (PIL.Image or np.ndarray): input image as PIL.Image or np.ndarray type
            type (str): output type option as below.
                        'rgba' will generate RGBA output regarding saliency score as an alpha map. 
                        'green' will change the background with green screen.
                        'white' will change the background with white color.
                        '[255, 0, 0]' will change the background with color code [255, 0, 0]. 
                        'blur' will blur the background.
                        'overlay' will cover the salient object with translucent green color, and highlight the edges.
                        Another image file (e.g., 'samples/backgroud.png') will be used as a background, and the object will be overlapped on it.
            threshold (float or str, optional): produce hard prediction w.r.t specified threshold value (0.0 ~ 1.0)
        Returns:
            PIL.Image: output image

        """

        if isinstance(img, np.ndarray):
            is_numpy = True
            shape = img.shape[:2]
            x = self.cv2_transform(image=img)["image"]
        else:
            is_numpy = False
            shape = img.size[::-1]
            x = self.transform(img)

        x = x.unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            pred = self.model(x)

        pred = F.interpolate(pred, shape, mode="bilinear", align_corners=True)
        pred = pred.data.cpu()
        pred = pred.numpy().squeeze()

        if threshold is not None:
            pred = (pred > float(threshold)).astype(np.float64)
        if reverse:
            pred = 1 - pred

        img = np.array(img)

        if type.startswith("["):
            type = [int(i) for i in type[1:-1].split(",")]

        if type == "map":
            img = (np.stack([pred] * 3, axis=-1) * 255).astype(np.uint8)

        elif type == "rgba":
            if threshold is None:
                # pymatting is imported here to avoid the overhead in other cases.
                try:
                    from pymatting.foreground.estimate_foreground_ml_cupy import estimate_foreground_ml_cupy as estimate_foreground_ml
                except ImportError:
                    try:
                        from pymatting.foreground.estimate_foreground_ml_pyopencl import estimate_foreground_ml_pyopencl as estimate_foreground_ml
                    except ImportError:
                        from pymatting import estimate_foreground_ml
                img = estimate_foreground_ml(img / 255.0, pred)
                img = 255 * np.clip(img, 0., 1.) + 0.5
                img = img.astype(np.uint8)

            r, g, b = cv2.split(img)
            pred = (pred * 255).astype(np.uint8)
            img = cv2.merge([r, g, b, pred])

        elif type == "green":
            bg = np.stack([np.ones_like(pred)] * 3, axis=-1) * [120, 255, 155]
            img = img * pred[..., np.newaxis] + bg * (1 - pred[..., np.newaxis])

        elif type == "white":
            bg = np.stack([np.ones_like(pred)] * 3, axis=-1) * [255, 255, 255]
            img = img * pred[..., np.newaxis] + bg * (1 - pred[..., np.newaxis])

        elif len(type) == 3:
            bg = np.stack([np.ones_like(pred)] * 3, axis=-1) * type
            img = img * pred[..., np.newaxis] + bg * (1 - pred[..., np.newaxis])

        elif type == "blur":
            img = img * pred[..., np.newaxis] + cv2.GaussianBlur(img, (0, 0), 15) * (
                1 - pred[..., np.newaxis]
            )

        elif type == "overlay":
            bg = (
                np.stack([np.ones_like(pred)] * 3, axis=-1) * [120, 255, 155] + img
            ) // 2
            img = bg * pred[..., np.newaxis] + img * (1 - pred[..., np.newaxis])
            border = cv2.Canny(((pred > 0.5) * 255).astype(np.uint8), 50, 100)
            img[border != 0] = [120, 255, 155]

        elif type.lower().endswith(IMG_EXTS):
            if self.background['name'] != type:
                background_img = cv2.cvtColor(cv2.imread(type), cv2.COLOR_BGR2RGB)
                background_img = cv2.resize(background_img, img.shape[:2][::-1])
                
                self.background['img'] = background_img
                self.background['shape'] = img.shape[:2][::-1]
                self.background['name'] = type
            
            elif self.background['shape'] != img.shape[:2][::-1]:
                self.background['img'] = cv2.resize(self.background['img'], img.shape[:2][::-1])
                self.background['shape'] = img.shape[:2][::-1]

            img = img * pred[..., np.newaxis] + self.background['img'] * (
                1 - pred[..., np.newaxis]
            )

        if is_numpy:
            return img.astype(np.uint8)
        else:
            return Image.fromarray(img.astype(np.uint8))

def to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    base64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return base64_img

def entry_point(out_type, mode, device, ckpt, source, dest, jit, threshold, resize, save_format=None, reverse=False, flet_progress=None, flet_page=None, preview=None, preview_out=None, options=None):
    warnings.filterwarnings("ignore")

    remover = Remover(mode=mode, jit=jit, device=device, ckpt=ckpt, resize=resize)

    if source.isnumeric() is True:
        save_dir = None
        _format = "Webcam"
        if importlib.util.find_spec('pyvirtualcam') is not None:
            try:
                import pyvirtualcam
                vcam = pyvirtualcam.Camera(width=640, height=480, fps=30)
            except:
                vcam = None
        else:
            raise ImportError("pyvirtualcam not found. Install with \"pip install transparent-background[webcam]\"")

    elif os.path.isdir(source):
        save_dir = os.path.join(os.getcwd(), source.split(os.sep)[-1])
        _format = get_format(os.listdir(source))

    elif os.path.isfile(source):
        save_dir = os.getcwd()
        _format = get_format([source])

    else:
        raise FileNotFoundError("File or directory {} is invalid.".format(source))

    if out_type == "rgba" and _format == "Video":
        raise AttributeError("type 'rgba' cannot be applied to video input.")

    if dest is not None:
        save_dir = dest

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    loader = eval(_format + "Loader")(source)
    frame_progress = tqdm.tqdm(
        total=len(loader),
        position=1 if (_format == "Video" and len(loader) > 1) else 0,
        leave=False,
        bar_format="{desc:<15}{percentage:3.0f}%|{bar:50}{r_bar}",
    )
    sample_progress = (
        tqdm.tqdm(
            total=len(loader),
            desc="Total:",
            position=0,
            bar_format="{desc:<15}{percentage:3.0f}%|{bar:50}{r_bar}",
        )
        if (_format == "Video" and len(loader) > 1)
        else None
    )
    if flet_progress is not None:
        assert flet_page is not None
        flet_progress.value = 0
        flet_step = 1 / frame_progress.total

    writer = None

    for img, name in loader:
        filename, ext = os.path.splitext(name) if name is not None else None, None
        ext = ext[1:] if ext is not None else None
        ext = save_format if save_format is not None else ext
        frame_progress.set_description("{}".format(name))
        if out_type.lower().endswith(IMG_EXTS):
            outname = "{}_{}".format(
                filename,
                os.path.splitext(os.path.split(out_type)[-1])[0],
            )
        else:
            outname = "{}_{}".format(filename, out_type) if filename is not None else None

        if reverse:
            outname += '_reverse'

        if _format == "Video" and writer is None:
            writer = cv2.VideoWriter(
                os.path.join(save_dir, f"{outname}.{ext}"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                loader.fps,
                img.size,
            )
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
            frame_progress.refresh()
            frame_progress.reset()
            frame_progress.total = int(loader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if sample_progress is not None:
                sample_progress.update()

            if flet_progress is not None:
                assert flet_page is not None
                flet_progress.value = 0
                flet_step = 1 / frame_progress.total
                flet_progress.update()

        if _format == "Video" and img is None:
            if writer is not None:
                writer.release()
            writer = None
            continue

        out = remover.process(img, type=out_type, threshold=threshold, reverse=reverse)

        if _format == "Image":
            if out_type == "rgba" and ext.lower() != 'png':
                warnings.warn('Output format for rgba mode only supports png format. Fallback to png output.')
                ext = 'png'
            out.save(os.path.join(save_dir, f"{outname}.{ext}"))
        elif _format == "Video" and writer is not None:
            writer.write(cv2.cvtColor(np.array(out), cv2.COLOR_BGR2RGB))
        elif _format == "Webcam":
            if vcam is not None:
                vcam.send(np.array(out))
                vcam.sleep_until_next_frame()
            else:
                cv2.imshow(
                    "transparent-background", cv2.cvtColor(np.array(out), cv2.COLOR_BGR2RGB)
                )
        frame_progress.update()
        if flet_progress is not None:
            flet_progress.value += flet_step
            flet_progress.update()

            if out_type == 'rgba':
                o = np.array(out).astype(np.float64)
                o[:, :, :3] *= (o[:, :, -1:] / 255)
                out = Image.fromarray(o[:, :, :3].astype(np.uint8))

            preview.src_base64 = to_base64(img.resize((480, 300)).convert('RGB'))
            preview_out.src_base64 = to_base64(out.resize((480, 300)).convert('RGB'))
            preview.update()
            preview_out.update()

        if options is not None and options['abort']:
            break
        
    print("\nDone. Results are saved in {}".format(os.path.abspath(save_dir)))

def console():
    args = parse_args()
    entry_point(args.type, args.mode, args.device, args.ckpt, args.source, args.dest, args.jit, args.threshold, args.resize, args.format, args.reverse)
