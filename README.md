# Transparent Background

<p align="center">
    <img src=https://raw.githubusercontent.com/plemeri/transparent-background/main/figures/logo.png width=200px>
</p>
<p align="center">
    <a href="https://github.com/plemeri/transparent-background/blob/main/LICENSE"><img  src="https://img.shields.io/badge/license-MIT-blue"></a>
    <a href="https://pypi.org/project/transparent-background/"><image src="https://badge.fury.io/py/transparent-background.svg"></a>
    <a href="https://pepy.tech/project/transparent-background"><image src="https://static.pepy.tech/personalized-badge/transparent-background?period=total&units=none&left_color=grey&right_color=orange&left_text=Downloads"></a>
</p>


This is a background removing tool powered by [InSPyReNet (ACCV 2022)](https://github.com/plemeri/InSPyReNet.git). You can easily remove background from the image or video or bunch of other stuffs when you can make the background transparent!

Image | Video | Webcam
:-:|:-:|:-:
<img src=https://raw.githubusercontent.com/plemeri/transparent-background/main/figures/demo_aeroplane.gif height=200px> | <img src=https://raw.githubusercontent.com/plemeri/transparent-background/main/figures/demo_b5.gif height=200px> | <img src=https://raw.githubusercontent.com/plemeri/transparent-background/main/figures/demo_webcam.gif height=200px>

## :newspaper: News

Our package is currently not working properly on small images without `--fast` argument. Sorry for the inconvenience and we'll fix this issue with better algorithm coming out shortly.

## :inbox_tray: Installation

### Dependencies (python packages)

package | version (>=)
:-|:-
`pytorch`       | `1.7.1`
`torchvision`   | `0.8.2`
`opencv-python` | `4.6.0.66`
`timm`          | `0.6.11`
`tqdm`          | `4.64.1`
`kornia`        | `0.5.4`
`gdown`         | `4.5.4`
`pyvirtualcam`  | `0.6.0`

Note: If you have any problem with [`pyvirtualcam`](https://pypi.org/project/pyvirtualcam/), please visit their github repository or pypi homepage. Due to the backend workflow for Windows and macOS, we only support Linux for webcam input.
### Dependencies (webcam input)

We basically follow the virtual camera settings from [`pyvirtualcam`](https://pypi.org/project/pyvirtualcam/). If you do not choose to install virtual camera, it will visualize real-time output with `cv2.imshow`.

#### A. Linux (v4l2loopback)

```bash
# Install v4l2loopback for webcam relay
$ git clone https://github.com/umlaeute/v4l2loopback.git && cd v4l2loopback
$ make && sudo make install
$ sudo depmod -a

# Create virtual webcam
$ sudo modprobe v4l2loopback devices=1
```

Note: If you have any problem with installing [`v4l2loopback`](https://github.com/umlaeute/v4l2loopback), please visit their github repository.

#### B. Windows (OBS)

Install OBS virtual camera from [install OBS](https://obsproject.com/).

#### C. macOS (OBS) [not stable]

Follow the steps below.
* [Install OBS](https://obsproject.com/).
* Start OBS.
* Click "Start Virtual Camera" (bottom right), then "Stop Virtual Camera".
* Close OBS.

### Install `transperent-background`
```bash
# via pypi
$ pip install transparent-background

# via github
$ pip install git+https://github.com/plemeri/transparent-background.git

# locally
$ pip install .
```

## :pencil2: Usage

### :computer: Command Line

```bash
# for apple silicon mps backend, use "PYTORCH_ENABLE_MPS_FALLBACK=1" before the command (requires torch >= 1.13)
$ transparent-background --source [SOURCE] --dest [DEST] --type [TYPE] --ckpt [CKPT] (--fast) (--jit)
```
* `--source [SOURCE]`: Specify your data in this argument.
    * Single image - `image.png`
    * Folder containing images - `path/to/img/folder`
    * Single video - `video.mp4`
    * Folder containing videos - `path/to/vid/folder`
    * Integer for webcam address - `0` (e.g., if your webcam is at `/dev/video0`.)
* `--dest [DEST]` (optional): Specify your destination folder. Default location is current directory.
* `--type [TYPE]` (optional): Choose between `rgba`, `map` `green`, `blur`, `overlay`, and another image file. Default is `rgba`.
    * `rgba` will generate RGBA output regarding saliency score as an alpha map. Note that this will not work for video and webcam input. 
    * `map` will output saliency map only. 
    * `green` will change the background with green screen. 
    * `white` will change the background with white color. -> [2023.05.24] Contributed by [carpedm20](https://github.com/carpedm20) 
    * `'[255, 0, 0]'` will change the background with color code [255, 0, 0]. Please use with single quotes. -> [2023.05.24] Contributed by [carpedm20](https://github.com/carpedm20) 
    * `blur` will blur the background.
    * `overlay` will cover the salient object with translucent green color, and highlight the edges.
    * Another image file (e.g., `samples/backgroud.png`) will be used as a background, and the object will be overlapped on it.
* `--ckpt [CKPT]` (optional): Use other checkpoint file. Default is trained with composite dataset and will be automatically downloaded if not available. Please refer to [Model Zoo](https://github.com/plemeri/InSPyReNet/blob/main/docs/model_zoo.md) from [InSPyReNet](https://github.com/plemeri/InSPyReNet) for available pre-trained checkpoints.
* `--fast` (optional): Fast mode. If specified, it will use low-resolution input and model trained with LR scale. May decrease performance but reduces inference time and gpu memory usage. 
* `--jit` (optional): Torchscript mode. If specified, it will trace model with pytorch built-in torchscript JIT compiler. May cause delay in initialization, but reduces inference time and gpu memory usage.
    
### :crystal_ball: Python API
* Usage Example
```python
import cv2

from PIL import Image
from transparent_background import Remover

# Load model
remover = Remover() # default setting
remover = Remover(fast=True, jit=True, device='cuda:0', ckpt='~/latest.pth') # custom setting

# Usage for image
img = Image.open('samples/aeroplane.jpg').convert('RGB') # read image

out = remover.process(img) # default setting - transparent background
out = remover.process(img, type='rgba') # same as above
out = remover.process(img, type='map') # object map only
out = remover.process(img, type='green') # image matting - green screen
out = remover.process(img, type='white') # change backround with white color -> [2023.05.24] Contributed by carpedm20
out = remover.process(img, type=[255, 0, 0]) # change background with color code [255, 0, 0] -> [2023.05.24] Contributed by carpedm20
out = remover.process(img, type='blur') # blur background
out = remover.process(img, type='overlay') # overlay object map onto the image
out = remover.process(img, type='samples/background.jpg') # use another image as a background

Image.fromarray(out).save('output.png') # save result

# Usage for video
cap = cv2.VideoCapture('samples/b5.mp4') # video reader for input
fps = cap.get(cv2.CAP_PROP_FPS)

writer = None

while cap.isOpened():
    ret, frame = cap.read() # read video

    if ret is False:
        break
        
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    img = Image.fromarray(frame).convert('RGB')

    if writer is None:
        writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, img.size) # video writer for output

    out = remover.process(img, type='map') # same as image, except for 'rgba' which is not for video.
    writer.write(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

cap.release()
writer.release()
```

## :tv: Tutorial

[rsreetech](https://github.com/rsreetech) shared a tutorial using colab. [[Youtube](https://www.youtube.com/watch?v=jKuQEnKmv4A)]

## :outbox_tray: Uninstall

```
pip uninstall transparent-background
```

## :page_facing_up: Licence

See [LICENCE](https://github.com/plemeri/transparent-background/blob/main/LICENSE) for more details.
