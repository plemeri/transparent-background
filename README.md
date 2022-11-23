# Transparent Background

<p align="center">
    <img src=https://github.com/plemeri/transparent-background/blob/main/figures/logo.png width=200px>
</p>
<p align="center">
    <a href="https://github.com/plemeri/transparent-background/blob/main/LICENSE"><img  src="https://img.shields.io/badge/license-MIT-blue"></a>
    <a href="https://pypi.org/project/transparent-background/"><image src="https://badge.fury.io/py/transparent-background.svg"></a>
</p>

This is a background removing tool powered by [InSPyReNet (ACCV 2022)](https://github.com/plemeri/InSPyReNet.git). You can easily remove background from the image or video or bunch of other stuffs when you can make the background transparent!


<p align="center">
    <img src=https://github.com/plemeri/transparent-background/blob/main/figures/demo_aeroplane.gif >
</p>

## :inbox_tray: Install

```
# via pypi
pip install transparent-background

# via github
pip install git+https://github.com/plemeri/transparent-background.git

# locally
pip install . e
```

## :pencil2: Usage

### :computer: Command Line

```
transparent-background --source [SOURCE] --dest [DEST] --type [TYPE]
```
* `--source [SOURCE]`: Specify your data in this argument.
    * Single image - `image.png`
    * Folder containing images - `path/to/img/folder`
    * Single video - `video.mp4`
    * Folder containing videos - `path/to/vid/folder`
* `--dest [DEST]` (optional): Specify your destination folder. If not specified, it will be saved in current directory.
* `--type [TYPE]`: Choose between `rgba`, `map` `green`, `blur`, `overlay`, and another image file.
    * `rgba` will generate RGBA output regarding saliency score as an alpha map. Note that this will not work for video input. 
    * `map` will output saliency map only. 
    * `green` will change the background with green screen. 
    * `blur` will blur the background.
    * `overlay` will cover the salient object with translucent green color, and highlight the edges.
    * Another image file (e.g., `backgroud.png`) will be used as a background, and the object will be overlapped on it.

<p>
    <img src=https://github.com/plemeri/transparent-background/blob/main/figures/demo_type.png >
</p>
    
### :crystal_ball: Python API
* Usage Example
```python
from PIL import Image
from transparent_background import Remover

remover = Remover()
img = Image.open('samples/aeroplane.jpg')

# default setting - transparent background
out = remover.process(img)
Image.fromarray(out).save('samples/aeroplane_rgba.png')

# object map only
out = remover.process(img, type='map')
Image.fromarray(out).save('samples/aeroplane_map.png')

# image matting - green screen
out = remover.process(img, type='green')
Image.fromarray(out).save('samples/aeroplane_green.png')

# blur background
out = remover.process(img, type='blur')
Image.fromarray(out).save('samples/aeroplane_blur.png')

# overlay object map onto the image
out = remover.process(img, type='overlay')
Image.fromarray(out).save('samples/aeroplane_overlay.png')

# use another image as a background
out = remover.process(img, type='samples/sheep.jpg')
Image.fromarray(out).save('samples/aeroplane_png.png')
```


## :outbox_tray: Uninstall

```
pip uninstall transparent-background
```

## :page_facing_up: Licence

See [LICENCE](https://github.com/plemeri/transparent-background/blob/main/LICENSE) for more details.