# Transparent Background

This is a background removing tool based on [InSPyReNet (ACCV 2022)](https://github.com/plemeri/InSPyReNet.git).

<p>
    <img src=figures/demo_type.png >
</p>

## Install

###
```
# via pypi
pip install transparent-background

# locally
pip install . e
```

## Usage

### Command Line

```
transparent-background --source [SOURCE] --dest [DEST] --type [TYPE]
```
* `--source [SOURCE]`: Specify your data in this argument.
    * Single image - `image.png`
    * Folder containing images - `path/to/img/folder`
    * Single video - `video.mp4`
    * Folder containing videos - `path/to/vid/folder`
* `--dest [DEST]` (optional): Specify your destination folder. If not specified, it will be saved in current directory.
* `--type [TYPE]`: Choose between `map` `green`, `rgba`, `blur`, `overlay`, and another image file.
    * `rgba` will generate RGBA output regarding saliency score as an alpha map. Note that this will not work for video input. 
    * `map` will output saliency map only. 
    * `green` will change the background with green screen. 
    * `blur` will blur the background.
    * `overlay` will cover the salient object with translucent green color, and highlight the edges.
    * Another image file (e.g., `backgroud.png`) will be used as a background, and the object will be overlapped on it.
    
### Python API
* Usage Example
```python
from PIL import Image
from transparent_background import Remover

remover = Remover()
img = Image.open('figures/sample.jpg')

# default setting - transparent background
out = remover.process(img)
Image.fromarray(out).save('output.png')

# object map only
out = remover.process(img, type='map')
Image.fromarray(out).save('output.png')

# image matting - green screen
out = remover.process(img, type='green')
Image.fromarray(out).save('output.png')

```