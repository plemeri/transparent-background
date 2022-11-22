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
