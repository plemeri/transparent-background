import cv2
import numpy as np

from PIL import Image
from transparent_background import Remover

# Load model
remover = Remover() # default setting
remover = Remover(mode='fast', jit=True, device='cuda:0', ckpt='~/latest.pth') # custom setting
remover = Remover(mode='base-nightly') # nightly release checkpoint

# Usage for image
img = Image.open('samples/aeroplane.jpg').convert('RGB') # read image

out = remover.process(img) # default setting - transparent background
out = remover.process(img, type='rgba') # same as above
out = remover.process(img, type='map') # object map only
out = remover.process(img, type='green') # image matting - green screen
out = remover.process(img, type='white') # change backround with white color
out = remover.process(img, type=[255, 0, 0]) # change background with color code [255, 0, 0]
out = remover.process(img, type='blur') # blur background
out = remover.process(img, type='overlay') # overlay object map onto the image
out = remover.process(img, type='samples/background.jpg') # use another image as a background

out = remover.process(img, threshold=0.5) # use threhold parameter for hard prediction.

out.save('output.png') # save result

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
    writer.write(cv2.cvtColor(np.array(out), cv2.COLOR_BGR2RGB))

cap.release()
writer.release()