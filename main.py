import argparse
from models import getModel
from  PIL import ImageDraw, Image, ImageFont
from torch.utils.tensorboard import SummaryWriter
import torch.autograd.profiler as profiler
import numpy as np
from labels.labels import get_labels
from torchvision import  transforms as T

writer = SummaryWriter()

import random

parser = argparse.ArgumentParser(
    description='CPU profilter With Pytorch')

parser.add_argument("--model", default="faster-rcnn", type=str,
                    help='model to profile. supported models : faster-rcnn, mask-rcnn')

parser.add_argument("--image", default="./data/cat.jpeg", type=str,
                    help='model to profile.')

args = parser.parse_args()

PROFILER_ROWS_TO_SHOW = 1000
THRESHHOLD = 0.7



def drawBoxes(im, predictions, dataset = 'coco'):
    '''
    draw bounding boxes
    :param im: image to draw on
    :param predictions: predictions of the model
    :param dataset: dataset to use for labels
    :return:
    '''

    draw = ImageDraw.Draw(im)
    i=0
    for score, box, label in zip( predictions[0]['scores'], predictions[0]['boxes'], predictions[0]['labels']):

        if score > THRESHHOLD:
            box = box.detach().numpy().tolist()
            draw.rectangle(box, outline = 'red')
            txt = get_labels(dataset)[label-1]
            draw.text( [box[0], box[1]], txt )

            if 'masks' in predictions[0]:
                mask = predictions[0]['masks'][i]
                color = [random.choice(range(255)), random.choice(range(255)), random.choice(range(255))]
                im = np.array(im)
                im = apply_mask(im, mask[0,:,:].detach().numpy(), color)
                im = Image.fromarray(im)


            #im.show()
        i+=1
    return im


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask > 0.8,
                                  image[:, :, c] * (1 - alpha) +
                                  alpha * color[c] ,
                                  image[:, :, c])
    return image


def addProfileData(prof, writer):
    '''
    add profiling data to tensorboard as an image
    :param prof:
    :param writer:
    :return:
    '''
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=40))

    out = Image.new("RGB", (2000, 3000), (255, 255, 255))
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 10)
    d = ImageDraw.Draw(out)
    # draw multiline text
    keyev=prof.key_averages()

    #we pick only profiling events that we are interested in
    toRem = []
    for e in keyev:
        if not e.key.startswith("NR_") and  not e.key.startswith("model"):
            toRem.append(e)
    for e in toRem:
        keyev.remove(e)

    #there is a bug in profiler , if we add sort by, it will repopulate events but we want only nr_ events
    d.multiline_text((10, 10), keyev.table(  row_limit=PROFILER_ROWS_TO_SHOW, sort_by="cpu_time_total",), font=fnt, fill=(0, 0, 0))
    writer.add_image("Profile data CPU", T.ToTensor()(out))
    print(keyev.table(  row_limit=PROFILER_ROWS_TO_SHOW, sort_by="cpu_time_total"))

    #have to upgrade to newer version ot torch to use profile memory functionality
    PROFILE_MEMORY = False
    if PROFILE_MEMORY:
        out1 = Image.new("RGB", (2000, 3000), (255, 255, 255))
        d1 = ImageDraw.Draw(out1)
        d1.multiline_text((10, 10), keyev.table(row_limit=PROFILER_ROWS_TO_SHOW, sort_by="self_cpu_memory_usage" ), font=fnt, fill=(0, 0, 0))
        writer.add_image("Profile data Memory", T.ToTensor()(out1))
        print(keyev.table(row_limit=PROFILER_ROWS_TO_SHOW, sort_by="self_cpu_memory_usage"))


def main():
    import torchvision
    model = getModel(args.model)
    model.eval()
    imageName = args.image
    im = Image.open(imageName)
    x =   np.array( im)
    x = T.ToTensor()(x)
    predictions = model([x])

    #display_instances(np.array(im), boxes = predictions[0]['boxes'][:1], masks = predictions[0]['masks'][:1].detach().numpy().transpose( [1,2,3,0])[0,:,:,:], class_ids= predictions[0]['labels'][:1], class_names = get_labels("coco"))

    im = drawBoxes(im, predictions)

    writer.add_image(imageName, T.ToTensor()(im))
    #have to upgrade to newer version of torch to use profile memory
    with profiler.profile(record_shapes=True) as prof:
    #with profiler.profile(record_shapes=True, profile_memory=True ) as prof:
        with profiler.record_function("model_inference"):
            model([x])

    addProfileData(prof, writer)
    writer.close()
    #load this trace in chrome
    prof.export_chrome_trace("trace.json")



if __name__ == "__main__":
    main()