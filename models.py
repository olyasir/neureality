import torchvision.models as models
from patchedModels import GeneralizedRCNN_patched, FasterRCNN_patched
import torchvision
import torch.autograd.profiler as profiler
import torch


def doWrapping(model, name):
    '''
    :param model:
    :param name:
    :return:
    '''
    actualForward = model.forward
    def wrappedForward(*args, **kwargs):
        with profiler.record_function(name):
            return actualForward(*args, **kwargs)
    model.forward = wrappedForward


def wrapForward(model, name):
    '''
    This function recursevily adds profiling to every module in given model
    :param model:
    :param name: the name of the model
    :return: model with wrapped forwards
    '''
    hasChildren = False
    for ch_name, module in model.named_children():
        if isinstance(module, torch.nn.Module):
            hasChildren = True
            wrapForward(module, name+"."+ch_name+"_"+module._get_name())
    if not hasChildren:
        doWrapping(model, name)
    return model




def getModel(model_str):
    if model_str == "faster-rcnn" :
        torchvision.models.detection.generalized_rcnn.GeneralizedRCNN = GeneralizedRCNN_patched
        torchvision.models.detection.faster_rcnn.FasterRCNN = FasterRCNN_patched
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    elif model_str == 'mask-rcnn':
        model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    else:
        raise ValueError(" unsupported model type {}".format(model_str))

    wrapForward(model, "NR_"+model._get_name())
    return model



