def _coco_labels():
    labels = open("./labels/coco")
    allLabels = labels.readlines()
    allLabels = [i.strip() for i in allLabels]
    return allLabels

def get_labels(dataset):
    '''
    get labels given a dataset
    :param dataset:  dataset
    :return: list of labels
    '''
    if dataset == 'coco':
        return _coco_labels()
    else:
        raise NotImplementedError("{} is not supported".format(dataset))



if __name__ == "__main__":
    _coco_labels()