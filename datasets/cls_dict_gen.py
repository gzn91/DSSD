import os
import xml.etree.ElementTree as ET
import pandas as pd

"""NÄR ALLA CLASSES FINNS REPRENSENTERADE I labels-mappen BÖR DENNA BYTAS SOM EN FÄRDIG DICT,
    SAMMA GÄLLER NUM CLS SOM DÅ BLIR FIXED"""



labels = {'background': 0, 'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7,
          'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
          'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}


def gen_labels():
    return labels


def num_cls():
    return max(labels.values()) + 1


def labels_to_name():
    d = dict()
    for key, value in labels.items():
        d[value] = key

    return d


if __name__ == '__main__':
    print(gen_labels())
    print(num_cls())
