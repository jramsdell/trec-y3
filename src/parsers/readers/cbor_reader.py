import json
import sys
from collections import OrderedDict
from typing import *

from trec_car.read_data import iter_paragraphs, Paragraph, ParaLink, ParaText, iter_annotations, iter_pages, iter_outlines


class CborReader(object):
    def __init__(self, loc):
        with open(loc, 'rb') as f:
            self.explore(f)

    def explore(self, f):
        for outline in iter_outlines(f):
            for sec in (outline.outline()):
                print(sec)



if __name__ == '__main__':
    test_loc = "/home/jsc57/data/y3/benchmarkY3test/benchmarkY3test.cbor-outlines.cbor"
    train_loc = "/home/jsc57/data/y3/benchmarkY3train/benchmarkY3train.cbor-outlines.cbor"
    CborReader(test_loc)
