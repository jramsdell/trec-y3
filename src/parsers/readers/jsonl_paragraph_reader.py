import json
from typing import *


class JSONLParagraphReader(object):
    def __init__(self, loc, index_loc):
        self.loc = loc
        self.index_loc = index_loc


    def retrieve_by_ids(self, ids: Set[str]):
        # indices = sorted([self.pmap[id] for id in ids], key=lambda x: x[0])
        indices = self.get_byte_offsets(ids)
        jsons = {}

        with open(self.loc, 'r') as f:
            for (start, stop) in indices:
                f.seek(start)
                retrieved = json.loads(f.read(stop - start))
                jsons[retrieved["pid"]] = retrieved
                # jsons.append(retrieved)

        return jsons


    def get_byte_offsets(self, ids: Set[str]):
        offsets = []
        with open(self.index_loc, 'r') as f:
            for line in f:
                pid, start, stop = line.split(" ")
                if pid in ids:
                    offsets.append([int(start), int(stop)])
        return sorted(offsets, key=lambda x: x[0])




if __name__ == '__main__':
    jsonl_loc = "/home/jsc57/projects/context_summarization/y2_test.jsonl"
    pmap_loc = "/home/jsc57/projects/context_summarization/y2_test_pmap.txt"
    preader = JSONLParagraphReader(jsonl_loc, pmap_loc)
    ktest = []

