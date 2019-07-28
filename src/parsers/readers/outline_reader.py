from typing import *
from typing import Dict

import trec_car
from textblob import TextBlob
from trec_car.read_data import iter_outlines, iter_pages

from src.parsers.readers.jsonl_paragraph_reader import JSONLParagraphReader


class OutlineReader(object):
    def __init__(self, outline_loc):
        self.outline_loc = outline_loc


    def retrieve_query_map(self) -> Dict[str, Tuple[str, str]]:
        with open(self.outline_loc, 'rb') as f:
            return self._parse(f)

    def _parse(self, f) -> Dict[str, Tuple[str, str]]:
        heading_map: Dict[str, Tuple[str, str]] = {}

        for page in iter_outlines(f):
            for nested in page.nested_headings():
                top_level = nested[0]
                key = page.page_id + "/" + top_level.headingId
                heading_map[key] = [page.page_name, top_level.heading]

        return heading_map

    def retrieve_ordinal_map(self):
        ordinal_map = {}
        context_map = {}
        with open(self.outline_loc, 'rb') as f:
            for page in iter_pages(f):
                contexts = self.retrieve_page_context(page)
                context_map[page.page_id] = contexts
                flattened = self.flatten(page.skeleton)
                ordinal_map[page.page_id] = dict([(i, idx) for idx, i in enumerate(flattened)])

        return ordinal_map, context_map

    def retrieve_page_context(self, page: trec_car.read_data.Page):
        contexts = []
        page_title = page.page_name
        page_words = list(TextBlob(page_title).words)
        contexts.append(page_words)
        for section in page.child_sections:
            context = page_title + " " + section.heading
            context_words = list(TextBlob(context).words)
            contexts.append(context_words)
        return contexts



    def flatten(self, remaining_list):
        if not remaining_list:
            return []

        next_node = remaining_list[0]
        children = []
        following_children = self.flatten(remaining_list[1:])

        if isinstance(next_node, trec_car.read_data.Section):
            children = self.flatten(next_node.children)
        elif isinstance(next_node, trec_car.read_data.Para):
            children = [next_node.paragraph.para_id]

        return children + following_children




if __name__ == '__main__':
    loc = "/home/jsc57/data/benchmark/benchmarkY1/benchmarkY1-train/train.pages.cbor"
    # with open(loc, "rb") as f:
    #     for outline in iter_outlines(f):
    #         print(outline.outline()[0].get_text())
    #         print(outline.page_meta)

    outline_reader = OutlineReader(loc)
    retrieved = outline_reader.retrieve_ordinal_map()

    jsonl_loc = "/home/jsc57/projects/context_summarization/y1_corpus.jsonl"
    pmap_loc = "/home/jsc57/projects/context_summarization/y1_corpus_pmap.txt"

    fst = list(list(retrieved.values())[0].keys())

    json_reader = JSONLParagraphReader(jsonl_loc, pmap_loc)
    fst = set(fst)
    wee = json_reader.retrieve_by_ids(fst)
    print(wee)


