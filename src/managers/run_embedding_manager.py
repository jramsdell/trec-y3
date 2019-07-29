#!/home/jsc57/anaconda3/bin/python3
from src.learning.embedding.elmo_sentence_embedder import ElmoVectorEmbedderRunner
from src.learning.trainers.special_trainer import SpecialTrainer
import os
import torch
from src.parsers.readers.jsonl_paragraph_reader import JSONLParagraphReader
from src.parsers.readers.outline_reader import OutlineReader
from src.parsers.readers.run_pid_retriever import RunPidRetriever
from src.parsers.tokenizers.sentence_tokenizer import SentenceTokenizer
import numpy as np


class RunEmbeddingManager(object):
    def __init__(self, page_loc, run_loc, json_loc, json_pmap_loc, y1_test_page_loc, y1_test_run_loc, y2_test_run_loc,
                 y3_train_run_loc, y3_test_run_loc):
        self.json_reader = JSONLParagraphReader(json_loc, json_pmap_loc)
        self.pid_retriever = RunPidRetriever(run_loc, max_depth=100)
        self.y1_test_pid_retriever = RunPidRetriever(y1_test_run_loc, max_depth=100)
        self.y2_test_pid_retriever = RunPidRetriever(y2_test_run_loc, max_depth=100)
        self.y3_train_pid_retriever = RunPidRetriever(y3_train_run_loc, max_depth=100)
        self.y3_test_pid_retriever = RunPidRetriever(y3_test_run_loc, max_depth=100)
        self.outline_reader = OutlineReader(page_loc)
        self.y1_test_outline_reader = OutlineReader(y1_test_page_loc)
        self.tokenizer = SentenceTokenizer()

    def run_embedding(self):
        embedder = ElmoVectorEmbedderRunner()
        pids = self.pid_retriever.get_unique_pids()
        pids = pids.union(self.y1_test_pid_retriever.get_unique_pids())
        pids = pids.union(self.y2_test_pid_retriever.get_unique_pids())
        pids = pids.union(self.y3_train_pid_retriever.get_unique_pids())
        pids = pids.union(self.y3_test_pid_retriever.get_unique_pids())

        keys = []
        texts = []

        for (key, contents) in self.json_reader.retrieve_by_ids(pids).items():
            if len(contents["text"]) <= 10:
                print("PROBLEM")
            keys.append(key)
            texts.append(contents["text"])

        texts = self.tokenizer.tokenize_documents(texts)
        embedded = embedder.map_function(texts)

        np.save("new_wubba2.npy", np.asarray(embedded))
        with open("new_key_map2.txt", "w") as f:
            f.write("\n".join(keys))

    def run_page_embedding(self):
        embedder = ElmoVectorEmbedderRunner()
        _, context_map = self.outline_reader.retrieve_ordinal_map()
        keys = []
        cmatrix = []
        for (key, contexts) in context_map.items():
            keys.append(key)
            embeddings = list(embedder.model.embed_sentences(contexts))
            vectors = []
            for embedding in embeddings:
                (f1, f2, f3) = embedding
                f1 = f1.mean(0)
                f2 = f2.mean(0)
                f3 = f3.mean(0)
                final = np.concatenate([f1, f2, f3], 0)
                final = np.expand_dims(final, 1)
                vectors.append(final)

            vectors = np.concatenate(vectors, 1)
            cmatrix.append(vectors.mean(1))


        np.save("page_context.npy", cmatrix)

        with open("page_context_key_map.txt", "w") as f:
            f.write("\n".join(keys))


    def run_training_construction(self):
        key_loc = "new_key_map.txt"
        key_map = {}
        with open(key_loc) as f:
            for (idx, key) in enumerate(f):
                key_map[key.rstrip()] = idx

        page_context_key_loc = "page_context_key_map.txt"
        page_key_map = {}
        with open(page_context_key_loc) as f:
            for (idx, key) in enumerate(f):
                page_key_map[key.rstrip()] = idx

        # ndarray = np.load("wubba.npy")
        ndarray = np.load("new_wubba.npy")
        page_context_ndarray = np.load("page_context.npy")
        ordinal_map, context_map = self.outline_reader.retrieve_ordinal_map()

        trainer = SpecialTrainer(
            pid_pmap=key_map,
            ndarray=ndarray,
            ordinal_map=ordinal_map,
            retrieved_pids=self.pid_retriever.pmap,
            page_key_map=page_key_map,
            page_context_ndarray=page_context_ndarray
        )

        trainer.do_train()










if __name__ == '__main__':
    torch.set_num_threads(30)
    jsonl_loc = "/home/jsc57/projects/context_summarization/y1_corpus.jsonl"
    pmap_loc = "/home/jsc57/projects/context_summarization/y1_corpus_pmap.txt"
    page_loc = "/home/jsc57/data/benchmark/benchmarkY1/benchmarkY1-train/train.pages.cbor"
    y1_test_page_loc = "/home/jsc57/data/benchmark/test/benchmarkY1/benchmarkY1-test/test.pages.cbor"
    # run_loc = "/mnt/grapes/share/car-input-runs/benchmarkY1train-lucene-runs-14-page/lucene-luceneindexlucene-v21-14-lucene-paragraph--paragraph-page--all-bm25-none--Text-english-k1000-20_20_20-benchmarkY1train.v201.cbor.outlines.run"
    run_loc = "/mnt/grapes/share/car-input-runs/benchmarkY1train-lucene-runs-14-page/lucene-luceneindexlucene-v21-14-lucene-paragraph--paragraph-page--title-bm25-none--Text-english-k1000-20_100_20-benchmarkY1train.v201.cbor.outlines.run"
    y1_test_loc = "/mnt/grapes/share/car-input-runs/benchmarkY1test-lucene-runs-14-page/lucene-luceneindexlucene-v21-14-lucene-paragraph--paragraph-page--title-bm25-none--Text-english-k1000-20_100_20-benchmarkY1test.v201.cbor.outlines.run"
    y2_test_loc = "/mnt/grapes/share/car-input-runs/benchmarkY2test-lucene-runs-14-page/lucene-luceneindexlucene-v21-14-lucene-paragraph--paragraph-page--title-bm25-none--Text-english-k1000-20_100_20-benchmarkY2test.v201.cbor.outlines.run"
    y3_train_loc = "/mnt/grapes/share/car-input-runs/benchmarkY3train-lucene-runs-14-page/lucene-luceneindexlucene-v21-14-lucene-paragraph--paragraph-page--title-bm25-none--Text-english-k1000-20_100_20-benchmarkY3train.v201.cbor.outlines.run"
    y3_test_loc = "/mnt/grapes/share/car-input-runs/benchmarkY3test-lucene-runs-14-page/lucene-luceneindexlucene-v21-14-lucene-paragraph--paragraph-page--title-bm25-none--Text-english-k1000-20_100_20-benchmarkY3test.v201.cbor.outlines.run"

    manager = RunEmbeddingManager(
        page_loc=page_loc,
        run_loc=run_loc,
        y1_test_run_loc=y1_test_loc,
        y1_test_page_loc=y1_test_page_loc,
        y2_test_run_loc=y2_test_loc,
        y3_train_run_loc=y3_train_loc,
        y3_test_run_loc=y3_test_loc,
        json_loc=jsonl_loc,
        json_pmap_loc=pmap_loc
    )

    manager.run_embedding()
    # manager.run_page_embedding()
    # manager.run_training_construction()
