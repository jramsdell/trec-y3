
from src.parsers.readers.run_ranking_parser import RunRankingParser


class RunPidRetriever(object):
    def __init__(self, run_loc, max_depth=20):
        self.max_depth = max_depth
        self.pmap = {}
        parsed = RunRankingParser(run_loc).run()[1]
        for (query,retrieved) in parsed.items():
            pids = [i.pid for i in retrieved[0:max_depth]]
            self.pmap[query] = pids


    def get_unique_pids(self):
        pid_set = set()
        for (query, pids) in self.pmap.items():
            for pid in pids:
                pid_set.add(pid)

        return pid_set





if __name__ == '__main__':
    loc = "/mnt/grapes/share/car-input-runs/benchmarkY1train-lucene-runs-14-page/lucene-luceneindexlucene-v21-14-lucene-paragraph--paragraph-page--all-bm25-none--Text-english-k1000-20_20_20-benchmarkY1train.v201.cbor.outlines.run"
    manager = RunPidRetriever(loc)
    print(manager.get_unique_pids())
