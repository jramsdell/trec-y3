import os
from multiprocess.pool import Pool
from collections import defaultdict

class RunLine(object):
    query: str
    pid: str
    rank: int

    def __init__(self, line):
        elements = line.split(" ")
        self.query =  elements[0]
        self.pid = elements[2]
        self.rank = int(elements[3])


class RunRankingParser(object):
    def __init__(self, runfile_loc, filter_fun=lambda x: True):
        self.filter_fun = filter_fun
        self.runfile_loc = runfile_loc
        self.run_name = runfile_loc.split()[-1]


    def run(self):
        with open(self.runfile_loc) as f:
            return self.parse_run_file(f)

    def parse_run_file(self, f):
        qmap = defaultdict(list)
        for line in f:
            run_line = RunLine(line)
            if not self.filter_fun(run_line):
                continue

            qmap[run_line.query].append(run_line)

        return self.run_name, qmap


class RunDirParser(object):
    def __init__(self, run_dir, filter_fun=lambda x: True):
        self.filter_fun = filter_fun
        self.runners = []
        for run_file in os.listdir(run_dir):
            runfile_loc = run_dir + "/" + run_file
            self.runners.append(RunRankingParser(runfile_loc, filter_fun))


    def _run(self, runner):
        return runner.run()

    def run(self):
        pool = Pool(40)
        return dict(pool.imap(self._run, self.runners))



if __name__ == '__main__':
    # run_dir = "/home/jsc57/fixed_psg_runs"
    loc = "/mnt/grapes/share/car-input-runs/benchmarkY1train-lucene-runs-14-page/lucene-luceneindexlucene-v21-14-lucene-aspect--aspect-page--all-bm25-none--Text-english-k1000-20_100_20-benchmarkY1train.v201.cbor.outlines.run"
    parser = RunRankingParser(loc)
    qmap = parser.run()
    print(list(qmap[1].items())[0])
    # run_maps = parser.run()


