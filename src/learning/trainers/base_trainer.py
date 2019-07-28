import torch
from torch.utils.data import TensorDataset

from learning.trainers.data_handler import DataHandler


class BaseTrainer(object):
    def __init__(self, data_handler: DataHandler, model, loss_function, lr=0.001):
        self.data_handler = data_handler
        self.model = model(data_handler)
        self.optimizer = None
        self.lr = lr
        self.loss_function = loss_function
        self.dataset = None
        self.key_map = None


    def tensorfy(self):
        counter = 0
        key_map = {}
        dataset = []
        dataset.append(self.data_handler.response)


        for idx, (k,v) in enumerate(self.data_handler.predictors.items()):
            key_map[k] = idx
            print(v.shape)
            dataset.append(v)
        dataset = TensorDataset(*dataset)

        return dataset, key_map


    def train(self, n=3000, report=True, weight_decay=0.005):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        # self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        # self.optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        # self.step_size = n / 10
        self.step_size = 50
        self.counter = 0

        orig_response = self.data_handler.response
        orig_predictors = self.data_handler.predictors

        if self.dataset is None:
            self.dataset, self.key_map = self.tensorfy()

        # rs = RandomSampler(dataset, replacement=True, num_samples=400)
        # dloader = DataLoader(batch_size=400, dataset=dataset, sampler=rs)


        def closure():
            self.optimizer.zero_grad()
            y_pred = self.model(self.data_handler)
            loss = self.loss_function(y_pred, self.data_handler.response)
            self.counter += 1

            if report and self.counter % self.step_size == 0:
                print(loss)
            loss.backward()
            return loss

        # for i in range(n):
        #     dloader = DataLoader(batch_size=800, dataset=self.dataset, shuffle=True)
        #     for datas in dloader:
        #         self.model.train()
        #
        #         # if i <= 200:
        #         self.data_handler.response = datas[0]
        #         predictors = datas[1:]
        #         for (k,v) in self.key_map.items():
        #             self.data_handler.predictors[k] = predictors[v]
        #
        #
        #
        #         self.optimizer.step(closure)


        self.data_handler.predictors = orig_predictors
        self.data_handler.response = orig_response






        for epoch in range(n):
            self.model.train()
            self.optimizer.step(closure)
