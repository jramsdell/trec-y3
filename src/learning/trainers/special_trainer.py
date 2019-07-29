import numpy as np
import torch
from torch.autograd import Variable
from random import shuffle
import os

from typing import List

from src.managers.beam_searcher import BeamChoiceFunction, BeamState, BeamNode

class LSTMState(BeamState):
    def __init__(self, internal_hidden_state, internal_state, score_state):
        super(LSTMState, self).__init__()
        self.internal_hidden_state = internal_hidden_state
        self.internal_state = internal_state
        self.sequence_state = score_state


class SpecialTrainer(torch.nn.Module):
    def __init__(self, pid_pmap, retrieved_pids, ordinal_map, ndarray, page_key_map, page_context_ndarray):
        super(SpecialTrainer, self).__init__()
        self.hidden_size = 100 * 2
        self.n_paragraphs = 100
        self.n_features = 1024
        self.n_context_hidden = 1
        self.n_seq_hidden = 1

        self.pid_map = pid_pmap
        self.retrieved_pids = retrieved_pids
        self.ordinal_map =  ordinal_map
        self.ndarray = ndarray
        self.page_context_keymap = page_key_map
        self.page_context_ndarray = page_context_ndarray
        self.tanh = torch.nn.Hardtanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=0)
        self.loss = torch.nn.BCELoss()
        self.bm25_maps = []
        self.predicted_maps = []
        # torch.nn.LSTM()
        self.dropout = torch.nn.Dropout(0.0)


        self.passage_context_lstm = torch.nn.LSTM(
            input_size=self.n_features * 3,
            hidden_size=self.hidden_size,
            num_layers=self.n_context_hidden,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )

        self.sequence_generator = torch.nn.LSTM(
            input_size=self.hidden_size * 2,
            hidden_size=self.hidden_size * 2,
            num_layers=self.n_seq_hidden,
            batch_first=True,
            bidirectional=False
        )

        # self.state_map = torch.nn.Linear(self.hidden_size, self.hidden_size * self.hidden_size)
        # self.page_context_map = torch.nn.Linear(self.n_features * 3, self.hidden_size * self.hidden_size)

        # self.reset_lever = torch.nn.Linear(self.hidden_size * 2, 1)
        # self.reset_lever2 = torch.nn.Linear(self.hidden_size * 2, 1)
        self.reset_lever = torch.nn.Linear(self.n_features * 3, 400)
        self.reset_lever2 = torch.nn.Linear(self.hidden_size * 2, self.hidden_size * 2)


        self.state_map = torch.nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self.page_context_map = torch.nn.Linear(self.n_features * 3, self.hidden_size * 2)
        self.context_converter = torch.nn.Linear(self.n_context_hidden * 2, 1)

        self.attention_weights = torch.nn.Linear(self.hidden_size * 2, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0004)


    def get_map(self, rankings, relevants):
        denom = 0.0
        numerator = 0.0
        total = 0.0
        for r in rankings:
            denom += 1.0
            if r in relevants:
                numerator += 1.0
                total += numerator / denom
        return total / len(relevants)

    def do_train(self, **kwargs):
        prevs = []
        counter = 0
        for _ in range(1):
            for query in sorted(self.ordinal_map.keys()):
                retrieved = self.retrieved_pids[query]
                ret_mat = np.asarray([self.ndarray[self.pid_map[i]] for i in retrieved])
                has_nan = torch.isnan(torch.Tensor(ret_mat)).any()
                n_count = len(set(retrieved).intersection(set(self.ordinal_map[query].keys())))
                if n_count <= 0 or has_nan:
                    continue

                prevs.append(query)

                print("{}: {}".format(query, n_count))
                ordinals = self.ordinal_map[query]
                relevant = set([i for i in retrieved if i in ordinals])
                map = self.get_map(retrieved, relevant)
                self.bm25_maps.append(map)
                # if len(self.bm25_maps) > 10:
                #     self.bm25_maps = self.bm25_maps[1:]
                print("Original MAP: {}".format(map))
                counter += 1

                for i in range(5):
                    self.optimizer.zero_grad()
                    loss = self.forward(query, first=i == 0)
                    # print(loss)
                    loss.backward()
                    self.optimizer.step()



                av1 = sum(self.bm25_maps) / len(self.bm25_maps)
                av2 = sum(self.predicted_maps) / len(self.predicted_maps)
                print("{} / {}".format(av1, av2))

                if counter > 20:
                    print("Starting prev training")
                    shuffle(prevs)
                    for prev in prevs[0:5]:
                        self.optimizer.zero_grad()
                        loss = self.forward(prev, first=False)
                        # print(loss)
                        loss.backward()
                        self.optimizer.step()





                # out = ""
                # for pid in retrieved:
                #     if pid in ordinals:
                #         out += (" [{}]".format(ordinals[pid]))
                #     else:
                #         out += " ."
                # out += "    [Original]"
                # print(out)



    def run_transform(self, state, mean_context, page_vector=None, seen=None):
        # transform_matrix = self.state_map(state).reshape((self.hidden_size, self.hidden_size))
        # context_matrix = self.page_context_map(page_vector).reshape((self.hidden_size, self.hidden_size))
        # result = mean_context @ (self.tanh(transform_matrix) * self.tanh(context_matrix))

        s1 = self.dropout(state)
        s2 = self.dropout(page_vector)
        # transform_matrix = self.state_map(state)
        transform_matrix = self.state_map(s1)
        # context_matrix = self.page_context_map(page_vector)
        context_matrix = self.page_context_map(s2)
        result = mean_context * (self.tanh(transform_matrix)) * (self.tanh(context_matrix))

        result = self.attention_weights(result).squeeze()
        # result = self.softmax(result).detach().numpy()
        mask = torch.ones(result.shape)
        for s in seen:
            mask[s] = 0
        result = result * mask
        result = self.softmax(result)
        sorted_args = torch.argsort(result, 0, descending=True)
        return sorted_args, result

    def run_transform2(self, state, cmean):
        transform_matrix = self.state_map(state).reshape((self.hidden_size, self.hidden_size))

        result = cmean @ transform_matrix
        result = self.attention_weights(result).squeeze()
        # result = self.softmax(result).detach().numpy()
        result = self.softmax(result)
        return result

    def get_initial_state(self, ret_mat, page_vector):
        context_out, (hidden, state) = self.passage_context_lstm(ret_mat)
        # hidden = torch.cat([hidden[0], hidden[1]], 0)
        # state = torch.cat([state[0], state[1]], 0)

        chosen = context_out.mean(0).unsqueeze(0)
        cmean = context_out.mean(1)

        # hmean = self.context_converter(hidden.permute(2, 1, 0))
        # smean = self.context_converter(state.permute(2, 1, 0))

        hmean = hidden.mean(1).unsqueeze(1)
        smean = state.mean(1).unsqueeze(1)

        # hmean = hidden.mean(1).unsqueeze(0)[:, -1, :].unsqueeze(0)
        # smean = state.mean(1).unsqueeze(0)[:, -1, :].unsqueeze(0)

        hmean = torch.cat([hmean[0], hmean[1]], 1).unsqueeze(0)
        smean = torch.cat([smean[0], smean[1]], 1).unsqueeze(0)


        # print(hmean.shape)
        # print(hidden.shape)

        seq_out, (hidden, state) = self.sequence_generator(chosen, (hmean, smean))
        return cmean, context_out, seq_out, hidden, state

    def run_prediction_step(self, state, mean_context, seen, page_vector):
        sorted_args, values = self.run_transform(state, mean_context, page_vector, seen)
        for (arg, value)  in zip(sorted_args, values):
            arg = int(arg)
            if arg not in seen:
                seen.add(arg)
                return arg, values, sorted_args

        return [None, None, sorted_args]


    def create_beam_choice_function(self, mean_context):
        rt = self.run_transform

        class CustomBeamChoiceFunction(BeamChoiceFunction):
            def choose(self, node: BeamNode, max_candidates=3) -> List[BeamNode]:

                # Given current state, what kinds of choices (passages to add to our current ordering) can we make?
                state: LSTMState = node.state
                descending_choices, unsorted_scores = rt(state.internal_state, mean_context)

                # A passage can only be added once to the ordering, so remove those that have already been added
                valid_choices = [choice for choice in descending_choices
                                 if choice not in node.all_choices]

                if not valid_choices:
                    return []  # no choices left to make, we're done

                descending_choices = descending_choices[0:max_candidates]

                nodes = []
                for choice in descending_choices:
                    probability_score = unsorted_scores[choice] # scores are indexed by passage index
                    probability_score = probability_score.reshape((1))
                    new_score_vector = torch.cat([state.score_vector, probability_score], 0)

                    # seq_out, (hidden, state) = self.sequence_generator(seq_out, (hidden, state))



                pass



    def run_prediction_step2(self, state, cmean, seen):
        result = self.run_transform(state, cmean)
        return result




    def construct_labels(self, args, ordinals, retrieved_pids):
        prev = -2
        labels = []

        ord_count = 0
        for arg in args:
            pid = retrieved_pids[arg]
            if pid in args:
                ord_count += 1

        cur_count = 0
        last_good = -1

        dist = 0
        for arg in args:
            pid = retrieved_pids[arg]


            if pid in ordinals:
                labels.append(1.0)
            else:
                labels.append(0.0)



            # if pid in ordinals:
            #     cur_count += 1
            #     pos = ordinals[pid]
            #     if prev == -2:
            #         labels.append(1.0 / (dist + 1))
            #     elif prev < pos and prev != -1:
            #         labels.append((2.0) / (dist + 1))
            #     else:
            #         labels.append(0.5)
            #     prev = pos
            #     dist = 0
            # else:
            #     dist += 1
            #     if cur_count >= ord_count:
            #         labels.append(0.0)
            #     else:
            #         labels.append(0.0)
            #     # prev = -1

        return torch.Tensor(labels)



    def forward(self, query, first=False):
        retrieved = self.retrieved_pids[query]
        ordinals = self.ordinal_map[query]
        seen = set()

        ret_mat = np.asarray([self.ndarray[self.pid_map[i]] for i in retrieved])
        ret_mat = torch.Tensor(ret_mat)
        page_vector = self.page_context_ndarray[self.page_context_keymap[query]]
        page_vector = Variable(torch.Tensor(page_vector))
        page_vector = page_vector.unsqueeze(0)
        ret_mat = Variable(ret_mat)

        cmean, context_out, seq_out, hidden, state = self.get_initial_state(ret_mat, page_vector)
        inside = len([i for i in retrieved if i in ordinals])




        args = []
        values = None


        counter = 0
        while True:

            # Predict
            arg, vals, sorted_args = self.run_prediction_step(state, cmean, seen, page_vector)

            # args = list(sorted_args.detach().numpy())
            # values = vals
            # values = values[sorted_args]
            # break

            if arg is None:
                break

            value = (vals[arg])

            # Update results
            args.append(arg)
            if values is None:
                values = value.reshape((1))
            else:
                values = torch.cat([values, value.reshape((1))], 0)
            # values.append(value)

            # Transform input for next state based on choice
            # seq_out = torch.cat([seq_out, context_out[arg].unsqueeze(0)], 1)

            # seq_out = torch.cat([seq_out, context_out[arg].unsqueeze(0).mean(1).unsqueeze(0)], 1)

            counter += 1
            # if counter >= 20:
            #     break

            # seq_out = torch.cat([seq_out[:, 0:(4 + counter), :], context_out[arg].unsqueeze(0)], 1)


            seq_out = torch.cat([seq_out, context_out[arg].unsqueeze(0)], 1)
            reset_page = self.tanh(self.reset_lever(page_vector))
            reset_context = self.tanh(self.reset_lever2(context_out[arg]))
            # print(reset_context.shape)
            reset_context = reset_context.mean(0).unsqueeze(0)

            reset_score = self.sigmoid((reset_page * reset_context).sum())

            # reset_score = self.sigmoid(self.reset_lever(hidden).sum())
            # reset_score2 = self.sigmoid(self.reset_lever2(state).sum())

            # if reset_score * reset_score2 <= 0.9:
            if reset_score <= 0.95:
                seq_out = seq_out.mean(1).unsqueeze(0)
                # seq_out = seq_out[:, 0:4, :]
            # seq_out = torch.cat([seq_out[:, 0:4, :], context_out[arg].unsqueeze(0)], 1)

            # seq_out = (seq_out + context_out[arg].unsqueeze(0)) / 2.0
            # seq_out = torch.cat([seq_out, context_out[arg].unsqueeze(0)], 1)
            # seq_out = torch.cat([seq_out, context_out[arg].unsqueeze(0)], 1)
            seq_out, (hidden, state) = self.sequence_generator(seq_out, (hidden, state))



        # prod_val = torch.nn.Sigmoid()(torch.log(torch.Tensor(values)).cumsum(0))
        # prod_val = values.cumprod(0)
        # prod_val = values
        others = list(range(len(retrieved) - counter))
        shuffle(others)
        for i in others:
            values = torch.cat([values, torch.Tensor([0.0])], 0)

        out = ""
        other_args = list(range(len(retrieved)))
        for i in other_args:
            if i not in args:
                args.append(i)
        for arg in args:
            pid = retrieved[arg]
            if pid in ordinals:
                out += (" [{}]".format(ordinals[pid]))
            else:
                out += " ."

        # values = values / torch.Tensor(np.asarray([(i + 1) **2 for i in range(len(values))]))
        values = values / torch.Tensor(np.asarray([(i + 1) ** 2.0 for i in range(len(values))]))
        # prod_val = (torch.Tensor(values)).cumprod(0)
        # prod_val = (torch.Tensor(values)).cumprod(0)
        prod_val = (torch.Tensor(values))
        labels = self.construct_labels(args, ordinals, retrieved)
        labels = Variable(labels)

        relevants = set([i for i in retrieved if i in ordinals])
        rankings = [retrieved[i] for i in args]
        map = self.get_map(rankings, relevants)
        if first:
            self.predicted_maps.append(map)
            # if len(self.predicted_maps) > 10:
            #     self.predicted_maps = self.predicted_maps[1:]

        loss = self.loss(prod_val, labels)
        # print(out + "     [{}]".format(float(loss)))
        print(out + "     [{}]".format(map))
        return loss


