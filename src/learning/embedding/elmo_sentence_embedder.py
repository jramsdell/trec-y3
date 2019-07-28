from typing import *

import numpy as np
from allennlp.commands.elmo import ElmoEmbedder


class ElmoVectorEmbedderRunner(object):

    def __init__(self):
        self.max_sentences = 4
        self.max_words = 40
        self.embedding_size = 1024 * 3
        self.model = ElmoEmbedder()
        self.null_vector = np.zeros((self.max_sentences, 1024 * 3))



    def get_embedding(self, tokens):
        # return [self.model.embed_sentence(i) for i in p1["tokens"]]
        # sentences =  [i[0] for i in self.model.embed_sentences(tokens[0:self.max_sentences])]
        sentences =  [i for i in self.model.embed_sentences(tokens[0:self.max_sentences])]
        for idx in range(len(sentences)):
            sentence = sentences[idx]
            f1, f2, f3 = sentence
            f1 = f1.mean(0)
            f2 = f2.mean(0)
            f3 = f3.mean(0)
            combined = np.concatenate([f1, f2, f3], 0)
            # if sentence.shape[0] < self.max_words:
            #     word_diff = self.max_words - sentence.shape[0]
            #     zshape = (word_diff, sentence.shape[1])
            #     sentence = np.concatenate([sentence, np.zeros(zshape)], 0)
            # sentences[idx] = sentence.mean(0)
            sentences[idx] = combined

        sentences = np.asarray(sentences)


        try:
            if sentences.shape[0] < self.max_sentences:
                sentence_diff = self.max_sentences - sentences.shape[0]
                # zshape = (sentence_diff, self.max_words, self.embedding_size)
                zshape = (sentence_diff, self.embedding_size)
                sentences = np.concatenate([sentences, np.zeros(zshape)], 0)
        except ValueError:
            return None


        return sentences





    def map_function(self, text_tokens):
        results = []
        mlength = len(text_tokens)
        for idx, tokens in enumerate(text_tokens):
            embedded = self.get_embedding(tokens)
            if embedded is not None:
                results.append(embedded)
            else:
                results.append(self.null_vector)
                print("Problem with: {}".format(idx))
            if idx % 100 == 0:
                print("{} out of {}".format(idx, mlength))
        return results





if __name__ == '__main__':
    pass
