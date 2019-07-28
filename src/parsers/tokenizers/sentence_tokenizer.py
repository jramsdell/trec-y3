from src.utils.text.token_utils import tokenize_sentences


class SentenceTokenizer(object):
    def __init__(self, max_sentences = 4, max_words = 20):
        self.max_sentences = max_sentences
        self.max_words = max_words


    def tokenize_documents(self, doc_list):
        tokenized_docs = []
        for doc in doc_list:
            tokenized_docs.append(tokenize_sentences(doc, max_sentences=self.max_sentences, max_words=self.max_words))
        return tokenized_docs




if __name__ == '__main__':
    pass