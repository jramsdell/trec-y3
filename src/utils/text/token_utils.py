from textblob import TextBlob

def tokenize_sentences(text, max_words=0, max_sentences=0):
    blob = TextBlob(text)
    sentences = []
    sentence_counter = 0

    for sentence in blob.sentences:
        tokens = sentence.words[0:max_words] if max_words > 0 else sentence.words[:]
        sentences.append(tokens)
        sentence_counter += 1
        if max_sentences > 0 and sentence_counter > max_sentences:
            break

    return sentences
