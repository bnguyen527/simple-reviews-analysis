import numpy as np
import pandas as pd


def read_corpus(corpus_file):
    out = []
    with open(corpus_file, encoding='utf8') as f:
        for line in f:
            tokens = line.strip().split()
            out.append( (tokens[1], tokens[3:]) )
        return out


def train_nb(training_data, alpha):
    num_neg = 0
    vocab = []
    counts_neg = []
    counts_pos = []
    num_tokens_neg = 0
    num_tokens_pos = 0
    for label, tokens in training_data:
        if label == 'neg':
            num_neg += 1
            num_tokens_neg += len(tokens)
        else:
            num_tokens_pos += len(tokens)
        for token in tokens:
            if token in vocab:
                idx = vocab.index(token)
            else:
                idx = len(vocab)
                vocab.append(token)
                counts_neg.append(0)
                counts_pos.append(0)
            if label == 'neg':
                counts_neg[idx] += 1
            else:
                counts_pos[idx] += 1
    probs_neg = list(map(lambda c: laplace_smoothing(c, num_tokens_neg, vocab, alpha), counts_neg))
    probs_pos = list(map(lambda c: laplace_smoothing(c, num_tokens_pos, vocab, alpha), counts_pos))
    df_probs = pd.DataFrame(index=vocab)
    df_probs['neg'] = probs_neg
    df_probs['pos'] = probs_pos
    return (num_neg / len(training_data), df_probs)


def laplace_smoothing(count, num_tokens, vocab, alpha):
    return (count+alpha) / (num_tokens+len(vocab)*alpha)


def main():
    labeled_corpus = read_corpus('all_sentiment_shuffled.txt')
    split = round(len(labeled_corpus) * 0.8)
    training_data = labeled_corpus[:split]
    testing_data = labeled_corpus[split:]
    prob_neg, df_probs = train_nb(training_data, 1)


if __name__ == "__main__":
    main()
