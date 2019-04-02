def read_corpus(corpus_file):
    out = []
    with open(corpus_file, encoding='utf8') as f:
        for line in f:
            tokens = line.strip().split()
            out.append( (tokens[1], tokens[3:]) )
        return out


def main():
    labeled_corpus = read_corpus('all_sentiment_shuffled.txt')
    print(len(labeled_corpus))


if __name__ == "__main__":
    main()
