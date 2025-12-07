class SimpleTokenizer:
    def __init__(self, vocab=None):
        # vocab: dict token -> id
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        if vocab is None:
            self.vocab = {self.pad_token: 0, self.unk_token: 1}
        else:
            self.vocab = vocab

    def build_from_texts(self, texts):
        """
        Build the vocabulary from a list of texts.
        """
        for txt in texts:
            for tok in txt.lower().split():
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)

    def encode(self, text):
        """
        Encode a text into a list of ids.
        """
        toks = text.lower().split()
        ids = [self.vocab.get(t, self.vocab[self.unk_token]) for t in toks]
        return ids
