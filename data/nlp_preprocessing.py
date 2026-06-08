import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)

        # positive sentiment first + negative sentiment
        sentences = positive + negative

        # turn all sentences into a set of words, to avoid duplicates
        vocab_words = set()
        for s in sentences:
            words = s.split()
            for w in words:
                vocab_words.add(w)
        vocab_words = sorted(vocab_words)

        # assign a tokenid to each words (token)
        # start iterating from 1 since 0 index is considered padding loc
        vocab = {}
        i=1
        for w in vocab_words:
            vocab[w] = i
            i += 1

        # convert each word back into its token id and append to the torch tensor
        seq = []
        for s in sentences:
            words = s.split()
            numbers=[]
            for w in words:
                numbers.append(vocab[w])
            seq.append(torch.tensor(numbers))

        # return padding seq
        return torch.nn.utils.rnn.pad_sequence(seq, padding_value=0, batch_first=True)

        