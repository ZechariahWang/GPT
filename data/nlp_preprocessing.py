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

        sentences = positive + negative

        vocab_words = set()
        for s in sentences:
            words = s.split()
            for w in words:
                vocab_words.add(w)
        vocab_words = sorted(vocab_words)

        vocab = {}
        i=1
        for w in vocab_words:
            vocab[w] = i
            i += 1

        seq = []
        for s in sentences:
            words = s.split()
            numbers=[]
            for w in words:
                numbers.append(vocab[w])
            seq.append(torch.tensor(numbers))

        return torch.nn.utils.rnn.pad_sequence(seq, padding_value=0, batch_first=True)

        