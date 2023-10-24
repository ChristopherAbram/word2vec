import json
import logging
import pickle
import random
from collections import Counter, OrderedDict
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torchdata.dataloader2 import DataLoader2
from torchdata.datapipes import iter as it
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import Vocab, vocab

SEED = 42
CBOW_N_WORDS = 4
SKIPGRAM_N_WORDS = 4
MAX_SEQUENCE_LENGTH = 256
MIN_FREQ = 50
SUBSAMPLING_THRESHOLD = 1e-3

# random.seed(SEED)


def preprocess_cbow(text_tokens_ids: List[int], context_size: int = CBOW_N_WORDS) -> List[Tuple[List[int], int]]:
    tokens = []
    for idx in range(len(text_tokens_ids) - context_size * 2):
        token_id_sequence = text_tokens_ids[idx : (idx + context_size * 2 + 1)]
        target_token_id = token_id_sequence.pop(context_size)
        context_token_ids = token_id_sequence
        tokens.append((context_token_ids, target_token_id))
    return tokens


def preprocess_skipgram(text_tokens_ids: List[int], context_size: int = SKIPGRAM_N_WORDS) -> List[Tuple[int, int]]:
    tokens = []
    for idx in range(len(text_tokens_ids) - context_size * 2):
        token_id_sequence = text_tokens_ids[idx : (idx + context_size * 2 + 1)]
        target_token_id = token_id_sequence.pop(context_size)
        context_token_ids = token_id_sequence
        for context_token_id in context_token_ids:
            tokens.append((target_token_id, context_token_id))
    return tokens


def subsample(sample: Tuple[int, int], discard_prob_dict: Dict[int, float]) -> bool:
    target_token_id, context_token_id = sample
    drop_prob = random.random()
    drop_target_token = drop_prob <= discard_prob_dict[target_token_id]
    drop_context_token = drop_prob <= discard_prob_dict[context_token_id]
    return not (drop_target_token or drop_context_token)


def map_to_vocabulary(vocab: Vocab, tokenizer: Callable[[str], List[str]], text: str) -> List[int]:
    return vocab(tokenizer(text))


def filter_sequences_by_length(text_tokens_ids: List[int], min_length: int = 2 * CBOW_N_WORDS + 1) -> bool:
    return len(text_tokens_ids) >= min_length


def collate(token_pairs: List[Tuple[Union[List[int], int], int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xy_batch, noise_batch = zip(*token_pairs)
    x_batch, y_batch = zip(*xy_batch)
    x_batch, y_batch, noise_batch = (
        torch.tensor(x_batch, dtype=torch.long),
        torch.tensor(y_batch, dtype=torch.long),
        torch.tensor(noise_batch, dtype=torch.long),
    )
    return x_batch, y_batch, noise_batch


def build_vocabulary(
    dp: it.IterDataPipe, tokenizer: Callable[[str], List[str]], min_freq: int = MIN_FREQ
) -> Tuple[Vocab, Dict[int, int]]:
    # Create a map of word frequencies
    tokens_dp = dp.map(tokenizer)
    token_to_count = Counter()
    for tokens in tokens_dp:
        token_to_count.update(tokens)

    # Build word vocabulary using <unk> as default token
    specials = ["<unk>"]
    sorted_by_freq_tuples = sorted(token_to_count.items(), key=lambda x: (-x[1], x[0]))
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    word_vocab = vocab(ordered_dict, min_freq=min_freq, specials=specials, special_first=True)
    word_vocab.set_default_index(word_vocab["<unk>"])

    # Create a map from token id to token count
    id_to_count = {token_id: token_to_count[token] for token, token_id in word_vocab.get_stoi().items()}
    return word_vocab, id_to_count


def create_discard_dict(id_to_freq: Dict[int, float], t: float = SUBSAMPLING_THRESHOLD) -> Dict[int, float]:
    discard_prob_dict = {token_id: 1 - np.sqrt(t / (freq + t)) for token_id, freq in id_to_freq.items()}
    return discard_prob_dict


def get_t(id_to_freq: Dict[int, float], q_percentile: float = 100) -> float:
    return np.percentile(list(id_to_freq.values()), q_percentile)


def get_freq(id_to_count: Dict[int, int]) -> Dict[int, float]:
    total_token_count = sum(id_to_count.values())
    return {token_id: token_count / total_token_count for token_id, token_count in id_to_count.items()}


def get_noise_dist(id_to_freq: Dict[int, float]) -> np.ndarray:
    """
    Compute noise distribution as a function of the unigram distribution. This implementation approximates
    unigram distribution using word frequencies.

    :param id_to_freq: Word frequency dictionary {token_id: relative_freq}
    :return: Numpy array that represents distribution, each
    """
    unigram_dist = np.array(sorted(id_to_freq.values(), reverse=True))
    noise_dist = unigram_dist**0.75
    noise_dist_norm = noise_dist / noise_dist.sum()
    return noise_dist_norm


def draw_noise_samples(
    rng: np.random.Generator, vocab_size: int, n_samples: int, noise_dist: Optional[np.ndarray] = None
) -> List[int]:
    """
    Draw n noise token ids from noise distribution given vocabulary size. Assume uniform
    distribution if noise_dist is None.

    :param rng: Random generator
    :param vocab_size: Size of considered vocabulary
    :param n_samples: Number of noise samples to draw
    :param noise_dist: Unigram noise distribution derived from word frequencies, defaults to None
    :return: Noise token ids.
    """
    return rng.choice(vocab_size, n_samples, replace=False, p=noise_dist, shuffle=False).tolist()


def get_noise_data_pipe(
    vocab_size: int, n_samples: int = 5, id_to_freq: Optional[Dict[int, float]] = None
) -> it.IterDataPipe:
    """
    Create data pipe that generates random noise samples (token ids) based on the vocabulary of vocab_size.
    Assume uniform distribution if word frequency is not given, otherwise draw from unigram
    distribution derived from word frequencies. This is a part of the negative sampling implementation.

    :param vocab_size: Size of considered vocabulary
    :param n_samples: Number of noise samples per one input sample, defaults to 5
    :param id_to_freq: Word frequency dictionary {token_id: relative_freq}, defaults to None
    :return: Data pipe that outputs noise samples based on word frequencies.
    """
    rng = np.random.default_rng(seed=SEED)
    if id_to_freq is None:
        # Sample token ids uniformly
        sample_fn = partial(draw_noise_samples, rng, vocab_size, n_samples)
    else:
        # Sample words from noise distribution (based on word freq)
        noise_dist = get_noise_dist(id_to_freq)
        sample_fn = partial(draw_noise_samples, rng, vocab_size, n_samples, noise_dist)

    return it.IterableWrapper(iter(sample_fn, -1))


def get_dataloader(
    dp: it.IterDataPipe,
    word_vocab: Vocab,
    id_to_count: Dict[int, int],
    tokenizer: Callable[[str], List[str]],
    context_size: int = SKIPGRAM_N_WORDS,
    n_noise_samples: int = 5,
    batch_size: int = 64,
    subsampling_threshold: int = 90,
) -> DataLoader2:
    """
    Build data loader based on raw data pipe that outputs lines from the input dataset.
    Preprocess raw data in the following order:

     - map to vocabulary: tokenize and convert words to token ids,
     - filter out lines that are shorter than a context window, i.e., 2 * context_size + 1,
     - apply skip-gram transformation per context window: form input and output samples,
     - filter out frequent words: apply sub-sampling technique,
     - add noise samples drawn randomly from noise distribution (unigram or uniform dist),
     - batch input, output, and noise samples,
     - convert lists to tensors.

    :param dp: Data pipe with raw data
    :param word_vocab: Vocabulary object
    :param id_to_count: Word frequency {token_id: count}
    :param tokenizer: Tokenizer function
    :param context_size: Size of the context window, defaults to SKIPGRAM_N_WORDS
    :param n_noise_samples: Number of noise samples per single input - output pair, defaults to 5
    :param batch_size: Size of the batch, defaults to 64
    :param subsampling_threshold: N-th percentile of the frequency list to be used as threshold t
        to compute discard probabilities, defaults to 90
    :return: Dataloader instance.
    """
    # Create token discard probability for subsampling
    id_to_freq = get_freq(id_to_count)
    threshold = get_t(id_to_freq, subsampling_threshold)
    discard_prob_dict = create_discard_dict(id_to_freq, t=threshold)

    # Create noise data pipe that draws tokens from noise distribution
    vocab_size = len(word_vocab.get_stoi())
    noise_dp = get_noise_data_pipe(vocab_size, n_noise_samples, id_to_freq)

    # Define preprocessing pipeline
    dp = (
        dp.map(partial(map_to_vocabulary, word_vocab, tokenizer))
        .filter(partial(filter_sequences_by_length, min_length=2 * context_size + 1))
        .flatmap(partial(preprocess_skipgram, context_size=context_size))
        .filter(partial(subsample, discard_prob_dict=discard_prob_dict))
        .zip(noise_dp)
        .batch(batch_size)
        .collate(collate)
    )
    dataloader = DataLoader2(dp)
    return dataloader


class WikiText2DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = ".data",
        output_dir: str = ".out",
        batch_size: int = 128,
        context_size: int = SKIPGRAM_N_WORDS,
        n_noise_samples: int = 5,
        word_min_freq: int = 10,
        subsampling_threshold_percentile: int = 90,
        max_vocab_size: int = 20000,
    ) -> None:
        """


        :param data_dir: Path to data directory, defaults to ".data"
        :param batch_size: _description_, defaults to 128
        :param context_size: _description_, defaults to SKIPGRAM_N_WORDS
        :param n_noise_samples: _description_, defaults to 5
        :param word_min_freq: _description_, defaults to 10
        :param subsampling_threshold_percentile: _description_, defaults to 90
        :param max_vocab_size: _description_, defaults to 20000
        """
        super().__init__()
        # Paths
        self.project_path = Path(__file__).parent.parent
        self.data_dir = self.project_path / data_dir
        self.output_dir = Path(output_dir)
        self.stoi_dict_path = self.output_dir / "stoi.json"
        self.itos_dict_path = self.output_dir / "itos.json"
        self.count_dict_path = self.output_dir / "count.json"
        self.vocab_path = self.output_dir / "vocab.obj"
        # Params
        self.context_size = context_size
        self.n_noise_samples = n_noise_samples
        self.max_vocab_size = max_vocab_size
        self.batch_size = batch_size
        self.word_min_freq = word_min_freq
        self.subsampling_threshold = subsampling_threshold_percentile
        # Other
        self.tokenizer = get_tokenizer("basic_english", language="en")

    def prepare_data(self) -> None:
        # Download dataset split
        dp_train, _, _ = WikiText2(root=self.data_dir, split=("train", "valid", "test"))

        # Build vocabulary (use only train split)
        word_vocab, id_to_count = build_vocabulary(dp_train, self.tokenizer, min_freq=self.word_min_freq)
        vocab_size = len(word_vocab.get_stoi())
        logging.info(f"Vocabulary size={vocab_size}")

        # Save raw vocabulary to file (word to id)
        with open(self.stoi_dict_path, "w") as fvocab:
            json.dump(word_vocab.get_stoi(), fvocab)
        # Save raw vocabulary to file (id to word)
        with open(self.itos_dict_path, "w") as fvocab:
            json.dump(word_vocab.get_itos(), fvocab)
        # Save raw word frequency dict to file
        with open(self.count_dict_path, "w") as fcount:
            json.dump(id_to_count, fcount)
        # Save pickled vocab object
        with open(self.vocab_path, "wb") as fvocab:
            pickle.dump(word_vocab, fvocab)

    def setup(self, stage: str) -> None:
        # Load vocabulary and word count
        with open(self.vocab_path, "rb") as fvocab:
            self.vocab: Vocab = pickle.load(fvocab)
        # Load word count dict
        with open(self.count_dict_path, "r") as fcount:
            self.id_to_count: Dict[int, int] = json.load(
                fcount, object_hook=lambda d: {int(k): v for k, v in d.items()}
            )
        # Get vocabulary size
        self.vocab_size = len(self.vocab.get_stoi())
        # Prepare dataset split
        self.dp_train, self.dp_valid, self.dp_test = WikiText2(root=self.data_dir, split=("train", "valid", "test"))

    def _dataloader(self, dp: it.IterDataPipe) -> DataLoader2:
        return get_dataloader(
            dp,
            word_vocab=self.vocab,
            id_to_count=self.id_to_count,
            tokenizer=self.tokenizer,
            context_size=self.context_size,
            n_noise_samples=self.n_noise_samples,
            batch_size=self.batch_size,
            subsampling_threshold=self.subsampling_threshold,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._dataloader(self.dp_train)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._dataloader(self.dp_valid)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._dataloader(self.dp_test)
