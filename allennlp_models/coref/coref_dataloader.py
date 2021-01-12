from typing import List, Dict, Union, Iterator
import random
from copy import deepcopy

import torch
from torch.utils import data

from allennlp.common.registrable import Registrable
from allennlp.common.lazy import Lazy
from allennlp.data import Token, Vocabulary
from allennlp.data.fields import TextField, SpanField, ListField
from allennlp.data.instance import Instance
from allennlp.data.batch import Batch
from allennlp.data.samplers import Sampler, BatchSampler
from allennlp.data.dataloader import allennlp_collate, TensorDict, DataLoader


def sentence_removal_collate(vocab: Vocabulary, instances: List[Instance], probability_of_modified_text: float = 1) -> TensorDict:
    augmented_instances = []
    for instance in instances:
        sentences = instance["metadata"]["sentences"]
        removed_sentence_index = random.randint(0, len(sentences)-1)
        removed_sentence_length = len(sentences[removed_sentence_index])
        modified_sentences = sentences[:removed_sentence_index]+sentences[removed_sentence_index+1:]
        words = [Token(word) for sentence in modified_sentences for word in sentence]
        sentence_index_span_map = instance["metadata"]["sentence_index_span_map"]
        spans = [span for sent_index in range(removed_sentence_index) for span in sentence_index_span_map[sent_index]]+[(span[0]-removed_sentence_length, span[1]-removed_sentence_length) for sent_index in range(removed_sentence_index+1, len(sentences)) for span in sentence_index_span_map[sent_index]]
        if len(spans) > 0 and len(sentences) > 1 and random.random() < probability_of_modified_text:
            instance.add_field("modified_text", TextField(words, instance["text"]._token_indexers))
            spans = [SpanField(span[0], span[1], instance["modified_text"]) for span in spans]
            instance.add_field("modified_spans", ListField(spans))
            instance["metadata"].metadata["removed_text_start"] = sum(len(s) for s in sentences[:removed_sentence_index])
            instance["metadata"].metadata["removed_text_end"] = instance["metadata"].metadata["removed_text_start"]+removed_sentence_length
            instance["metadata"].metadata["modified_span_indices"] = [i for i in range(len(instance["spans"].field_list)) if instance["spans"].field_list[i].span_start < instance["metadata"].metadata["removed_text_start"] or instance["spans"].field_list[i].span_start >= instance["metadata"].metadata["removed_text_end"]]
            instance["modified_text"].index(vocab)
            instance["metadata"].metadata["modified_text_loss"] = True
            augmented_instances.append(instance)
            instance2 = deepcopy(instance)
            instance2["metadata"].metadata["modified_text_loss"] = False
            augmented_instances.append(instance2)
        else:
            instance.add_field("modified_text", instance["text"])
            instance.add_field("modified_spans", instance["spans"])
            instance["metadata"].metadata["modified_span_indices"] = list(range(len(instance["spans"].field_list)))
            instance["metadata"].metadata["modified_text_loss"] = True
            augmented_instances.append(instance)

    batch = Batch(augmented_instances)
    return batch.as_tensor_dict(batch.get_padding_lengths())


@DataLoader.register("coref_dataloader", constructor="from_partial_objects")
class CorefDataLoader(data.DataLoader, DataLoader):
    """
    A registrable version of the pytorch
    [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).
    Firstly, this class exists is so that we can construct a DataLoader
    from a configuration file and have a different default `collate_fn`.
    You can use this class directly in python code, but it is identical to using
    pytorch dataloader with allennlp's custom collate function:

    ```
    from torch.utils.data import DataLoader

    from allennlp.data import allennlp_collate
    # Construct a dataloader directly for a dataset which contains allennlp
    # Instances which have _already_ been indexed.
    my_loader = DataLoader(dataset, batch_size=32, collate_fn=allennlp_collate)
    ```

    Secondly, this class adds a `batches_per_epoch` parameter which, if given, determines the number
    of batches after which an epoch ends.  If this is `None`, then an epoch is set to be one full pass
    through your data.  You might use this if you have a very large dataset and want more frequent
    checkpoints and evaluations on validation data, for instance.

    In a typical AllenNLP configuration file, the `dataset` parameter does not get an entry under
    the "data_loader", it gets constructed separately.
    """

    def __init__(
        self,
        dataset: data.Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Sampler = None,
        batch_sampler: BatchSampler = None,
        num_workers: int = 0,
        # NOTE: The default for collate_fn is different from the normal `None`.
        # We assume that if you are using this class you are using an
        # allennlp dataset of instances, which would require this.
        collate_fn=sentence_removal_collate,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: int = 0,
        worker_init_fn=None,
        multiprocessing_context: str = None,
        batches_per_epoch: int = None,
        probability_of_modified_text: float = 1
    ):
        collate_fn = lambda x: sentence_removal_collate(vocab=batch_sampler.vocab, instances=x, probability_of_modified_text=probability_of_modified_text)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
        )
        self._data_generator = super().__iter__()
        self._batches_per_epoch = batches_per_epoch

    def __len__(self):
        if self._batches_per_epoch is not None:
            return self._batches_per_epoch
        return super().__len__()*2

    def slice_dict(self, dictionary, slice_index, slice_size):
        slice_dictionary = {}
        for key in dictionary:
            if isinstance(dictionary[key], Dict):
                field_slice = self.slice_dict(dictionary[key], slice_index, slice_size)
            else:
                field_slice = dictionary[key][slice_index:slice_index+slice_size]
            slice_dictionary[key] = field_slice
        return slice_dictionary

    def __iter__(self):
        if self._batches_per_epoch is None:
            # NOTE: since torch's DataLoader is listed as the first super class of this class,
            # super().__iter__() will resolve to the __iter__ method from torch's DataLoader,
            # which is what we want.
            # yield from super().__iter__()
            iterator = super().__iter__()
            for batch in iterator:
                arbitrary_key = None
                for key in batch:
                    if not isinstance(batch[key], Dict):
                        arbitrary_key = key
                batch_length = len(batch[arbitrary_key])
                for i in range(0, batch_length, 1):
                    smaller_batch = self.slice_dict(batch, i, 1)
                    yield smaller_batch
        else:
            for i in range(self._batches_per_epoch):
                try:
                    yield next(self._data_generator)
                except StopIteration:  # data_generator is exhausted
                    self._data_generator = super().__iter__()  # so refresh it
                    yield next(self._data_generator)  # and yield required instance

    @classmethod
    def from_partial_objects(
        cls,
        dataset: data.Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Lazy[Sampler] = None,
        batch_sampler: Lazy[BatchSampler] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: int = 0,
        worker_init_fn=None,
        multiprocessing_context: str = None,
        batches_per_epoch: int = None,
        probability_of_modified_text: float = 1,
    ) -> "CorefDataLoader":
        batch_sampler_ = (
            None if batch_sampler is None else batch_sampler.construct(data_source=dataset)
        )
        sampler_ = None if sampler is None else sampler.construct(data_source=dataset)
        batch_size = batch_sampler_.batch_size

        return cls(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler_,
            batch_sampler=batch_sampler_,
            num_workers=num_workers,
            # NOTE: The default for collate_fn is different from the normal `None`.
            # We assume that if you are using this class you are using an
            # allennlp dataset of instances, which would require this.
            collate_fn=sentence_removal_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            batches_per_epoch=batches_per_epoch,
            probability_of_modified_text=probability_of_modified_text
        )
