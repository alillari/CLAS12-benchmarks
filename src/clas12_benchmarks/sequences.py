"""Indexable, bounded-cache views for event and derived graph data."""
from collections import OrderedDict
from collections.abc import Sequence
import operator


class CachedSequence(Sequence):
    def __init__(self, count, load, cache_size=8):
        self.count, self.load = operator.index(count), load
        self.cache_size = operator.index(cache_size)
        if self.count < 0 or self.cache_size < 0:
            raise ValueError("Sequence count/cache size must be nonnegative")
        self.cache = OrderedDict()

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = range(*index.indices(self.count))
            return CachedSequence(len(indices), lambda i: self[indices[i]], self.cache_size)
        index = operator.index(index)
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        value = self.load(index)
        if self.cache_size:
            self.cache[index] = value
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return value


def event_sequence(cohort, role, count=None, cache_size=8):
    count = len(cohort.roles[role]) if count is None else count
    if cohort.is_hdf5:
        return CachedSequence(count, lambda i: cohort.event(role, i), cache_size)
    return [cohort.event(role, i) for i in range(count)]


def derived_sequence(events, load, cache_size=8):
    if isinstance(events, CachedSequence):
        return CachedSequence(len(events), load, cache_size)
    return [load(i) for i in range(len(events))]
