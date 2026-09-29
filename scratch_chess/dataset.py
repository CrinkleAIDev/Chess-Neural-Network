"""Bounded-memory shard loading and shuffled sequential passes over large data."""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

from .features import COMPACT_DTYPE, DTYPE, VERSION


class ShardDataset:
    def __init__(self, root, split, cache_shards=32):
        self.root=Path(root)
        manifest=json.loads((self.root/'manifest.json').read_text())
        if manifest['feature_version'] != VERSION:
            raise ValueError('Feature version mismatch')
        # Row format 1 stores both perspectives; format 2 only the side to move.
        self.dtype=COMPACT_DTYPE if manifest.get('row_format',1)==2 else DTYPE
        self.entries=manifest['shards'][split]
        if not self.entries:
            raise ValueError(f'No {split} data')
        self.ends=np.cumsum([e['count'] for e in self.entries])
        self.cache=OrderedDict()
        self.cache_shards=cache_shards

    def __len__(self):
        return int(self.ends[-1])

    def shard(self,index):
        if index not in self.cache:
            path=self.root/self.entries[index]['file']
            rows=np.load(path,allow_pickle=False) # only bounded active shards live in RAM
            if rows.dtype!=self.dtype or len(rows)!=self.entries[index]['count']:
                raise ValueError(f'Invalid shard: {path}')
            self.cache[index]=rows
            if len(self.cache)>self.cache_shards:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def __getitem__(self,indices):
        if isinstance(indices,slice):
            indices=np.arange(*indices.indices(len(self)))
        scalar=np.isscalar(indices)
        indices=np.atleast_1d(indices).astype(np.int64)
        if np.any(indices<0) or np.any(indices>=len(self)):
            raise IndexError('Dataset index out of range')
        ids=np.searchsorted(self.ends,indices,side='right')
        result=np.empty(len(indices),dtype=self.dtype)
        for shard in np.unique(ids):
            select=ids==shard
            start=int(self.ends[shard-1]) if shard else 0
            result[select]=self.shard(int(shard))[indices[select]-start]
        return result[0] if scalar else result


class ShuffledStream:
    """Random shard order, random rows, bounded mixed-shard block; visits every row.

    State records exact block offset and shuffle seed, so resume needs no giant
    permutation checkpoint. Fresh data coverage isn't replaced by repeated draws
    from a small in-memory subset.
    """
    # Shards are contiguous slices of the source file, so mix many per block (32 x 250k compact rows ~ 640 MB).
    def __init__(self,dataset,rng,state=None,block_shards=32):
        self.dataset=dataset
        self.rng=rng
        self.block_shards=block_shards
        self.order=[]
        self.cursor=0
        self.offset=0
        self.seed=None
        self.block=None
        self.epochs=0
        if state:
            self.order=state['order'];self.cursor=state['cursor'];self.offset=state['offset']
            self.seed=state['seed'];self.epochs=state['epochs']
            self.block_shards=state['block_shards']

    def state(self):
        return dict(order=self.order,cursor=self.cursor,offset=self.offset,seed=self.seed,
                    epochs=self.epochs,block_shards=self.block_shards)

    def load_block(self):
        if not self.order or self.cursor>=len(self.order):
            self.order=self.rng.permutation(len(self.dataset.entries)).tolist()
            self.cursor=0;self.offset=0;self.seed=None;self.epochs+=1
        if self.seed is None:
            self.seed=int(self.rng.integers(0,2**63))
        selected=self.order[self.cursor:self.cursor+self.block_shards]
        rows=np.concatenate([self.dataset.shard(i) for i in selected])
        np.random.default_rng(self.seed).shuffle(rows)
        self.block=rows

    def next(self,batch_size):
        parts=[];needed=batch_size
        while needed:
            if self.block is None:
                self.load_block()
            n=min(needed,len(self.block)-self.offset)
            if n:
                parts.append(self.block[self.offset:self.offset+n])
                self.offset+=n;needed-=n
            if self.offset==len(self.block):
                self.block=None;self.offset=0;self.seed=None;self.cursor+=self.block_shards
        return parts[0] if len(parts)==1 else np.concatenate(parts)
