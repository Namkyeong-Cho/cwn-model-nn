"""
Code is adapted from https://github.com/rusty1s/pytorch_geometric/blob/6442a6e287563b39dae9f5fcffc52cd780925f89/torch_geometric/data/dataloader.py

Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
Copyright (c) 2021 The CWN Project Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import os
import torch
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch
from torch._six import container_abcs, string_classes, int_classes

from definitions import ROOT_DIR
from data.complex import Cochain, CochainBatch, Complex, ComplexBatch
from data.datasets import (
    load_sr_graph_dataset, load_tu_graph_dataset, load_zinc_graph_dataset, load_ogb_graph_dataset,
    load_ring_transfer_dataset, load_ring_lookup_dataset)
from data.datasets import (
    SRDataset, ClusterDataset, TUDataset, ComplexDataset, FlowDataset,
    OceanDataset, ZincDataset, CSLDataset, OGBDataset, RingTransferDataset, RingLookupDataset,
    DummyDataset, DummyMolecularDataset, Model_NN_Graph)


class Collater(object):
    """Object that converts python lists of objects into the appropiate storage format.

    Args:
        follow_batch: Creates assignment batch vectors for each key in the list.
        max_dim: The maximum dimension of the cochains considered from the supplied list.
    """
    def __init__(self, follow_batch, max_dim=2):
        self.follow_batch = follow_batch
        self.max_dim = max_dim

    def collate(self, batch):
        """Converts a data list in the right storage format."""
        elem = batch[0]
        # print("elem :" , elem, self.follow_batch, elem.model_nm)
        # assert False
        if isinstance(elem, Cochain):
            return CochainBatch.from_cochain_list(batch, self.follow_batch)
        elif isinstance(elem, Complex):
            return ComplexBatch.from_complex_list(batch, self.follow_batch, max_dim=self.max_dim)
        elif isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges cochain complexes into to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
        max_dim (int): The maximum dimension of the chains to be used in the batch.
            (default: 2)
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 max_dim=2, **kwargs):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for Pytorch Lightning...
        self.follow_batch = follow_batch

        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=Collater(follow_batch, max_dim), **kwargs)


def load_dataset(name, root=os.path.join(ROOT_DIR, 'datasets'), task=None,
                 node_feature_type=None, max_dim=2, fold=0,
                 init_method='sum', n_jobs=2, **kwargs) -> ComplexDataset:
    """Returns a ComplexDataset with the specified name and initialised with the given params."""

    print("*" * 100)
    print("load_dataset")
    print("*"*100)

    return Model_NN_Graph(os.path.join(root, name), task, node_feature_type=node_feature_type,
                             max_ring_size=kwargs.get('max_ring_size', None))

