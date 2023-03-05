import torch
import random
import os, glob, dill, json
import os.path as osp
from data.datasets import InMemoryComplexDataset
from data.dummy_complexes import get_testing_complex_list, get_mol_testing_complex_list
from data.utils import convert_graph_dataset_with_rings
from torch_geometric.data import Data

class Model_NN_Graph(InMemoryComplexDataset):
    """A dummy dataset using a list of hand-crafted cell complexes with many edge cases."""

    def __init__(self, root, task, node_feature_type, feature_dim=192,max_ring_size =None):
        self.name = 'Mode_NN_Graph'
        self.node_feature_type = node_feature_type
        self.task = task
        self.feature_dim = feature_dim
        self._max_ring_size = max_ring_size
        self._use_edge_features = False
        self._n_jobs = 1
        print("initiailizing Model_NN_Graph with root ", root, "task : ",  task)
        super(Model_NN_Graph, self).__init__(root, max_dim=192, num_classes=1,
                                           init_method="sum", include_down_adj=True, cellular=False)

        print("Data loader initiated")
        print("processed name : " , self.processed_paths)
        self.data, self.slices = torch.load(self.processed_paths[0])
        total_indexes = list(range(self.len()))
        train_ratio = 0.8

        # Calculate the number of examples in the training set
        num_train = int(train_ratio * len(total_indexes))

        # Shuffle the indices randomly
        random.shuffle(total_indexes)

        # Split the indices into training and test sets
        train_indices = total_indexes[:num_train]
        test_indices = total_indexes[num_train:]
        self.train_ids = train_indices
        self.test_ids = test_indices

    @property
    def processed_file_names(self):
        name = self.name
        print("set processed_file names ")
        return [f'{self.task}_{self.node_feature_type}_complex_list.pt',
                f'{self.task}_{self.node_feature_type}_model_nm.dill']

    @property
    def raw_file_names(self):
        self.raw_file_dir =os.path.join(
            self.root,'raw',self.node_feature_type)
        self.task_performance_dir = os.path.join(
            self.root, 'raw', f'{self.task}_performance_score.json')
        print("called raw file names", self.raw_file_dir)
        print("called raw file names", self.task_performance_dir)
        # The processed graph files are our raw files.
        # They are obtained when running the initial data conversion S2V_to_PyG.
        return []

    def download(self):
        return

    @staticmethod
    def factory():
        print("calling factory")
        complexes = get_testing_complex_list()
        for c, complex in enumerate(complexes):
            complex.y = torch.LongTensor([c % 2])
            print("complex . y : ", complex, complex.y)
        return complexes

    def process(self):
        with open(self.task_performance_dir, 'r') as f:
            performance=json.load(f)
            performance=performance[self.task]
        dill_files_dirs=glob.glob(self.raw_file_dir+"/*")
        graph_info = []
        for dill_file_dir in dill_files_dirs:
            if dill_file_dir.endswith('.dill') is False:
                continue
            with open(dill_file_dir, 'rb') as f:
                graph_obj=dill.load(f)
                graph_info += graph_obj
        self.graphs= [x[0] for x in graph_info]
        self.params= [x[1] for x in graph_info]
        self.model_nm= [x[2] for x in graph_info]
        for (G,p) in zip(self.graphs, self.params):
            param_keys = p.keys()
            for u in G.nodes:
                label = G.nodes[u]['label']
                if label in param_keys:
                    feature=p[label]
                else:
                    feature = torch.zeros(self.feature_dim).squeeze()
                G.nodes[u]['feat'] = feature.float()

        graphs, params, model_nms, accuracies = [], [], [], []
        for idx, full_name in enumerate(self.model_nm):
            if full_name in performance.keys():
                graphs.append(self.graphs[idx])
                params.append(self.params[idx])
                model_nms.append(self.model_nm[idx])
                accuracies.append(performance[full_name]["eval_matthews_correlation"])
        data_list = []
        for graph, model_name, y in zip(graphs, model_nms, accuracies):
            key_to_num = { u:idx for idx, u in enumerate(graph.nodes)}
            x = torch.stack([graph.nodes[u]['feat'] for u in graph.nodes])
            edge_index =torch.tensor([
                [key_to_num[edge[0]] for edge in graph.edges()],
                [key_to_num[edge[1]] for edge in graph.edges()] ]).to(torch.long)
            edge_attr = None
            data = Data(x=x, edge_index=edge_index, edge_attr= edge_attr, y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        complexes, _, _ = convert_graph_dataset_with_rings(
            data_list,
            max_ring_size=self._max_ring_size,
            include_down_adj=self.include_down_adj,
            init_edges=True,
            init_rings=True,
            n_jobs=self._n_jobs)
        for idx, (complex, y, model_nm) in enumerate(zip(complexes, accuracies, model_nms )):
            complexes[idx].model_nm = model_nm
            assert complex.y == y

            # print("complex.model_nm :", complexes[idx].model_nm)
            print("Type : ", type(complexes[idx]))
        # assert False
        torch.save(self.collate(complexes,self.max_dim), self.processed_paths[0])
        # with open(self.processed_paths[1]))
        # torch.save(self.collate(model_nms))
        # print("Torch data saved")
