from h5py import string_dtype
from numpy.dtypes import StringDType
import torch
from torch._dynamo.eval_frame import OptimizeContext
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.data.remote_backend_utils import num_nodes
from torch_geometric.io import fs
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from typing import Union, List, Tuple, Any, Optional, Callable
from kipoiseq import Interval
from kipoiseq.dataloaders import BedDataset

from data import FastaInterval, BigWigInterval, EpiInterval
import cooler

import numpy as np

from tqdm import tqdm


ENFORMER_LENTH = 196_608
GRAPH_REGION_DICT = {
    "1M": 2**20, "2M": 2**21, "4M": 2**22,
     "8M": 2**23,"16M": 2**24, "32M": 2**25
}


class GraphDataset(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None, pre_transform=None, pre_filter=None, map_location=None):
        """
        Args:
            root (str): The root directory where the dataset will be stored. 
                        The dataset will be saved in root/raw and root/processed.
            data_list (list): A list containing torch_geometric.data.Data objects.
                              If you already have a list of Data objects, you can pass them directly.
                              If the data needs to be read from files or downloaded, you can leave 
                                  this empty and handle it in download/process.
            transform (callable, optional): A function that is applied to each graph. It is executed before batching in the DataLoader.
            pre_transform (callable, optional): A function that is applied to each graph. It is executed before the data is saved to disk.
            pre_filter (callable, optional): A function used to filter graphs. It is executed before the data is saved to disk.
        """
        self.data_list = data_list
        super().__init__(root, transform, pre_transform, pre_filter)

        out = fs.torch_load(self.processed_paths[0], map_location=map_location)

        if not isinstance(out, tuple) or len(out) < 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:  # Backward compatibility.
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)
        
    @property
    def raw_file_names(self):
        return [] 

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        pass
        
    def process(self):
        if self.data_list is None:
            raise RuntimeError("data_list not provided. In a real scenario, you'd load from raw files here.")

        data_list = self.data_list

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.data, self.slices = self.collate(data_list)             
        self._data_list = None

        assert isinstance(self._data, Data)
        fs.torch_save(
            (self._data.to_dict(), self.slices, len(self.data_list), self._data.__class__),
            self.processed_paths[0],
        )


# class GraphDataset2(InMemoryDataset):
#     def __init__(self, root, interval_file, iterval_file_columns: int = 3, num_neighbors: int = 50, 
#         fasta_file: str = None, cool_file: Union[str, None] = None, 
#         region_length: str = "1M", model: Optional[torch.nn.Module] = None,
#         resolution: int = 4096, transform: Optional[Callable] = None, 
#         pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None):

#         self.interval_file_path = interval_file
#         self.interval_file_columns = iterval_file_columns
#         self.num_neighbors = num_neighbors
#         self.fasta_file_path = fasta_file
#         self.cool_file_path = cool_file
#         self.region_length = region_length
#         self.model = model
#         self.resolution = resolution

#         self.region_length_int = GRAPH_REGION_DICT[self.region_length]
#         self.num_nodes_per_graph = self.region_length_int // self.resolution


#         super().__init__(root, transform, pre_transform, pre_filter)

#         out = fs.torch_load(self.processed_paths[0])

#         if not isinstance(out, tuple) or len(out) < 3:
#             raise RuntimeError(
#                 "The 'data' object was created by an older version of PyG. "
#                 "If this error occurred while loading an already existing "
#                 "dataset, remove the 'processed/' directory in the dataset's "
#                 "root folder and try again.")
#         assert len(out) == 3 or len(out) == 4

#         if len(out) == 3:  # Backward compatibility.
#             data, self.slices, self.sizes = out
#             data_cls = Data
#         else:
#             data, self.slices, self.sizes, data_cls = out   

#         if not isinstance(data, dict):  # Backward compatibility.
#             self.data = data
#         else:
#             self.data = data_cls.from_dict(data)


#     @property
#     def raw_file_names(self):
#         return [self.interval_file_path]

#     @property
#     def processed_file_names(self):
#         return "data.pt"


#     def download(self):
#         pass
#     def process(self):

#         fasta_reader = FastaInterval(self.fasta_file_path)
#         cooler_reader = cooler.Cooler(self.cool_file_path)

#         model_instance = self.model

#         bed_dataset = BedDataset(self.interval_file_path, bed_columns=self.interval_file_columns)
#         all_interval = [bed_dataset[idx][0] for idx in range(len(bed_dataset))]

#         data_list = []

#         from multiprocessing import Pool, cpu_count

#         def process_interval(interval_tuple: Tuple[Interval]):
#             local_fasta_reader = FastaInterval(self.fasta_file_path)
#             local_cooler_reader = cooler.Cooler(self.cool_file_path) if self.cool_file_path is not None else None

#             chr, start, end = interval_tuple.chrom, interval_tuple.start, interval_tuple.end
#             interval = Interval(chr, start, end)

#             enformer_num, graph_num = self.get_enformer_num()
#             start_s = start - 320 * 128

#             enformer_interval_list = [Interval(chr, start_s + idx * ENFORMER_LENTH, start_s + (idx + 1) * ENFORMER_LENTH) for idx in range(enformer_num)]
#             enformer_input = [local_fasta_reader.extract(inter) for inter in enformer_interval_list]


#             if not enformer_input:
#                 return None

#             enformer_input_batch = torch.stack([torch.tensor(x, dtype=torch.float32) for x in enformer_input], dim=0)

#             with torch.no_grad():
#                 enformer_output_batch = model_instance(enformer_input_batch.to(model_instance.device))
#                 enformer_output_list = [output.t() for output in enformer_output_batch]

#             chip_track = torch.cat(enformer_output_list, dim=1)[:, :len(interval) // 128]

#             chip_chunk = torch.chunk(chip_track, self.num_nodes_per_graph, dim=1)
#             chip_chunk = [cp.reshape(1, -1) for cp in chip_chunk]
#             chip_track = torch.cat(chip_chunk, dim=0)



#             target = None

#             if local_cooler_reader is not None:
#                 cooler_region_str = f"{str(interval)}"
#                 cooler_target = local_cooler_reader.matrix(balance=True).fetch(f"{str(Interval(chr, start, end-self.resolution))[:-2]}")
#                 cooler_target = np.nan_to_num(cooler_target, nan=0)
#                 target = torch.tensor(cooler_target, dtype=torch.float32)

#             edge_index = create_neighbor_graph_simplified(num_nodes=self.num_nodes_per_graph, num_neighbors=self.num_neighbors)

#             chip_track = chip_track.t().contiguous()

#             chip_chunk = torch.chunk(chip_track, self.num_nodes_per_graph, dim=1)
#             chip_chunk = [cp.reshape(1, -1) for cp in chip_chunk]
#             chip_track = torch.cat(chip_chunk, dim=0).contiguous()

#             data = Data(x=chip_track, edge_index=edge_index, target=target, region = str(interval))
#             return data.cpu()

        
#         num_processes = cpu_count()
#         num_processes = 5

#         print(f"Processing data with (num_processes={num_processes} single process)")

#         data_list = []
#         for interval in tqdm(all_interval, desc="Processing data"):
#             data = process_interval(interval)
#             if data is not None:
#                 data_list.append(data)

#         # with Pool(
#         #     processes=num_processes,
#         #     initializer=process_interval,
#         #     initargs=(
#         #         all_interval,
#         #     )
#         # ) as pool:
#         #     results = list(tqdm(
#         #         pool.imap(process_interval, all_interval),
#         #     ))




#         self.data, self.slices = self.collate(data_list)
#         self._data_list = None


#         fs.torch_save(
#             (self.data.to_dict(), self.slices, len(self.data_list), self._data.__class__),
#             self.processed_paths[0],
#         )

#     def _get_enformer_num(self) -> Tuple[int, int]:
#         region_dna_len = GRAPH_REGION_DICT[self.region_length]
#         enformer_center = ENFORMER_LENTH -  320 * 128 * 2

#         tail = 0 if region_dna_len % enformer_center == 0 else 1
#         enformer_num = region_dna_len // enformer_center + tail
#         graph_num = region_dna_len // 128
#         return int(enformer_num), int(graph_num)

        
            




  




class EpiConcatMatrix(object):
    def __init__(self, fasta_file: str, cool_file: Union[str, None] = None, 
                resolution: int = 4096, 
                complement: bool = False,
                region_length: str = "1M", model: Optional[torch.nn.Module] = None) -> None:
        self.fasta_file = FastaInterval(fasta_file)

        if cool_file is not None:
            self.cool_file = cooler.Cooler(cool_file)
        else:
            self.cool_file = None

        self.resolution = resolution
        self.region_length = region_length
        if self.region_length not in ["1M", "2M", "4M", "8M", "16M", "32M"]:
            raise ValueError(f"region_length must be one of 1M, 2M, 4M, 8M, 16M, 32M, but got {self.region_length}")
        
        self.region_length_int = GRAPH_REGION_DICT[self.region_length]
        
        self.complement = complement

        assert model is not None, "model is required"

        self.model = model
        self.device = self.model.device


    def get_enformer_num(self) -> Tuple[int, int]:
        region_dna_len = GRAPH_REGION_DICT[self.region_length]
        enformer_center = ENFORMER_LENTH -  320 * 128 * 2

        tail = 0 if region_dna_len % enformer_center == 0 else 1
        enformer_num = region_dna_len // enformer_center + tail
        graph_num = region_dna_len // 128
        return int(enformer_num), int(graph_num)


    def __call__(self, interval: Interval, shuffle: bool = False) -> Any:
        chr, start, end = interval.chrom, interval.start, interval.end
        start_s = start - 320 * 128

        enformer_num, graph_num = self.get_enformer_num()
        stride = ENFORMER_LENTH - 320 * 128 * 2
        enformer_interval_list = [Interval(chr, start_s + idx * stride, start_s + idx * stride + ENFORMER_LENTH) for idx in range(enformer_num)]
        enformer_input = [torch.tensor(self.fasta_file.extract(inter)) for inter in enformer_interval_list]

        with torch.no_grad():
            enformer_output = [self.model(input.to(self.device)).t() for input in enformer_input]

        chip_track = torch.cat(enformer_output, dim=1)[:, :len(interval) // 128]
        chip_track = chip_track.cpu()

        if self.cool_file is not None:
            
            target = self.cool_file.matrix(balance=True).fetch(f"{str(Interval(chr, start, end-self.resolution))[:-2]}")
            target = np.nan_to_num(target, nan=0)
            target = torch.tensor(target, dtype=torch.float32, device=self.device)

            return chip_track, target
        else:
            return chip_track


# def create_neighbor_graph(num_nodes: int, num_neighbors: int) -> torch.Tensor:
#     edge_j = torch.repeat_interleave(torch.arange(num_nodes-1, dtype=torch.long), num_neighbors)

#     edge_i = torch.tensor([j for i in range(1, num_nodes) for j in range(i, i+ num_neighbors)], dtype=torch.long)
#     edge_index = torch.stack([edge_i, edge_j], dim=0)
#     edge_index = edge_index[:, :-(num_neighbors - 1)]

#     edge_index = torch.cat([edge_index, torch.flip(edge_index, dims=[0])], dim=1)

#     return edge_index




def create_neighbor_graph(num_nodes: int, num_neighbors: int) -> torch.Tensor:
    # num_neighbors is half-bandwidth, meaning nodes are connected if their distance is <= num_neighbors
    # Let's consider `num_neighbors` as the maximum distance in bins.
    
    # Upper triangular part
    rows_upper = []
    cols_upper = []
    for k in range(1, num_neighbors + 1): # k is the genomic distance in bins
        # Connect i to i+k
        rows_upper.append(torch.arange(0, num_nodes - k, dtype=torch.long))
        cols_upper.append(torch.arange(k, num_nodes, dtype=torch.long))

    if not rows_upper: # Handle case where num_nodes is too small for any neighbors
        return torch.empty((2, 0), dtype=torch.long)

    edge_i_upper = torch.cat(rows_upper)
    edge_j_upper = torch.cat(cols_upper)

    # Concatenate and add symmetric edges
    edge_index = torch.stack([
        torch.cat([edge_i_upper, edge_j_upper]),
        torch.cat([edge_j_upper, edge_i_upper])
    ], dim=0)

    # Remove self-loops (if k=0 was included or for other reasons, not here)
    # edge_index = edge_index[:, edge_index[0] != edge_index[1]]

    return edge_index


class CreateGraphData(EpiConcatMatrix):
    def __init__(self, interval_file, iterval_file_columns: int = 3, num_neighbors: int = 50, **kwargs):
        super().__init__(**kwargs)

        self.interval_file = BedDataset(interval_file, bed_columns=iterval_file_columns)
        self.num_neighbors = num_neighbors

        self.num_nodes = self.region_length_int // self.resolution
        
        
    def extract_data(self, interval: Interval):
        edge_index = create_neighbor_graph(num_nodes=self.num_nodes, num_neighbors=self.num_neighbors)

        target = None
        if self.cool_file is not None:
            chip_track, target = super().__call__(interval)
        else:
            chip_track = super().__call__(interval)

        chip_track = torch.chunk(chip_track, self.num_nodes, dim=1)
        chip_track = [cp.reshape(1, -1) for cp in chip_track]
        chip_track = torch.cat(chip_track, dim=0)

    
        data = Data(x=chip_track, edge_index=edge_index, target=target, region = str(interval))

        return data

    def __call__(self) -> List[torch_geometric.data.Data]:
        data_list = []

        for idx in tqdm(range(len(self.interval_file)), desc="Extracting data"):
            interval, _ = self.interval_file[idx]
            data = self.extract_data(interval)
            data_list.append(data)

        return data_list
        

if __name__ == "__main__":
    from finetune_enformer import EnformerFineTunerPL

    # data = CreateGraphData(interval_file="./", 
    #                         num_neighbors=50, 
    #                 model=.model.to("cuda"),
    #                 interval_file_columns=3,
    #                 )


#     data = CreateGraphData(
#     interval_file="./develop_test/train_3d_region.bed", 
#     fasta_file="./data/hg38.fa",
#     cool_file="./develop_test/wt_hic_4096.cool",
#     region_length="2M", model=EnformerFineTunerPL.load_from_checkpoint("./checkpoints/enformer-finetuned-epoch=09-val_loss_epoch=-0.1210.ckpt"),
# )

    data = CreateGraphData(
    interval_file="./develop_test/validation_3d_region.bed", 
    fasta_file="./data/hg38.fa",
    cool_file="./develop_test/wt_hic_4096.cool",
    region_length="2M", model=EnformerFineTunerPL.load_from_checkpoint("./checkpoints/enformer-finetuned-epoch=09-val_loss_epoch=-0.1210.ckpt"),
)

    interval =Interval("chr20", 50634852, 52732004)

    out = data.extract_data(interval)
    print(out)

    outputs = data()

    print("——————success process data——————")

    datast = GraphDataset(root=f"./develop_test/graph_data_validation_4096_2M", data_list=outputs)


    



        


        
        
