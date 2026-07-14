import torch
import numpy as np
import cooler
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs
from kipoiseq import Interval
from kipoiseq.dataloaders import BedDataset
from typing import Union, List, Tuple, Optional
from tqdm import tqdm

from data_pipeline.data import FastaInterval

ENFORMER_LENGTH = 196_608
GRAPH_REGION_DICT = {
    "1M": 2**20, "2M": 2**21, "4M": 2**22,
    "8M": 2**23, "16M": 2**24, "32M": 2**25
}


def create_neighbor_graph(num_nodes: int, num_neighbors: int) -> torch.Tensor:
    rows, cols = [], []
    for k in range(1, num_neighbors + 1):
        rows.append(torch.arange(0, num_nodes - k, dtype=torch.long))
        cols.append(torch.arange(k, num_nodes, dtype=torch.long))
    if not rows:
        return torch.empty((2, 0), dtype=torch.long)
    r, c = torch.cat(rows), torch.cat(cols)
    return torch.stack([torch.cat([r, c]), torch.cat([c, r])], dim=0)


class GraphDataset(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None, pre_transform=None,
                 pre_filter=None, map_location=None):
        self.data_list = data_list
        super().__init__(root, transform, pre_transform, pre_filter)

        out = fs.torch_load(self.processed_paths[0], map_location=map_location)
        if not isinstance(out, tuple) or len(out) < 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "Remove the 'processed/' directory and try again.")
        if len(out) not in (3, 4):
            raise RuntimeError(f"Unexpected checkpoint format: {len(out)} elements.")

        if len(out) == 3:
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        self.data = data if not isinstance(data, dict) else data_cls.from_dict(data)

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
            raise RuntimeError("data_list not provided.")
        data_list = self.data_list
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        self.data, self.slices = self.collate(data_list)
        self._data_list = None
        if not isinstance(self._data, Data):
            raise RuntimeError("Collated data is not a Data instance.")
        fs.torch_save(
            (self._data.to_dict(), self.slices, len(self.data_list), self._data.__class__),
            self.processed_paths[0],
        )


class EpiConcatMatrix:
    def __init__(self, fasta_file: str, cool_file: Union[str, None] = None,
                 resolution: int = 4096, region_length: str = "1M",
                 model: Optional[torch.nn.Module] = None) -> None:
        if region_length not in GRAPH_REGION_DICT:
            raise ValueError(f"region_length must be one of {list(GRAPH_REGION_DICT)}")
        if model is None:
            raise ValueError("model is required")

        self.fasta_file = FastaInterval(fasta_file)
        self.cool_file = cooler.Cooler(cool_file) if cool_file is not None else None
        self.resolution = resolution
        self.region_length = region_length
        self.region_length_int = GRAPH_REGION_DICT[region_length]
        self.model = model
        self.device = next(model.parameters()).device

    def _get_enformer_num(self) -> Tuple[int, int]:
        region_len = GRAPH_REGION_DICT[self.region_length]
        stride = ENFORMER_LENGTH - 320 * 128 * 2
        enformer_num = region_len // stride + (0 if region_len % stride == 0 else 1)
        return int(enformer_num), int(region_len // 128), stride

    def __call__(
            self, interval: Interval
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        chrom, start, end = interval.chrom, interval.start, interval.end
        start_s = start - 320 * 128
        enformer_num, _, stride = self._get_enformer_num()

        intervals = [Interval(chrom, start_s + i * stride, start_s + i * stride + ENFORMER_LENGTH)
                     for i in range(enformer_num)]
        inputs = [torch.tensor(self.fasta_file.extract(iv)) for iv in intervals]

        with torch.no_grad():
            try:
                outputs = [self.model(x.to(self.device)).t() for x in inputs]
            except Exception as e:
                raise RuntimeError(f"Enformer inference failed for {interval}: {e}")

        chip_track = torch.cat(outputs, dim=1)[:, :(interval.end - interval.start) // 128].cpu()

        if self.cool_file is not None:
            region_str = f"{chrom}:{start}-{end - self.resolution}"
            target = self.cool_file.matrix(balance=True).fetch(region_str)
            target = torch.tensor(np.nan_to_num(target, nan=0), dtype=torch.float32)
            return chip_track, target
        return chip_track


class CreateGraphData(EpiConcatMatrix):
    def __init__(self, interval_file: str, interval_file_columns: int = 3,
                 num_neighbors: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.interval_file = BedDataset(interval_file, bed_columns=interval_file_columns)
        self.num_neighbors = num_neighbors
        self.num_nodes = self.region_length_int // self.resolution

    def extract_data(self, interval: Interval) -> Data:
        edge_index = create_neighbor_graph(self.num_nodes, self.num_neighbors)

        if self.cool_file is not None:
            chip_track, target = super().__call__(interval)
        else:
            chip_track = super().__call__(interval)
            target = None

        chunks = torch.chunk(chip_track, self.num_nodes, dim=1)
        chip_track = torch.cat([c.reshape(1, -1) for c in chunks], dim=0)
        return Data(x=chip_track, edge_index=edge_index, target=target, region=str(interval))

    def __call__(self) -> List[Data]:
        data_list = []
        for idx in tqdm(range(len(self.interval_file)), desc="Extracting data"):
            interval, _ = self.interval_file[idx]
            data_list.append(self.extract_data(interval))
        return data_list
