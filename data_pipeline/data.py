import torch
import numpy as np
import pyBigWig
from torch.utils.data import Dataset
from pyfaidx import Fasta
from kipoiseq import Interval
from kipoiseq.dataloaders import BedDataset
from typing import List, Tuple

ENFORMER_LENGTH = 196_608


class FastaInterval:
    def __init__(self, fasta_file: str) -> None:
        self.fasta = Fasta(fasta_file)

    def extract(self, interval: Interval) -> np.ndarray:
        seq = str(self.fasta[interval.chrom][interval.start:interval.end]).upper()
        return _one_hot(seq)


def _one_hot(seq: str) -> np.ndarray:
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            arr[i, mapping[base]] = 1.0
    return arr


class EnformerSignalDataset(Dataset):
    def __init__(self, interval_file: str, fasta_file: str,
                 bigwig_files: List[str], num_bins: int = 896) -> None:
        self.fasta = FastaInterval(fasta_file)
        self.bigwig_files = bigwig_files
        self.num_bins = num_bins
        bed = BedDataset(interval_file, bed_columns=3)
        self.intervals = [bed[i][0] for i in range(len(bed))]

    def __len__(self) -> int:
        return len(self.intervals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        interval = self.intervals[idx]
        seq = self.fasta.extract(interval)
        sequence = torch.tensor(seq, dtype=torch.float32)

        signals = []
        for bw_path in self.bigwig_files:
            with pyBigWig.open(bw_path) as bw:
                sig = bw.stats(interval.chrom, interval.start, interval.end,
                               nBins=self.num_bins, type="mean")
            sig = np.nan_to_num(np.array(sig, dtype=np.float32), nan=0.0)
            signals.append(sig)

        target = torch.tensor(np.stack(signals, axis=-1), dtype=torch.float32)
        return sequence, target
