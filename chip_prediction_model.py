import torch
import torch.nn as nn
import torch.nn.functional as F
from enformer_pytorch import Enformer, from_pretrained
from enformer_pytorch.config_enformer import EnformerConfig
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pyBigWig


ENFORMER_CONFIG = EnformerConfig()

class ChIPPredictionModel(nn.Module):
    """
    Model specifically for predicting epigenomic histone ChIP-seq signals.
    Based on the Enformer architecture, but optimized for ChIP-seq signals.
    """
    def __init__(self, 
                 enformer_model: Optional[Enformer] = None,
                 sequence_length: int = 196608,
                 target_length: int = 896,
                 dropout_rate: float = 0.1,
                 use_pretrained: bool = True,
                 finetune_model: bool = False,
                 num_chip_tracks: int = 100):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.target_length = target_length
        self.num_chip_tracks = num_chip_tracks
        self.num_chip_tracks = num_chip_tracks
        
        # Load pretrained Enformer model
        if enformer_model is not None:
            self.enformer = enformer_model
        elif use_pretrained:
            self.enformer = from_pretrained('EleutherAI/enformer-official-rough', 
                                          target_length=target_length,
                                          dropout_rate=dropout_rate,
                                          use_tf_gamma=True)
        else:
            # Create a new Enformer model
            self.enformer = Enformer(ENFORMER_CONFIG)


        if finetune_model:
            self.enformer.use_tf_gamma = False

            output = torch.nn.ModuleDict({
                'human': torch.nn.Sequential(
                    torch.nn.Linear(in_features=3072, out_features=self.num_chip_tracks, bias=True),
                    torch.nn.Softplus(beta=1.0, threshold=20.0)),
                'mouse': torch.nn.Sequential(
                    torch.nn.Linear(in_features=3072, out_features=self.num_chip_tracks, bias=True),
                    torch.nn.Softplus(beta=1.0, threshold=20.0))
            })

            self.enformer._heads = output
            
    
    def forward(self, 
                sequence: torch.Tensor,
                return_embeddings: bool = False,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            sequence: Input DNA sequence (batch_size, sequence_length, 4)
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary containing prediction results
        """
        
        # Get embeddings from Enformer
        if return_embeddings:
            enformer_output, embeddings = self.enformer(sequence, return_embeddings=True, **kwargs)
        else:
            enformer_output = self.enformer(sequence, **kwargs)

        
        if return_embeddings:
            return {
                'chip_predictions': enformer_output,
                'embeddings': embeddings
            }
        else:
            return {
                'chip_predictions': enformer_output,
                # 'embeddings': torch.empty(0)  # Return empty tensor instead of None for type consistency
            }


    def predict_chip_signal(self, 
                           sequence: torch.Tensor,
                           return_embeddings: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict ChIP-seq signal
        
        Args:
            sequence: DNA sequence
            chip_track_names: List of ChIP-seq track names
            
        Returns:
            ChIP-seq signal predictions (batch_size, target_length, num_chip_tracks)
        """

        with torch.no_grad():

            if return_embeddings:
                output = self.forward(sequence, return_embeddings=True)

                chip_predictions = output['chip_predictions']
                embeddings = output['embeddings']

                return chip_predictions, embeddings
            else:
                output = self.forward(sequence)
                chip_predictions = output['chip_predictions']
                return chip_predictions


# Prepare enformer train data set
# Step 1: input sequence
# step 2: input region chip track signal
# step 3: Crate training dataset input, and target




class ChIPDataProcessor:
    """
    Utility class for processing ChIP-seq data
    """
    
    def __init__(self, 
                 fasta_file: str,
                 targets_file: str,
                 chip_tracks: Optional[List[str]] = None):
        """
        Initialize data processor
        
        Args:
            fasta_file: Path to genome reference file
            targets_file: Path to targets file
            chip_tracks: List of ChIP-seq tracks to use
        """
        self.fasta_extractor = pyfaidx.Fasta(fasta_file)
        self.targets_df = pd.read_csv(targets_file, sep='\t')
        
        # Filter ChIP-seq tracks
        if chip_tracks is None:
            self.chip_tracks = self.targets_df[self.targets_df['description'].str.contains('CHIP')]
        else:
            self.chip_tracks = self.targets_df[self.targets_df['description'].isin(chip_tracks)]
        
        self.chip_track_indices = self.chip_tracks.index.tolist()
        
    def extract_sequence(self, interval: Interval) -> str:
        """Extract DNA sequence"""
        return str(self.fasta_extractor[interval.chrom][interval.start:interval.end].seq).upper()
    
    def one_hot_encode(self, sequence: str) -> torch.Tensor:
        """Convert DNA sequence to one-hot encoding"""
        return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)
    
    def load_chip_signal(self, 
                         bigwig_file: str, 
                         interval: Interval, 
                         num_bins: int = 896) -> np.ndarray:
        """Load ChIP-seq signal from BigWig file"""
        with pyBigWig.open(bigwig_file) as bw:
            signal = np.array(bw.stats(interval.chrom, 
                                     interval.start, 
                                     interval.end, 
                                     nBins=num_bins, 
                                     type="mean"))
            # Handle NaN values
            signal = np.nan_to_num(signal, nan=0.0)
            return signal
    
    def create_training_sample(self, 
                              interval: Interval,
                              chip_signals: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Create training sample"""
        # Extract sequence
        sequence = self.extract_sequence(interval)
        sequence_one_hot = self.one_hot_encode(sequence)
        
        # Prepare ChIP signal
        chip_target = np.zeros((self.target_length, len(self.chip_track_indices)))
        for i, track_idx in enumerate(self.chip_track_indices):
            if track_idx in chip_signals:
                chip_target[:, i] = chip_signals[track_idx]
        
        return {
            'sequence': torch.tensor(sequence_one_hot[np.newaxis], dtype=torch.float32),
            'chip_target': torch.tensor(chip_target[np.newaxis], dtype=torch.float32)
        }

class ChIPTrainer:
    """
    ChIP-seq model trainer
    """
    
    def __init__(self, 
                 model: ChIPPredictionModel,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def train_step(self, 
                   sequence: torch.Tensor, 
                   chip_target: torch.Tensor) -> float:
        """Single training step"""
        self.model.train()
        
        sequence = sequence.to(self.device)
        chip_target = chip_target.to(self.device)
        
        # Forward pass
        output = self.model(sequence)
        predictions = output['chip_predictions']
        
        # Compute loss (using MSE loss)
        loss = F.mse_loss(predictions, chip_target)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, 
                 sequence: torch.Tensor, 
                 chip_target: torch.Tensor) -> Tuple[float, float]:
        """Validation"""
        self.model.eval()
        
        with torch.no_grad():
            sequence = sequence.to(self.device)
            chip_target = chip_target.to(self.device)
            
            output = self.model(sequence)
            predictions = output['chip_predictions']
            
            # Compute loss and correlation coefficient
            loss = F.mse_loss(predictions, chip_target)
            
            # Compute Pearson correlation coefficient
            pred_flat = predictions.view(-1)
            target_flat = chip_target.view(-1)
            correlation = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]
            
            return loss.item(), correlation.item()
    
    def save_model(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Usage example
def create_chip_model(num_chip_tracks: int = 100) -> ChIPPredictionModel:
    """Create ChIP-seq prediction model"""
    model = ChIPPredictionModel(
        num_chip_tracks=num_chip_tracks,
        use_pretrained=True,
        dropout_rate=0.1
    )
    return model

def main():
    """Main function example"""
    # Create model
    model = create_chip_model(num_chip_tracks=50)
    
    # Create data processor
    processor = ChIPDataProcessor(
        fasta_file="./data/hg38.fa",
        targets_file="./data/targets_human.txt"
    )
    
    # Create trainer
    trainer = ChIPTrainer(model)
    
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of ChIP tracks: {len(processor.chip_track_indices)}")
    
    # Example: Predict ChIP-seq signal for a region
    interval = kipoiseq.Interval('chr11', 35082742, 35197430)
    sequence = processor.extract_sequence(interval.resize(196608))
    sequence_one_hot = processor.one_hot_encode(sequence)
    sequence_tensor = torch.tensor(sequence_one_hot[np.newaxis], dtype=torch.float32)
    
    # Prediction
    with torch.no_grad():
        predictions = model.predict_chip_signal(sequence_tensor)
        print(f"Prediction shape: {predictions.shape}")
        print(f"Prediction value range: {predictions.min().item():.4f} - {predictions.max().item():.4f}")

if __name__ == "__main__":
    main() 