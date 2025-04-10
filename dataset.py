import os
import pandas as pd
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset

class WhisperDataset(Dataset):
    """Dataset for loading audio and transcript pairs from a CSV file."""
    
    def __init__(self, csv_file, tokenizer=None, sample_rate=16000, max_length=30*16000):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_length = max_length
        print(f"Loaded dataset with {len(self.data)} entries")

        # Validate file paths
        missing_files = []
        for idx, row in self.data.iterrows():
            if not os.path.exists(row['audio_filepath']):
                missing_files.append(row['audio_filepath'])
        
        if missing_files:
            print(f"Warning: {len(missing_files)} audio files not found.")
            print(f"First few missing files: {missing_files[:5]}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_path = self.data.iloc[idx]['audio_filepath']
        transcript = self.data.iloc[idx]['transcript']
        
        # Load audio file
        try:
            if os.path.exists(audio_path):
                waveform, sr = torchaudio.load(audio_path)
                
                # Convert to mono if necessary
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample if needed
                if sr != self.sample_rate:
                    waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
                
                # Ensure minimum length and handling for max length
                if waveform.shape[1] > self.max_length:
                    # Take the center portion
                    start = (waveform.shape[1] - self.max_length) // 2
                    waveform = waveform[:, start:start + self.max_length]
            else:
                # Create synthetic white noise for missing files
                print(f"File not found, creating synthetic audio: {audio_path}")
                waveform = torch.randn(1, self.sample_rate)  # 1 second of white noise
            
            # Ensure minimum length for processing
            if waveform.shape[1] < 16000:  # At least 1 second
                padding = torch.zeros(1, 16000 - waveform.shape[1])
                waveform = torch.cat([waveform, padding], dim=1)
                
        except Exception as e:
            print(f"Error loading audio {audio_path}: {str(e)}")
            # Create synthetic white noise audio in case of error
            waveform = torch.randn(1, self.sample_rate)  # 1 second of white noise
        
        # Tokenize transcript if tokenizer is provided
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(transcript)
                # Add end of text token
                tokens = torch.tensor(tokens + [self.tokenizer.eot])
            except Exception as e:
                print(f"Error tokenizing transcript '{transcript}': {str(e)}")
                # Create dummy tokens in case of failure
                tokens = torch.tensor([self.tokenizer.sot, self.tokenizer.eot])
        else:
            tokens = transcript
        
        return {
            "waveform": waveform.squeeze(0),
            "tokens": tokens,
            "transcript": transcript
        }

def collate_fn(batch, processor=None):
    """Custom collation function for variable length audio and tokens."""
    waveforms = [item["waveform"] for item in batch]
    tokens = [item["tokens"] for item in batch]
    transcripts = [item["transcript"] for item in batch]
    
    # Get sequence lengths
    audio_lengths = [w.shape[0] for w in waveforms]
    max_audio_len = max(audio_lengths)
    
    # Pad waveforms
    padded_waveforms = torch.zeros(len(batch), max_audio_len)
    for i, waveform in enumerate(waveforms):
        padded_waveforms[i, :waveform.shape[0]] = waveform
    
    # Process audio if processor is provided
    if processor is not None:
        # Convert waveforms to mel spectrograms
        mel_specs = processor(padded_waveforms)
    else:
        mel_specs = padded_waveforms
    
    # Pad tokens
    max_token_len = max(len(t) for t in tokens)
    padded_tokens = torch.ones(len(batch), max_token_len, dtype=torch.long) * -100  # -100 is ignored by CrossEntropyLoss
    
    for i, token_seq in enumerate(tokens):
        padded_tokens[i, :len(token_seq)] = token_seq
    
    return {
        "mel_specs": mel_specs,
        "tokens": padded_tokens,
        "transcripts": transcripts,
    }
