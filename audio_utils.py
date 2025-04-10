import numpy as np
import torch
import torch.nn.functional as F
from typing import Union
import librosa

def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    sample_rate: int = 16000,
    padding: int = 0,
):
    """
    Convert audio waveform to log-mel spectrogram.
    
    Args:
        audio: Audio waveform as numpy array or path to audio file
        n_mels: Number of mel filterbanks
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        sample_rate: Audio sample rate
        padding: Amount of padding to add for consistent shape
    
    Returns:
        torch.Tensor: Log-mel spectrogram of shape (n_mels, T)
    """
    if isinstance(audio, str):
        audio, _ = librosa.load(audio, sr=sample_rate)
    
    if not isinstance(audio, torch.Tensor):
        audio = torch.from_numpy(audio).float()
    
    if audio.dim() == 2:
        audio = audio.mean(dim=0)
    
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    
    window = torch.hann_window(n_fft).to(audio.device)
    stft = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    
    magnitudes = torch.abs(stft) ** 2
    
    # Create mel filter bank
    mel_filters = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
    )
    mel_filters = torch.from_numpy(mel_filters).to(audio.device)
    
    mel_spec = torch.matmul(mel_filters, magnitudes)
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    
    return log_spec

class AudioProcessor:
    """Processes audio waveforms to mel spectrograms for Whisper model."""
    
    def __init__(self, n_mels=80, n_fft=400, hop_length=160, sample_rate=16000, target_length=3000):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.target_length = target_length  # Target length for the mel spectrogram timesteps (before downsampling)
        
    def __call__(self, waveform):
        """Process a batch of waveforms to mel spectrograms with consistent shape."""
        specs = []
        
        for i in range(waveform.shape[0]):
            # Calculate expected length of spectrogram
            audio_length = waveform[i].shape[0]
            expected_mel_len = 1 + (audio_length - self.n_fft) // self.hop_length
            
            # Calculate padding needed to make the spectrogram have consistent length
            padding_needed = 0
            if expected_mel_len < self.target_length // 2:  # Target before downsampling by conv
                samples_needed = (self.target_length * 2 - expected_mel_len) * self.hop_length
                padding_needed = samples_needed
            
            spec = log_mel_spectrogram(
                waveform[i], 
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                sample_rate=self.sample_rate,
                padding=padding_needed
            )
            
            # Make sure it's at least the target length
            if spec.shape[1] < self.target_length // 2:
                pad_amount = self.target_length // 2 - spec.shape[1]
                spec = F.pad(spec, (0, pad_amount))
            
            # Limit to target length if somehow longer 
            if spec.shape[1] > self.target_length:
                spec = spec[:, :self.target_length]
            
            specs.append(spec)
            
        # Stack all spectrograms and ensure the batch has the right shape
        result = torch.stack(specs)
        return result
