import numpy as np
import torch
import torch.nn.functional as F
from typing import Union

def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    sample_rate: int = 16000,
):
    """
    Convert audio waveform to log-mel spectrogram.
    
    Args:
        audio: Audio waveform as numpy array or path to audio file
        n_mels: Number of mel filterbanks
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        sample_rate: Audio sample rate
    
    Returns:
        torch.Tensor: Log-mel spectrogram of shape (n_mels, T)
    """
    if isinstance(audio, str):
        import librosa
        audio, _ = librosa.load(audio, sr=sample_rate)
    
    if not isinstance(audio, torch.Tensor):
        audio = torch.from_numpy(audio)
    
    window = torch.hann_window(n_fft)
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
    import librosa
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
    
    def __init__(self, n_mels=80, n_fft=400, hop_length=160, sample_rate=16000):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
    def __call__(self, waveform):
        """Process a batch of waveforms to mel spectrograms."""
        if waveform.dim() == 1:
            # Single waveform
            return log_mel_spectrogram(
                waveform,
                n_mels=self.n_mels,
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                sample_rate=self.sample_rate
            ).unsqueeze(0)  # Add batch dimension
        else:
            # Batch of waveforms
            specs = []
            for i in range(waveform.shape[0]):
                spec = log_mel_spectrogram(
                    waveform[i], 
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    sample_rate=self.sample_rate
                )
                specs.append(spec)
            return torch.stack(specs)
