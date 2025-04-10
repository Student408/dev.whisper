import argparse
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import re

from .model import Whisper, ModelDimensions
from .tokenizer import get_tokenizer
from .dataset import WhisperDataset, collate_fn
from .audio_utils import AudioProcessor

def pad_or_trim_mel_spectrogram(mel, target_length):
    """
    Pad or trim mel spectrogram to target length in the time dimension (last dimension).
    
    Args:
        mel: Mel spectrogram tensor of shape [batch_size, n_mels, time]
        target_length: Target length in the time dimension
        
    Returns:
        Padded or trimmed mel spectrogram
    """
    # Get current length
    current_length = mel.shape[-1]
    
    if current_length < target_length:
        # Pad if shorter
        pad_amount = target_length - current_length
        mel = F.pad(mel, (0, pad_amount))
    elif current_length > target_length:
        # Trim if longer
        mel = mel[..., :target_length]
    
    return mel

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate between reference and hypothesis texts.
    
    Args:
        reference: Reference text (string)
        hypothesis: Hypothesis text (string) from model prediction
        
    Returns:
        WER score (float), bounded between 0.0 and 1.0
    """
    # Normalize text: lowercase, remove punctuation, and split into words
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    ref_words = normalize_text(reference)
    hyp_words = normalize_text(hypothesis)
    
    # Edge case handling
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    
    # Initialize edit distance matrix
    d = [[0 for _ in range(len(hyp_words) + 1)] for _ in range(len(ref_words) + 1)]
    
    # Fill the matrix
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
        
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    
    # Calculate WER and bound it to maximum of 1.0
    return min(1.0, d[len(ref_words)][len(hyp_words)] / len(ref_words))

def calculate_cer(reference, hypothesis):
    """
    Calculate Character Error Rate between reference and hypothesis texts.
    
    Args:
        reference: Reference text (string)
        hypothesis: Hypothesis text (string) from model prediction
        
    Returns:
        CER score (float), bounded between 0.0 and 1.0
    """
    # Normalize text: lowercase and remove excess whitespace
    def normalize_text(text):
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    ref_chars = list(normalize_text(reference))
    hyp_chars = list(normalize_text(hypothesis))
    
    # Edge case handling
    if len(ref_chars) == 0:
        return 1.0 if len(hyp_chars) > 0 else 0.0
    
    # Initialize edit distance matrix
    d = [[0 for _ in range(len(hyp_chars) + 1)] for _ in range(len(ref_chars) + 1)]
    
    # Fill the matrix
    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j
        
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    
    # Calculate CER and bound it to maximum of 1.0
    return min(1.0, d[len(ref_chars)][len(hyp_chars)] / len(ref_chars))

def truncate_sequence(tokens, max_length):
    """
    Truncate a token sequence to the maximum allowed length.
    
    Args:
        tokens: Token tensor of shape [batch_size, seq_length]
        max_length: Maximum allowed sequence length
        
    Returns:
        Truncated token tensor
    """
    if tokens.shape[1] <= max_length:
        return tokens
    return tokens[:, :max_length]

def train(args):
    """Main training function for training Whisper model from scratch."""
    device = torch.device(args.device)
    
    # Load tokenizer with custom vocabulary
    print(f"Loading tokenizer with custom vocabulary from {args.custom_vocab_path}")
    tokenizer = get_tokenizer(
        multilingual=True,
        language=args.language,
        task="transcribe",
        custom_vocab_path=args.custom_vocab_path
    )
    
    # Create audio processor
    audio_processor = AudioProcessor(
        n_mels=80,
        n_fft=400,
        hop_length=160,
        sample_rate=16000
    )
    
    # Create dataset
    print(f"Loading training dataset from {args.train_csv}")
    train_dataset = WhisperDataset(
        args.train_csv,
        tokenizer=tokenizer,
        sample_rate=16000
    )
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, audio_processor),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create validation dataloader if provided
    val_loader = None
    if args.val_csv:
        print(f"Loading validation dataset from {args.val_csv}")
        val_dataset = WhisperDataset(
            args.val_csv,
            tokenizer=tokenizer,
            sample_rate=16000
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, audio_processor),
            num_workers=args.num_workers,
            pin_memory=True,
        )
    
    # Create a new model from scratch
    print(f"Creating new model with size {args.model_size}")
    
    # Define model dimensions based on size
    if args.model_size == "tiny":
        dims = ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=384,
            n_audio_head=6,
            n_audio_layer=4,
            n_vocab=tokenizer.encoding.n_vocab,
            n_text_ctx=448,
            n_text_state=384,
            n_text_head=6,
            n_text_layer=4
        )
    elif args.model_size == "base":
        dims = ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=512,
            n_audio_head=8,
            n_audio_layer=6,
            n_vocab=tokenizer.encoding.n_vocab,
            n_text_ctx=448,
            n_text_state=512,
            n_text_head=8,
            n_text_layer=6
        )
    elif args.model_size == "small":
        dims = ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=768,
            n_audio_head=12,
            n_audio_layer=12,
            n_vocab=tokenizer.encoding.n_vocab,
            n_text_ctx=448,
            n_text_state=768,
            n_text_head=12,
            n_text_layer=12
        )
    elif args.model_size == "medium":
        dims = ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=1024,
            n_audio_head=16,
            n_audio_layer=24,
            n_vocab=tokenizer.encoding.n_vocab,
            n_text_ctx=448,
            n_text_state=1024,
            n_text_head=16,
            n_text_layer=24
        )
    else:  # Default to base if size not recognized
        dims = ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=512,
            n_audio_head=8,
            n_audio_layer=6,
            n_vocab=tokenizer.encoding.n_vocab,
            n_text_ctx=448,
            n_text_state=512,
            n_text_head=8,
            n_text_layer=6
        )
    
    # Create new model
    model = Whisper(dims).to(device)
    print(f"Created new model with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    print(f"Vocabulary size: {tokenizer.encoding.n_vocab}")
    print(f"Maximum text context length: {model.dims.n_text_ctx}")
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps)
    
    # Training loop
    step = 0
    best_val_loss = float("inf")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_examples = 0
        
        # Statistics for sequence lengths
        max_seq_length = 0
        total_sequences = 0
        truncated_sequences = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in progress_bar:
            mel_specs = batch["mel_specs"].to(device)
            tokens = batch["tokens"].to(device)
            
            # Track sequence length statistics
            batch_max_len = tokens.shape[1]
            max_seq_length = max(max_seq_length, batch_max_len)
            total_sequences += tokens.shape[0]
            if batch_max_len > model.dims.n_text_ctx:
                truncated_sequences += tokens.shape[0]
            
            # Calculate the target length for the mel spectrogram
            # Need to ensure that after the encoder's strided convolution (stride=2),
            # the resulting length will match n_audio_ctx
            target_length = model.dims.n_audio_ctx * 2
            
            # Pad or trim mel spectrograms to match expected length
            mel_specs = pad_or_trim_mel_spectrogram(mel_specs, target_length)
            
            # Forward pass through the encoder
            audio_features = model.encoder(mel_specs)
            
            # Ensure tokens don't exceed maximum context length
            # Save 1 position for the SOT token
            max_token_length = model.dims.n_text_ctx - 1
            tokens_truncated = truncate_sequence(tokens, max_token_length)
            
            # Teacher forcing for decoder input (shift right)
            decoder_input = torch.cat([
                torch.ones((tokens_truncated.shape[0], 1), dtype=torch.long, device=device) * tokenizer.sot,
                tokens_truncated[:, :-1]
            ], dim=1)
            decoder_input = decoder_input.masked_fill(decoder_input == -100, tokenizer.eot)
            
            # Forward pass through the decoder
            logits = model.decoder(decoder_input, audio_features)
            
            # Calculate loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tokens_truncated.reshape(-1),
                ignore_index=-100,
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            
            # Update stats
            train_loss += loss.item() * tokens.shape[0]
            train_examples += tokens.shape[0]
            step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item(),
                "avg_loss": train_loss / train_examples,
                "step": step
            })
            
            # Save checkpoint
            if step % args.save_steps == 0:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{step}.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "step": step,
                    "dims": model.dims,
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
                
            # Validate
            if val_loader is not None and step % args.eval_steps == 0:
                val_loss, val_wer, val_cer = evaluate(model, val_loader, tokenizer, device)
                print(f"Validation loss: {val_loss:.4f}, WER: {val_wer:.4f}, CER: {val_cer:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(args.output_dir, "best_model.pt")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "step": step,
                        "dims": model.dims,
                        "val_loss": val_loss,
                        "val_wer": val_wer,
                        "val_cer": val_cer,
                    }, best_path)
                    print(f"Saved best model with val_loss {val_loss:.4f}, WER: {val_wer:.4f}, CER: {val_cer:.4f}")
        
        # Print sequence length statistics after each epoch
        print(f"\nSequence length statistics for epoch {epoch+1}:")
        print(f"  Maximum sequence length: {max_seq_length}")
        print(f"  Truncated sequences: {truncated_sequences}/{total_sequences} ({truncated_sequences/total_sequences*100:.2f}%)")
        
        # Validate at the end of each epoch
        if val_loader is not None:
            print(f"\nEvaluating at the end of epoch {epoch+1}/{args.num_epochs}...")
            val_loss, val_wer, val_cer = evaluate(model, val_loader, tokenizer, device)
            print(f"End of epoch {epoch+1} validation - Loss: {val_loss:.4f}, WER: {val_wer:.4f}, CER: {val_cer:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "step": step,
                    "dims": model.dims,
                    "val_loss": val_loss,
                    "val_wer": val_wer,
                    "val_cer": val_cer,
                    "epoch": epoch + 1,
                }, best_path)
                print(f"Saved best model with val_loss {val_loss:.4f}, WER: {val_wer:.4f}, CER: {val_cer:.4f}")
    
    # Final evaluation after training completion
    if val_loader is not None:
        print("\nFinal evaluation after training completion...")
        final_val_loss, final_val_wer, final_val_cer = evaluate(model, val_loader, tokenizer, device)
        print(f"Final validation - Loss: {final_val_loss:.4f}, WER: {final_val_wer:.4f}, CER: {final_val_cer:.4f}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": step,
        "dims": model.dims,
        "final_val_loss": final_val_loss if val_loader is not None else None,
        "final_val_wer": final_val_wer if val_loader is not None else None,
        "final_val_cer": final_val_cer if val_loader is not None else None,
    }, final_path)
    print(f"Training completed. Saved final model to {final_path}")

def evaluate(model, dataloader, tokenizer, device):
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_wer = 0.0
    total_cer = 0.0
    
    # Log some examples to help debug
    debug_examples = []
    
    with torch.no_grad():
        for batch in dataloader:
            mel_specs = batch["mel_specs"].to(device)
            tokens = batch["tokens"].to(device)
            
            # Calculate the target length for the mel spectrogram
            target_length = model.dims.n_audio_ctx * 2
            
            # Pad or trim mel spectrograms to match expected length
            mel_specs = pad_or_trim_mel_spectrogram(mel_specs, target_length)
            
            # Forward pass through the encoder
            audio_features = model.encoder(mel_specs)
            
            # Ensure tokens don't exceed maximum context length
            # Save 1 position for the SOT token
            max_token_length = model.dims.n_text_ctx - 1
            tokens_truncated = truncate_sequence(tokens, max_token_length)
            
            # Teacher forcing for decoder input
            decoder_input = torch.cat([
                torch.ones((tokens_truncated.shape[0], 1), dtype=torch.long, device=device) * tokenizer.sot,
                tokens_truncated[:, :-1]
            ], dim=1)
            decoder_input = decoder_input.masked_fill(decoder_input == -100, tokenizer.eot)
            
            # Forward pass through the decoder
            logits = model.decoder(decoder_input, audio_features)
            
            # Calculate loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tokens_truncated.reshape(-1),
                ignore_index=-100,
            )
            
            total_loss += loss.item() * tokens_truncated.shape[0]
            total_examples += tokens_truncated.shape[0]
            
            # Generate predictions for WER calculation
            # First, get the predicted token indices
            predictions = torch.argmax(logits, dim=-1)
            
            # Then decode each sequence into text
            for i in range(len(predictions)):
                # Get reference tokens (ground truth)
                ref_tokens = tokens[i].cpu().tolist()
                # Remove padding tokens (-100)
                ref_tokens = [t for t in ref_tokens if t != -100]
                # Remove tokens after eot if present
                if tokenizer.eot in ref_tokens:
                    ref_tokens = ref_tokens[:ref_tokens.index(tokenizer.eot)]
                
                # Get predicted tokens
                pred_tokens = predictions[i].cpu().tolist()
                # Remove tokens after eot if present
                if tokenizer.eot in pred_tokens:
                    pred_tokens = pred_tokens[:pred_tokens.index(tokenizer.eot)]
                
                # Decode both to text
                reference_text = tokenizer.decode(ref_tokens)
                predicted_text = tokenizer.decode(pred_tokens)
                
                # Calculate WER and CER for this example
                wer = calculate_wer(reference_text, predicted_text)
                cer = calculate_cer(reference_text, predicted_text)
                total_wer += wer
                total_cer += cer
                
                # Store some examples for debugging
                if len(debug_examples) < 5:
                    debug_examples.append({
                        "reference": reference_text,
                        "prediction": predicted_text,
                        "wer": wer,
                        "cer": cer
                    })
    
    avg_loss = total_loss / total_examples
    avg_wer = total_wer / total_examples
    avg_cer = total_cer / total_examples
    
    # Print debug examples
    print("\nDebug examples:")
    for i, example in enumerate(debug_examples):
        print(f"Example {i+1}:")
        print(f"  Reference: '{example['reference']}'")
        print(f"  Prediction: '{example['prediction']}'")
        print(f"  WER: {example['wer']:.4f}, CER: {example['cer']:.4f}")
    
    return avg_loss, avg_wer, avg_cer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Whisper model from scratch with custom vocabulary")
    
    # Data arguments
    parser.add_argument("--train-csv", required=True, help="Path to training CSV file")
    parser.add_argument("--val-csv", help="Path to validation CSV file (optional)")
    
    # Model arguments
    parser.add_argument("--model-size", default="base", 
                       choices=["tiny", "base", "small", "medium"], 
                       help="Model size for training from scratch")
    parser.add_argument("--custom-vocab-path", required=True, help="Path to custom vocabulary file")
    parser.add_argument("--language", default="kn", help="Language code (e.g., 'kn' for Kannada)")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--save-steps", type=int, default=1000, help="Steps between checkpoints")
    parser.add_argument("--eval-steps", type=int, default=500, help="Steps between evaluations")
    parser.add_argument("--output-dir", default="./checkpoints", help="Output directory")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    args = parser.parse_args()
    train(args)
