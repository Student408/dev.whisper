import argparse
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import Whisper
from .tokenizer import get_tokenizer
from .dataset import WhisperDataset, collate_fn
from .audio_utils import AudioProcessor

def train(args):
    """Main training function for fine-tuning Whisper with custom vocabulary."""
    device = torch.device(args.device)
    
    # Load tokenizer with custom vocabulary
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
    
    # Load model
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path does not exist: {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model = Whisper(checkpoint["dims"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps)
    
    # Training loop
    step = 0
    best_val_loss = float("inf")
    
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_examples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in progress_bar:
            mel_specs = batch["mel_specs"].to(device)
            tokens = batch["tokens"].to(device)
            
            # Forward pass through the encoder
            audio_features = model.encoder(mel_specs)
            
            # Teacher forcing for decoder input (shift right)
            decoder_input = torch.cat([
                torch.ones((tokens.shape[0], 1), dtype=torch.long, device=device) * tokenizer.sot,
                tokens[:, :-1]
            ], dim=1)
            decoder_input = decoder_input.masked_fill(decoder_input == -100, tokenizer.eot)
            
            # Forward pass through the decoder
            logits = model.decoder(decoder_input, audio_features)
            
            # Calculate loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tokens.reshape(-1),
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
                os.makedirs(args.output_dir, exist_ok=True)
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
                val_loss = evaluate(model, val_loader, tokenizer, device)
                print(f"Validation loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(args.output_dir, "best_model.pt")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "step": step,
                        "dims": model.dims,
                        "val_loss": val_loss,
                    }, best_path)
                    print(f"Saved best model with val_loss {val_loss:.4f}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": step,
        "dims": model.dims,
    }, final_path)
    print(f"Training completed. Saved final model to {final_path}")

def evaluate(model, dataloader, tokenizer, device):
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            mel_specs = batch["mel_specs"].to(device)
            tokens = batch["tokens"].to(device)
            
            # Forward pass through the encoder
            audio_features = model.encoder(mel_specs)
            
            # Teacher forcing for decoder input
            decoder_input = torch.cat([
                torch.ones((tokens.shape[0], 1), dtype=torch.long, device=device) * tokenizer.sot,
                tokens[:, :-1]
            ], dim=1)
            decoder_input = decoder_input.masked_fill(decoder_input == -100, tokenizer.eot)
            
            # Forward pass through the decoder
            logits = model.decoder(decoder_input, audio_features)
            
            # Calculate loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tokens.reshape(-1),
                ignore_index=-100,
            )
            
            total_loss += loss.item() * tokens.shape[0]
            total_examples += tokens.shape[0]
    
    return total_loss / total_examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Whisper model with custom vocabulary")
    
    # Data arguments
    parser.add_argument("--train-csv", required=True, help="Path to training CSV file")
    parser.add_argument("--val-csv", help="Path to validation CSV file (optional)")
    
    # Model arguments
    parser.add_argument("--model-path", required=True, help="Path to pretrained model")
    parser.add_argument("--custom-vocab-path", required=True, help="Path to custom vocabulary file")
    parser.add_argument("--language", default="kn", help="Language code (e.g., 'kn' for Kannada)")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
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
