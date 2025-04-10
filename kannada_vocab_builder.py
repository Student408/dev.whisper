import argparse
import base64
import collections
import os
import re
import tiktoken
from typing import Dict, List, Optional, Tuple

def tokenize_text(text: str) -> List[str]:
    """Simple tokenization for building initial vocabulary"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Split into basic tokens (customize this regex for Kannada if needed)
    tokens = re.findall(r'\S+|\s+', text)
    return tokens

def build_vocab_from_text(input_file: str, output_file: str, vocab_size: int = 50000, 
                          include_english: bool = True, english_vocab_path: Optional[str] = None):
    """
    Build a tiktoken vocabulary file from a text file
    
    Args:
        input_file: Path to input text file (Kannada corpus)
        output_file: Path to output .tiktoken file
        vocab_size: Maximum vocabulary size
        include_english: Whether to include basic English tokens
        english_vocab_path: Path to existing English vocabulary to merge with
    """
    print(f"Building vocabulary from {input_file}...")
    
    # Read Kannada text
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize and count frequencies
    tokens = tokenize_text(text)
    counter = collections.Counter(tokens)
    
    # Get most common tokens
    common_tokens = counter.most_common(vocab_size)
    print(f"Found {len(common_tokens)} unique tokens in the text")
    
    # If we want to include English vocabulary
    if include_english and english_vocab_path:
        english_tokens = {}
        with open(english_vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) == 2:
                        token, rank = parts
                        english_tokens[base64.b64decode(token)] = int(rank)
        
        # Merge vocabularies (prioritizing Kannada for overlapping tokens)
        print(f"Loaded {len(english_tokens)} English tokens")
        
        # Create final vocab, prioritizing most common Kannada tokens
        final_vocab = {}
        for token, count in common_tokens:
            token_bytes = token.encode('utf-8')
            final_vocab[token_bytes] = len(final_vocab)
            
        # Add English tokens that aren't already in the vocabulary
        english_items = sorted(english_tokens.items(), key=lambda x: x[1])
        for token_bytes, _ in english_items:
            if token_bytes not in final_vocab and len(final_vocab) < vocab_size:
                final_vocab[token_bytes] = len(final_vocab)
                
        print(f"Final vocabulary size: {len(final_vocab)}")
    else:
        # Just use the Kannada tokens
        final_vocab = {token.encode('utf-8'): rank for rank, (token, _) in enumerate(common_tokens)}
    
    # Write the vocabulary file in tiktoken format
    with open(output_file, 'w', encoding='utf-8') as f:
        for token_bytes, rank in sorted(final_vocab.items(), key=lambda x: x[1]):
            encoded = base64.b64encode(token_bytes).decode('ascii')
            f.write(f"{encoded} {rank}\n")
    
    print(f"Vocabulary written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a tiktoken vocabulary from Kannada text")
    parser.add_argument("input", help="Input text file path")
    parser.add_argument("output", help="Output tiktoken file path")
    parser.add_argument("--vocab-size", type=int, default=50000, help="Maximum vocabulary size")
    parser.add_argument("--include-english", action="store_true", help="Include English vocabulary")
    parser.add_argument("--english-vocab", help="Path to English vocabulary file")
    
    args = parser.parse_args()
    
    build_vocab_from_text(
        args.input, 
        args.output, 
        args.vocab_size, 
        args.include_english, 
        args.english_vocab
    )
