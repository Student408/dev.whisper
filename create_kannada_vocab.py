import argparse
import os
from .kannada_vocab_builder import build_vocab_from_text

def main():
    parser = argparse.ArgumentParser(description="Create and use Kannada vocabulary with Whisper")
    parser.add_argument("input", help="Input Kannada text file")
    parser.add_argument("--output", default="assets/multilingual_kn.tiktoken", 
                        help="Output vocabulary file path")
    parser.add_argument("--merge-english", action="store_true", 
                        help="Merge with English vocabulary")
    parser.add_argument("--english-vocab", default="assets/multilingual.tiktoken",
                        help="Path to English vocabulary file")
    
    args = parser.parse_args()
    
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Build the vocabulary file
    build_vocab_from_text(
        args.input, 
        args.output, 
        include_english=args.merge_english,
        english_vocab_path=args.english_vocab
    )
    
    print("\nYou can now use this vocabulary by passing the path to get_tokenizer:")
    print("from whisper.tokenizer import get_tokenizer")
    print(f"tokenizer = get_tokenizer(multilingual=True, language='kn', custom_vocab_path='{args.output}')")

if __name__ == "__main__":
    main()
