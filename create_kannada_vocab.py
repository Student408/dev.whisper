import argparse
import os
from kannada_vocab_builder import build_vocab_from_text  # Removed the dot import

def main():
    parser = argparse.ArgumentParser(description="Create and use Kannada vocabulary with Whisper")
    parser.add_argument("input", help="Input Kannada text file")
    parser.add_argument("--output", default="assets/multilingual_kn.tiktoken", 
                        help="Output vocabulary file path")
    parser.add_argument("--merge-english", action="store_true", 
                        help="Merge with English vocabulary")
    parser.add_argument("--english-vocab", default="assets/multilingual.tiktoken",
                        help="Path to English vocabulary file")
    parser.add_argument("--extract-from-csv", help="Extract Kannada text from training CSV")
    
    args = parser.parse_args()
    
    # Extract text from CSV if specified
    if args.extract_from_csv:
        import pandas as pd
        print(f"Extracting text from CSV file: {args.extract_from_csv}")
        df = pd.read_csv(args.extract_from_csv)
        
        # Extract transcripts and save to a temporary file
        temp_file = "temp_kannada_text.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            for transcript in df['transcript']:
                f.write(transcript + '\n')
        
        args.input = temp_file
        print(f"Extracted {len(df)} transcripts to temporary file")
    
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
