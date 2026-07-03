#!/usr/bin/env python3
"""
Generate embeddings for corpus using sentence-transformers.
Exports to Protocol Buffers format for C++ consumption.
"""
import argparse
import sys
from pathlib import Path
from typing import List

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "sentence-transformers"])
    import numpy as np
    from sentence_transformers import SentenceTransformer


def chunk_document(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Split document into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for corpus')
    parser.add_argument("--corpus-dir", required=True, help="Directory containing text files")
    parser.add_argument("--output", required=True, help="Output protobuf file path")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Sentence transformer model name")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in words")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap between chunks")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    corpus_data = []
    corpus_dir = Path(args.corpus_dir)

    print(f"Processing corpus from: {corpus_dir}")
    for doc_path in sorted(corpus_dir.glob("**/*.txt")):
        print(f"  Processing: {doc_path.name}")
        text = doc_path.read_text(encoding='utf-8')
        chunks = chunk_document(text, args.chunk_size, args.overlap)

        for i, chunk in enumerate(chunks):
            corpus_data.append({
                'id': f"{doc_path.stem}_{i}",
                'text': chunk,
                'source': str(doc_path),
                'position': i
            })

    if not corpus_data:
        print("No documents found!")
        return 1

    print(f"Generating embeddings for {len(corpus_data)} chunks...")
    texts = [d['text'] for d in corpus_data]
    embeddings = model.encode(texts, show_progress_bar=True)

    # Save in simple binary format for C++ consumption
    # Format: [int32: count] [int32: dimension] [float32: embeddings...]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        count = len(corpus_data)
        dim = embeddings.shape[1]

        # Write header
        f.write(count.to_bytes(4, byteorder='little'))
        f.write(dim.to_bytes(4, byteorder='little'))

        # Write embeddings as contiguous float32 array
        embeddings.astype(np.float32).tofile(f)

    print(f"Exported {count} chunks with {dim}-dimensional embeddings to {output_path}")
    print(f"Output size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
