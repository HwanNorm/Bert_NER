"""
PhoBERT NER Production Pipeline
Drop-in replacement for Gemini-based NER_extractor.py

Usage: Same as NER_extractor.py but uses PhoBERT instead of Gemini
- No API key needed
- No API costs
- Faster inference
- 100% offline
"""

import os
import json
import time
from typing import List, Dict
from phobert_inference import PhoBERTNERInference
import phobert_config as config

# ======== 🔧 USER SETTINGS ========
INPUT_FILE = r"../chunk_testing.txt"
OUTPUT_DIR = r"../outputs/json_test"
MODEL_PATH = config.OUTPUT_DIR  # Path to trained PhoBERT model
CHUNK_SIZE = 400
OVERLAP = 40
# =================================


def load_text(path: str) -> str:
    """Load text file"""
    return open(path, "r", encoding="utf-8").read()


def chunk_text(text: str, size=400, overlap=40) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(len(words), start + size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += size - overlap
    return chunks


def process_with_phobert(input_file: str, output_dir: str, model_path: str):
    """
    Process text file with PhoBERT NER

    Args:
        input_file: Path to input text file
        output_dir: Directory to save outputs
        model_path: Path to trained PhoBERT model
    """
    print("="*60)
    print("🏥 PhoBERT Medical NER Pipeline")
    print("="*60)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Trained model not found at: {model_path}")
        print(f"💡 Please train the model first:")
        print(f"   python phobert_train.py")
        return

    # Load model
    print(f"\n📦 Loading PhoBERT model from: {model_path}")
    ner = PhoBERTNERInference(model_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load and chunk text
    print(f"\n📄 Loading text from: {input_file}")
    text = load_text(input_file)
    chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
    print(f"✅ Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={OVERLAP})")

    # Process chunks
    print(f"\n🔍 Processing chunks with PhoBERT...\n")
    results = []
    base = os.path.splitext(os.path.basename(input_file))[0]

    start_time = time.time()

    for i, chunk in enumerate(chunks):
        # Cache file for this chunk
        cache_file = os.path.join(output_dir, f"{base}_chunk{i}_phobert.json")

        if os.path.exists(cache_file):
            # Load cached result
            with open(cache_file, "r", encoding="utf-8") as f:
                entities = json.load(f)
            print(f"✓ Chunk {i+1}/{len(chunks)} (cached)")
        else:
            # Extract entities
            entities = ner.extract_entities(chunk)

            # Save cache
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(entities, f, ensure_ascii=False, indent=2)

            print(f"✓ Chunk {i+1}/{len(chunks)} - Found: {sum(len(v) for v in entities.values())} entities")

        # Build result in same format as Gemini output
        result = {
            "content": chunk.strip(),
            "entities": entities
        }
        results.append(result)

    elapsed = time.time() - start_time

    # Save final results
    output_path = os.path.join(output_dir, f"{base}_ner_results_phobert.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ Processing Complete!")
    print(f"{'='*60}")
    print(f"📊 Statistics:")
    print(f"   Total chunks: {len(results)}")
    print(f"   Processing time: {elapsed:.2f}s")
    print(f"   Speed: {len(results)/elapsed:.2f} chunks/sec")

    # Count entities by category
    entity_counts = {cat: 0 for cat in config.NER_KEYS}
    for result in results:
        for category, items in result["entities"].items():
            entity_counts[category] += len(items)

    print(f"\n📋 Entities extracted:")
    for category, count in entity_counts.items():
        print(f"   {category}: {count}")

    print(f"\n💾 Output saved to:")
    print(f"   {output_path}")

    # Compare with Gemini output if exists
    gemini_output = output_path.replace("_phobert.json", "_strict.json")
    if os.path.exists(gemini_output):
        print(f"\n🔬 Comparing with Gemini output...")
        compare_outputs(output_path, gemini_output)


def compare_outputs(phobert_path: str, gemini_path: str):
    """Compare PhoBERT and Gemini outputs"""
    with open(phobert_path, "r", encoding="utf-8") as f:
        phobert_results = json.load(f)
    with open(gemini_path, "r", encoding="utf-8") as f:
        gemini_results = json.load(f)

    print(f"\n📊 Comparison Summary:")
    print(f"   Chunks: PhoBERT={len(phobert_results)}, Gemini={len(gemini_results)}")

    # Compare entity counts
    phobert_counts = {cat: 0 for cat in config.NER_KEYS}
    gemini_counts = {cat: 0 for cat in config.NER_KEYS}

    for result in phobert_results:
        for cat, items in result["entities"].items():
            phobert_counts[cat] += len(items)

    for result in gemini_results:
        for cat, items in result["entities"].items():
            gemini_counts[cat] += len(items)

    print(f"\n   Entity counts by category:")
    for cat in config.NER_KEYS:
        p_count = phobert_counts[cat]
        g_count = gemini_counts[cat]
        diff = p_count - g_count
        symbol = "+" if diff > 0 else ""
        print(f"   {cat:15} PhoBERT: {p_count:3d} | Gemini: {g_count:3d} | Diff: {symbol}{diff}")


def main():
    """Main pipeline"""
    process_with_phobert(INPUT_FILE, OUTPUT_DIR, MODEL_PATH)


if __name__ == "__main__":
    main()
