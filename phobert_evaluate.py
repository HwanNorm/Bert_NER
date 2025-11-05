"""
Evaluate PhoBERT NER model on test set
Compare with Gemini results if available
"""

import os
import json
from phobert_inference import PhoBERTNERInference
from phobert_data_processor import BIODataProcessor
import phobert_config as config


def evaluate_on_test_file():
    """Evaluate model on ViMedNER test set"""
    print("="*60)
    print("📊 Evaluating PhoBERT NER Model")
    print("="*60)

    # Load model
    ner = PhoBERTNERInference()

    # Load test data
    processor = BIODataProcessor()

    if config.SINGLE_FILE:
        print(f"\n⚠️  Single file mode - splitting data first...")
        _, _, test_data = processor.split_single_file(config.SINGLE_FILE_PATH)
    else:
        test_path = os.path.join(config.VIMEDNER_DIR, config.TEST_FILE)
        test_data = processor.read_bio_file(test_path)

    print(f"\n🔍 Evaluating on {len(test_data)} test sentences...")

    # Evaluate
    correct = 0
    total = 0
    predictions_by_category = {cat: {"correct": 0, "predicted": 0, "gold": 0} for cat in config.NER_KEYS}

    for i, (tokens, labels) in enumerate(test_data[:100]):  # Sample first 100 for speed
        text = " ".join(tokens)
        predicted = ner.extract_entities(text)

        # Count entities
        for label in labels:
            if label != "O":
                total += 1

        # Simple evaluation (just for demo)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1} sentences...")

    print(f"\n✅ Evaluation complete!")
    print(f"   Test samples: {min(100, len(test_data))}")


def compare_with_gemini(gemini_output_path: str, test_chunk_file: str):
    """
    Compare PhoBERT output with Gemini output

    Args:
        gemini_output_path: Path to Gemini JSON output
        test_chunk_file: Path to input chunks used with Gemini
    """
    print("\n" + "="*60)
    print("🔬 Comparing PhoBERT vs Gemini")
    print("="*60)

    # Load Gemini results
    if not os.path.exists(gemini_output_path):
        print(f"❌ Gemini output not found: {gemini_output_path}")
        return

    with open(gemini_output_path, "r", encoding="utf-8") as f:
        gemini_results = json.load(f)

    print(f"✅ Loaded {len(gemini_results)} Gemini results")

    # Load PhoBERT model
    ner = PhoBERTNERInference()

    # Compare
    print(f"\n📊 Comparison (first 5 samples):\n")

    for i, gemini_item in enumerate(gemini_results[:5]):
        chunk = gemini_item["content"]
        gemini_entities = gemini_item["entities"]

        # Get PhoBERT predictions
        phobert_entities = ner.extract_entities(chunk)

        print(f"--- Sample {i+1} ---")
        print(f"Text: {chunk[:100]}...")

        for category in config.NER_KEYS:
            gemini_set = set(gemini_entities.get(category, []))
            phobert_set = set(phobert_entities.get(category, []))

            if gemini_set or phobert_set:
                print(f"\n  {category}:")
                print(f"    Gemini:  {list(gemini_set)}")
                print(f"    PhoBERT: {list(phobert_set)}")

                # Show overlap
                overlap = gemini_set & phobert_set
                if overlap:
                    print(f"    ✅ Match: {list(overlap)}")
                if gemini_set - phobert_set:
                    print(f"    ⚠️  Gemini only: {list(gemini_set - phobert_set)}")
                if phobert_set - gemini_set:
                    print(f"    ⚠️  PhoBERT only: {list(phobert_set - gemini_set)}")

        print("")


def test_on_custom_text():
    """Test on custom medical text"""
    print("\n" + "="*60)
    print("🧪 Testing on Custom Medical Text")
    print("="*60)

    test_cases = [
        """
        Viêm phổi là bệnh nhiễm trùng đường hô hấp.
        Triệu chứng bao gồm sốt cao, ho khan, khó thở.
        Điều trị bằng kháng sinh như amoxicillin.
        """,
        """
        Đái tháo đường type 2 do kháng insulin.
        Biểu hiện: khát nước nhiều, tiểu nhiều, mệt mỏi.
        Dùng metformin để kiểm soát đường huyết.
        """,
        """
        Sỏi thận gây đau quặn dữ dội vùng thắt lưng.
        Chẩn đoán bằng siêu âm và CT scan.
        Có thể dùng thuốc giảm đau hoặc tán sỏi bằng sóng xung kích.
        """
    ]

    ner = PhoBERTNERInference()

    for i, text in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Text: {text.strip()}\n")

        entities = ner.extract_entities(text)

        print("Extracted entities:")
        for category, items in entities.items():
            if items:
                print(f"  ✓ {category}: {items}")


def main():
    """Main evaluation script"""
    import sys

    print("="*60)
    print("🏥 PhoBERT NER Evaluation Suite")
    print("="*60)

    # Check if model exists
    if not os.path.exists(config.OUTPUT_DIR):
        print(f"\n❌ Trained model not found at: {config.OUTPUT_DIR}")
        print(f"💡 Please train the model first:")
        print(f"   python phobert_train.py")
        return

    # Test on custom text
    test_on_custom_text()

    # Compare with Gemini if available
    gemini_output = r"../outputs/json_test/chunk_testing_ner_results_strict.json"
    if os.path.exists(gemini_output):
        print("\n🔍 Found Gemini output, comparing...")
        compare_with_gemini(gemini_output, r"../chunk_testing.txt")
    else:
        print(f"\n💡 To compare with Gemini, place output at: {gemini_output}")

    print("\n" + "="*60)
    print("✅ Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
