"""
PhoBERT NER Inference Script
Use trained PhoBERT model for entity extraction
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict
import phobert_config as config


class PhoBERTNERInference:
    """PhoBERT NER Inference Engine"""

    def __init__(self, model_path: str = None):
        """
        Initialize inference engine

        Args:
            model_path: Path to trained model (default: from config)
        """
        self.model_path = model_path or config.OUTPUT_DIR
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🤖 Loading PhoBERT NER model from: {self.model_path}")
        print(f"🔧 Device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        # Load label mappings
        label_map_path = os.path.join(self.model_path, "label_mappings.json")
        with open(label_map_path, "r", encoding="utf-8") as f:
            mappings = json.load(f)
            self.label2id = mappings["label2id"]
            self.id2label = {int(k): v for k, v in mappings["id2label"].items()}

        # Entity type mapping (BIO tags -> your categories)
        self.bio_to_category = {}
        for bio_tag, category in config.ENTITY_MAPPING.items():
            self.bio_to_category[f"B-{bio_tag}"] = category
            self.bio_to_category[f"I-{bio_tag}"] = category

        print(f"✅ Model loaded successfully!")
        print(f"📋 Categories: {list(config.NER_KEYS)}")

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text

        Args:
            text: Input Vietnamese text

        Returns:
            Dictionary with entity categories as keys and lists of entities as values
        """
        # Tokenize
        words = text.split()
        inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)

        # Decode predictions
        predicted_labels = []
        word_ids = inputs.word_ids(0)
        previous_word_idx = None

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                label_id = predictions[0][idx].item()
                predicted_labels.append(self.id2label[label_id])
            previous_word_idx = word_idx

        # Extract entities from BIO tags
        entities = self._bio_to_entities(words, predicted_labels)

        # Group by category
        result = {category: [] for category in config.NER_KEYS}

        for entity_text, bio_tag in entities:
            if bio_tag in self.bio_to_category:
                category = self.bio_to_category[bio_tag]
                if entity_text not in result[category]:
                    result[category].append(entity_text)

        return result

    def _bio_to_entities(self, words: List[str], labels: List[str]) -> List[tuple]:
        """
        Convert BIO tags to entity spans

        Args:
            words: List of words
            labels: List of BIO labels

        Returns:
            List of (entity_text, entity_type) tuples
        """
        entities = []
        current_entity = []
        current_tag = None

        for word, label in zip(words, labels):
            if label.startswith("B-"):
                # Save previous entity
                if current_entity:
                    entity_text = " ".join(current_entity)
                    entities.append((entity_text, current_tag))

                # Start new entity
                current_entity = [word]
                current_tag = label

            elif label.startswith("I-"):
                # Continue entity
                if current_entity and label.replace("I-", "B-") == current_tag:
                    current_entity.append(word)
                else:
                    # Inconsistent tag, start new
                    if current_entity:
                        entity_text = " ".join(current_entity)
                        entities.append((entity_text, current_tag))
                    current_entity = [word]
                    current_tag = label.replace("I-", "B-")

            else:  # O tag
                if current_entity:
                    entity_text = " ".join(current_entity)
                    entities.append((entity_text, current_tag))
                    current_entity = []
                    current_tag = None

        # Don't forget last entity
        if current_entity:
            entity_text = " ".join(current_entity)
            entities.append((entity_text, current_tag))

        return entities

    def extract_from_chunks(self, chunks: List[str]) -> List[Dict]:
        """
        Extract entities from multiple text chunks

        Args:
            chunks: List of text chunks

        Returns:
            List of dictionaries with 'content' and 'entities' keys
        """
        results = []

        for i, chunk in enumerate(chunks):
            entities = self.extract_entities(chunk)
            results.append({
                "content": chunk,
                "entities": entities
            })

            if (i + 1) % 10 == 0:
                print(f"✅ Processed {i+1}/{len(chunks)} chunks")

        return results


def main():
    """Test inference"""
    # Test text
    test_text = """
    Sỏi tiết niệu là bệnh phổ biến. Triệu chứng chính là đau quặn thận và đái máu.
    Có thể điều trị bằng thuốc lợi tiểu hoặc phẫu thuật. Biến chứng nguy hiểm là suy thận cấp.
    """

    print("="*60)
    print("🧪 Testing PhoBERT NER Inference")
    print("="*60)

    # Check if model exists
    if not os.path.exists(config.OUTPUT_DIR):
        print(f"\n❌ Model not found at: {config.OUTPUT_DIR}")
        print(f"💡 Please train the model first: python phobert_train.py")
        return

    # Initialize inference
    ner = PhoBERTNERInference()

    # Extract entities
    print(f"\n📝 Input text:")
    print(test_text)

    print(f"\n🔍 Extracting entities...")
    entities = ner.extract_entities(test_text)

    print(f"\n✅ Extracted entities:")
    for category, items in entities.items():
        if items:
            print(f"  {category}: {items}")

    # Test with JSON output format
    print(f"\n📦 JSON format:")
    result = {
        "content": test_text.strip(),
        "entities": entities
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
