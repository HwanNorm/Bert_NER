"""
Data preprocessing for PhoBERT NER training
Handles BIO format data from ViMedNER
"""

import os
import random
from typing import List, Tuple, Dict
from datasets import Dataset, DatasetDict
import phobert_config as config


class BIODataProcessor:
    """Process BIO-tagged data for PhoBERT NER training"""

    def __init__(self):
        self.label2id = {}
        self.id2label = {}

    def read_bio_file(self, file_path: str) -> List[Tuple[List[str], List[str]]]:
        """
        Read BIO format file

        Format:
        word1 TAG
        word2 TAG
        <blank line>
        word1 TAG
        ...

        Returns:
            List of (tokens, labels) tuples
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        sentences = []
        tokens = []
        labels = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:  # Empty line = sentence boundary
                    if tokens:
                        sentences.append((tokens, labels))
                        tokens = []
                        labels = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        label = parts[1]
                        tokens.append(token)
                        labels.append(label)

            # Don't forget last sentence
            if tokens:
                sentences.append((tokens, labels))

        print(f"✅ Loaded {len(sentences)} sentences from {file_path}")
        return sentences

    def filter_entity_types(self, sentences: List[Tuple[List[str], List[str]]]) -> List[Tuple[List[str], List[str]]]:
        """
        Filter to keep only entity types we want (ten_benh, trieu_chung, thuoc)
        Convert other entities to 'O' (outside)
        """
        allowed_entities = set(config.ENTITY_MAPPING.keys())
        filtered = []

        for tokens, labels in sentences:
            new_labels = []
            for label in labels:
                if label == "O":
                    new_labels.append("O")
                else:
                    # Extract entity type (B-ten_benh -> ten_benh)
                    parts = label.split("-", 1)
                    if len(parts) == 2:
                        prefix, entity_type = parts
                        if entity_type in allowed_entities:
                            new_labels.append(label)
                        else:
                            new_labels.append("O")
                    else:
                        new_labels.append("O")

            filtered.append((tokens, new_labels))

        return filtered

    def build_label_mapping(self, sentences: List[Tuple[List[str], List[str]]]):
        """Build label2id and id2label mappings"""
        unique_labels = set()

        for _, labels in sentences:
            unique_labels.update(labels)

        # Sort for consistency
        unique_labels = sorted(list(unique_labels))

        # Create mappings
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        print(f"\n📋 Found {len(unique_labels)} unique labels:")
        for label, idx in sorted(self.label2id.items(), key=lambda x: x[1]):
            print(f"  {idx}: {label}")

    def convert_to_dataset(self, sentences: List[Tuple[List[str], List[str]]]) -> Dataset:
        """Convert sentences to HuggingFace Dataset format"""
        tokens_list = []
        labels_list = []

        for tokens, labels in sentences:
            tokens_list.append(tokens)
            # Convert labels to IDs
            label_ids = [self.label2id[label] for label in labels]
            labels_list.append(label_ids)

        dataset = Dataset.from_dict({
            "tokens": tokens_list,
            "ner_tags": labels_list
        })

        return dataset

    def split_single_file(self, file_path: str) -> Tuple[List, List, List]:
        """Split single file into train/dev/test"""
        sentences = self.read_bio_file(file_path)

        # Shuffle
        random.seed(42)
        random.shuffle(sentences)

        # Calculate split points
        total = len(sentences)
        train_end = int(total * config.TRAIN_RATIO)
        dev_end = train_end + int(total * config.DEV_RATIO)

        train_data = sentences[:train_end]
        dev_data = sentences[train_end:dev_end]
        test_data = sentences[dev_end:]

        print(f"\n📊 Split data:")
        print(f"  Train: {len(train_data)} sentences")
        print(f"  Dev:   {len(dev_data)} sentences")
        print(f"  Test:  {len(test_data)} sentences")

        return train_data, dev_data, test_data

    def process_data(self) -> DatasetDict:
        """Main processing function"""
        print("🔄 Processing ViMedNER data...\n")

        # Load data
        if config.SINGLE_FILE:
            print(f"📁 Loading from single file: {config.SINGLE_FILE_PATH}")
            train_data, dev_data, test_data = self.split_single_file(config.SINGLE_FILE_PATH)
        else:
            print(f"📁 Loading from directory: {config.VIMEDNER_DIR}")
            train_path = os.path.join(config.VIMEDNER_DIR, config.TRAIN_FILE)
            dev_path = os.path.join(config.VIMEDNER_DIR, config.DEV_FILE)
            test_path = os.path.join(config.VIMEDNER_DIR, config.TEST_FILE)

            train_data = self.read_bio_file(train_path)
            dev_data = self.read_bio_file(dev_path)
            test_data = self.read_bio_file(test_path)

        # Filter to keep only our 3 entity types
        print(f"\n🔍 Filtering to keep only: {list(config.ENTITY_MAPPING.keys())}")
        train_data = self.filter_entity_types(train_data)
        dev_data = self.filter_entity_types(dev_data)
        test_data = self.filter_entity_types(test_data)

        # Build label mappings
        all_data = train_data + dev_data + test_data
        self.build_label_mapping(all_data)

        # Convert to datasets
        train_dataset = self.convert_to_dataset(train_data)
        dev_dataset = self.convert_to_dataset(dev_data)
        test_dataset = self.convert_to_dataset(test_data)

        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": dev_dataset,
            "test": test_dataset
        })

        print(f"\n✅ Data processing complete!")
        print(f"📦 Dataset sizes:")
        for split, dataset in dataset_dict.items():
            print(f"  {split}: {len(dataset)} examples")

        return dataset_dict, self.label2id, self.id2label


def main():
    """Test data processing"""
    processor = BIODataProcessor()

    try:
        dataset_dict, label2id, id2label = processor.process_data()

        # Show example
        print(f"\n📝 Example from training set:")
        example = dataset_dict["train"][0]
        print(f"  Tokens: {example['tokens'][:10]}...")
        print(f"  Labels: {[id2label[l] for l in example['ner_tags'][:10]]}...")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"\n💡 Please check:")
        print(f"  1. Update VIMEDNER_DIR in phobert_config.py")
        print(f"  2. Make sure data files exist")
        print(f"  3. Check data format (word TAG per line, blank line between sentences)")


if __name__ == "__main__":
    main()
