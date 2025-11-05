"""
PhoBERT NER Fine-tuning Script
Train PhoBERT on Vietnamese medical NER (Bệnh, Triệu chứng, Thuốc)
"""

import os
import json
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import phobert_config as config
from phobert_data_processor import BIODataProcessor


class PhoBERTNERTrainer:
    """PhoBERT NER Training Manager"""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.label2id = None
        self.id2label = None

    def load_tokenizer_and_model(self, num_labels: int):
        """Load PhoBERT tokenizer and model"""
        print(f"\n🤖 Loading PhoBERT model: {config.PHOBERT_MODEL}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.PHOBERT_MODEL)

        # Load model
        self.model = AutoModelForTokenClassification.from_pretrained(
            config.PHOBERT_MODEL,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        print(f"✅ Model loaded with {num_labels} labels")

    def tokenize_and_align_labels(self, examples):
        """
        Tokenize inputs and align labels with subword tokens
        PhoBERT-compatible version (works with slow tokenizer)
        """
        tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for tokens, ner_tags in zip(examples["tokens"], examples["ner_tags"]):
            # Join tokens into text (PhoBERT expects space-separated text)
            text = " ".join(tokens)

            # Tokenize the text
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=config.MAX_LENGTH,
                padding=False,
            )

            # Now align labels manually
            # PhoBERT uses BPE, so we need to map original tokens to BPE tokens
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            # Create label sequence
            # Start with -100 for [CLS]
            label_ids = [-100]

            token_idx = 0
            bpe_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[1:-1])  # Skip CLS and SEP

            for bpe_token in bpe_tokens:
                if token_idx >= len(tokens):
                    label_ids.append(-100)
                    continue

                # Check if this BPE token starts a new word (no underscore prefix)
                if not bpe_token.startswith("@@"):
                    # New word - assign label
                    label_ids.append(ner_tags[token_idx])
                    # Check if we should move to next token
                    current_word = tokens[token_idx].lower()
                    if bpe_token.replace("@@", "").lower() == current_word:
                        token_idx += 1
                else:
                    # Continuation of previous word - use -100
                    label_ids.append(-100)

            # Add -100 for [SEP]
            label_ids.append(-100)

            # Ensure same length
            if len(label_ids) != len(input_ids):
                # Fallback: simple assignment
                label_ids = [-100] + ner_tags[:len(input_ids)-2] + [-100]
                # Pad if needed
                while len(label_ids) < len(input_ids):
                    label_ids.append(-100)
                label_ids = label_ids[:len(input_ids)]

            tokenized_inputs["input_ids"].append(input_ids)
            tokenized_inputs["attention_mask"].append(attention_mask)
            tokenized_inputs["labels"].append(label_ids)

        return tokenized_inputs

    def compute_metrics(self, p):
        """Compute NER metrics (precision, recall, F1)"""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens with -100)
        true_labels = []
        true_predictions = []

        for prediction, label in zip(predictions, labels):
            true_label = []
            true_pred = []
            for pred_id, label_id in zip(prediction, label):
                if label_id != -100:
                    true_label.append(self.id2label[label_id])
                    true_pred.append(self.id2label[pred_id])

            true_labels.append(true_label)
            true_predictions.append(true_pred)

        # Compute metrics
        f1 = f1_score(true_labels, true_predictions)
        precision = precision_score(true_labels, true_predictions)
        recall = recall_score(true_labels, true_predictions)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def train(self, dataset_dict, label2id, id2label):
        """Main training function"""
        self.label2id = label2id
        self.id2label = id2label
        num_labels = len(label2id)

        # Load model
        self.load_tokenizer_and_model(num_labels)

        # Tokenize datasets
        print(f"\n🔄 Tokenizing datasets...")
        tokenized_datasets = dataset_dict.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset_dict["train"].column_names,
        )

        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=config.OUTPUT_DIR,
            eval_strategy="steps",
            eval_steps=config.EVAL_STEPS,
            save_strategy="steps",
            save_steps=config.SAVE_STEPS,
            learning_rate=config.LEARNING_RATE,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE,
            num_train_epochs=config.NUM_EPOCHS,
            weight_decay=config.WEIGHT_DECAY,
            warmup_ratio=config.WARMUP_RATIO,
            logging_steps=config.LOGGING_STEPS,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=config.FP16 and torch.cuda.is_available(),
            push_to_hub=False,
            report_to="none",  # Disable tensorboard (install tensorboard if you want logging)
        )

        # Early stopping
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping],
        )

        # Train!
        print(f"\n🚀 Starting training...")
        print(f"   Epochs: {config.NUM_EPOCHS}")
        print(f"   Batch size: {config.BATCH_SIZE}")
        print(f"   Learning rate: {config.LEARNING_RATE}")
        print(f"   GPU: {torch.cuda.is_available()}")
        print(f"\n{'='*50}\n")

        trainer.train()

        # Evaluate on test set
        print(f"\n📊 Evaluating on test set...")
        test_results = trainer.predict(tokenized_datasets["test"])
        print(f"\n✅ Test Results:")
        print(f"   Precision: {test_results.metrics['test_precision']:.4f}")
        print(f"   Recall:    {test_results.metrics['test_recall']:.4f}")
        print(f"   F1-Score:  {test_results.metrics['test_f1']:.4f}")

        # Save final model
        print(f"\n💾 Saving model to {config.OUTPUT_DIR}")
        trainer.save_model(config.OUTPUT_DIR)
        self.tokenizer.save_pretrained(config.OUTPUT_DIR)

        # Save label mappings
        label_map_path = os.path.join(config.OUTPUT_DIR, "label_mappings.json")
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump({
                "label2id": self.label2id,
                "id2label": self.id2label,
            }, f, ensure_ascii=False, indent=2)

        print(f"✅ Label mappings saved to {label_map_path}")

        # Generate detailed classification report
        predictions = test_results.predictions
        labels = test_results.label_ids
        predictions = np.argmax(predictions, axis=2)

        true_labels = []
        true_predictions = []

        for prediction, label in zip(predictions, labels):
            true_label = []
            true_pred = []
            for pred_id, label_id in zip(prediction, label):
                if label_id != -100:
                    true_label.append(self.id2label[label_id])
                    true_pred.append(self.id2label[pred_id])
            true_labels.append(true_label)
            true_predictions.append(true_pred)

        report = classification_report(true_labels, true_predictions, digits=4)
        print(f"\n📋 Detailed Classification Report:")
        print(report)

        # Save report
        report_path = os.path.join(config.OUTPUT_DIR, "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\n✅ Training complete! Model saved to: {config.OUTPUT_DIR}")
        print(f"\n🎯 Next steps:")
        print(f"   1. Check tensorboard: tensorboard --logdir={config.OUTPUT_DIR}")
        print(f"   2. Test inference: python phobert_inference.py")
        print(f"   3. Use in production: python phobert_ner_pipeline.py")


def main():
    """Main training script"""
    print("="*60)
    print("🏥 PhoBERT Vietnamese Medical NER Training")
    print("="*60)

    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        print(f"⚠️  No GPU detected - training will be slower")

    # Process data
    processor = BIODataProcessor()
    dataset_dict, label2id, id2label = processor.process_data()

    # Train model
    trainer = PhoBERTNERTrainer()
    trainer.train(dataset_dict, label2id, id2label)


if __name__ == "__main__":
    main()
