"""
Quick Start Script for PhoBERT NER
Checks setup and guides you through the process
"""

import os
import sys


def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")


def check_dependencies():
    """Check if required packages are installed"""
    print_header("1️⃣  Checking Dependencies")

    required = {
        "transformers": "transformers",
        "torch": "torch",
        "datasets": "datasets",
        "seqeval": "seqeval",
    }

    missing = []
    for name, package in required.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"📦 Install with:")
        print(f"   pip install -r requirements_phobert.txt")
        return False

    print(f"\n✅ All dependencies installed!")
    return True


def check_data():
    """Check if ViMedNER data exists"""
    print_header("2️⃣  Checking Training Data")

    import phobert_config as config

    if config.SINGLE_FILE:
        if os.path.exists(config.SINGLE_FILE_PATH):
            print(f"✅ Found data file: {config.SINGLE_FILE_PATH}")
            return True
        else:
            print(f"❌ Data file not found: {config.SINGLE_FILE_PATH}")
    else:
        data_dir = config.VIMEDNER_DIR
        train_path = os.path.join(data_dir, config.TRAIN_FILE)
        dev_path = os.path.join(data_dir, config.DEV_FILE)
        test_path = os.path.join(data_dir, config.TEST_FILE)

        if all(os.path.exists(p) for p in [train_path, dev_path, test_path]):
            print(f"✅ Found all data files in: {data_dir}")
            print(f"   - {config.TRAIN_FILE}")
            print(f"   - {config.DEV_FILE}")
            print(f"   - {config.TEST_FILE}")
            return True
        else:
            print(f"❌ Data files not found in: {data_dir}")

    print(f"\n💡 To fix:")
    print(f"   1. Download ViMedNER dataset")
    print(f"   2. Place in: NER LLM/vimedner_data/")
    print(f"   3. Update paths in phobert_config.py")
    print(f"\n📖 See PHOBERT_SETUP_GUIDE.md for details")
    return False


def check_gpu():
    """Check if GPU is available"""
    print_header("3️⃣  Checking GPU")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print(f"⚠️  No GPU detected")
            print(f"   Training will work but be slower (2-4 hours vs 30-60 min)")
            print(f"   Inference will still work fine on CPU")
            return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False


def check_existing_model():
    """Check if model is already trained"""
    print_header("4️⃣  Checking Existing Model")

    import phobert_config as config

    if os.path.exists(config.OUTPUT_DIR):
        model_file = os.path.join(config.OUTPUT_DIR, "pytorch_model.bin")
        if os.path.exists(model_file):
            print(f"✅ Found trained model at: {config.OUTPUT_DIR}")
            print(f"   You can skip training and go directly to inference!")
            return True

    print(f"📭 No trained model found")
    print(f"   You need to train first: python phobert_train.py")
    return False


def show_next_steps(has_model):
    """Show what to do next"""
    print_header("🎯 Next Steps")

    if has_model:
        print("✅ You're ready to use PhoBERT NER!")
        print("\n📝 Try these commands:\n")
        print("   # Test on sample text")
        print("   python phobert_inference.py\n")
        print("   # Evaluate model")
        print("   python phobert_evaluate.py\n")
        print("   # Run full pipeline (replacement for Gemini)")
        print("   python phobert_ner_pipeline.py\n")
    else:
        print("🚀 Ready to train!")
        print("\n📝 Run this command:\n")
        print("   python phobert_train.py\n")
        print("⏱️  Expected time:")
        print("   - GPU: 30-60 minutes")
        print("   - CPU: 2-4 hours\n")
        print("📊 You can monitor training with:")
        print("   tensorboard --logdir=phobert_medical_ner\n")

    print("📖 For more details, see: PHOBERT_SETUP_GUIDE.md")


def main():
    print("\n" + "🏥"*30)
    print("  PhoBERT Vietnamese Medical NER - Quick Start")
    print("🏥"*30)

    # Run checks
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n❌ Please install dependencies first!")
        return

    data_ok = check_data()
    gpu_ok = check_gpu()
    model_exists = check_existing_model()

    # Summary
    print_header("📋 Summary")

    status = []
    status.append(("Dependencies", "✅" if deps_ok else "❌"))
    status.append(("Training Data", "✅" if data_ok else "❌"))
    status.append(("GPU Available", "✅" if gpu_ok else "⚠️"))
    status.append(("Trained Model", "✅" if model_exists else "📭"))

    for item, symbol in status:
        print(f"  {symbol} {item}")

    # Can we proceed?
    can_train = deps_ok and data_ok
    can_infer = deps_ok and model_exists

    print()
    if can_infer:
        print("✅ READY TO USE - Model is trained!")
    elif can_train:
        print("🟡 READY TO TRAIN - Data is prepared!")
    else:
        print("❌ NOT READY - Please fix issues above")

    # Next steps
    if can_train or can_infer:
        show_next_steps(model_exists)
    else:
        print("\n💡 Fix the issues above, then run this script again:")
        print("   python quick_start.py")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
