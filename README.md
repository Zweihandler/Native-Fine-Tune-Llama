# Native Fine-Tuning with Llama 3.1

A comprehensive guide for fine-tuning the Llama 3.1-8B model using native PyTorch training with Supervised Fine-Tuning (SFT) on an Indonesian language dataset.

## 📋 Overview

This project demonstrates how to efficiently fine-tune the Llama 3.1-8B model using:
- **4-bit Quantization** with BitsAndBytes for memory-efficient training
- **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning
- **Supervised Fine-Tuning (SFT)** on the Indonesian Alpaca dataset
- **Native PyTorch integration** with HuggingFace transformers

The pipeline creates a context-aware Indonesian language assistant optimized for your specific use cases.

## ✨ Key Features

- **Memory-Efficient Training**: 4-bit quantization reduces memory requirements significantly
- **Parameter Efficient**: LoRA adapts only a small fraction of model parameters
- **Indonesian Language Support**: Fine-tunes on Indonesian Alpaca dataset (31.6K samples)
- **Chat Format Support**: Implements proper chat template formatting for conversational use
- **GPU Monitoring**: Real-time GPU memory tracking and optimization
- **Flexible Monitoring**: Optional integration with Weights & Biases (W&B) for experiment tracking

## Quick Start

### Basic Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("No GPU available. CPU training may be slow.")
```

### Run the Notebook

Open and run `Native-Fine-Tune.ipynb` in Jupyter:

```bash
jupyter notebook Native-Fine-Tune.ipynb
```

Follow the cells in order:
1. Install dependencies
2. Set up GPU monitoring
3. Configure quantization
4. Load model and tokenizer
5. Prepare dataset
6. Configure LoRA
7. Train model
8. Save and test

### Learning Rate Schedules

- **Linear**: Gradually decrease from max to 0
- **Constant**: Keep learning rate constant
- **Cosine**: Cosine annealing schedule

### Batch Size Tuning

- Increase batch size for faster training (if GPU memory allows)
- Larger effective batch size may require learning rate adjustment
- Sweet spot: effective batch size of 32-64

## 📖 Additional Resources

- [LLaMA 3.1 Model Card](https://huggingface.co/meta-llama/Llama-2-7b)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [BitsAndBytes Guide](https://github.com/TimDettmers/bitsandbytes)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

## 🎓 Dataset Information

**cahya/alpaca-id-cleaned:**
- Language: Indonesian
- Size: 31.6K samples
- Format: Instruction, Input, Output
- Source: Indonesian translation of Stanford Alpaca
- License: CC0 1.0 Universal

---

# Native-Fine-Tune-Llama
fine-tuning the Llama 3.1-8B model using native PyTorch training with Supervised Fine-Tuning (SFT) on an Indonesian language dataset.
