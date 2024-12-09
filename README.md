# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## Introduction

BERT (Bidirectional Encoder Representations from Transformers) revolutionizes language understanding by leveraging bidirectional context in text. This enables better performance on various natural language processing (NLP) tasks, including sentiment analysis, named entity recognition, and question answering. BERT is pre-trained on large text datasets and fine-tuned for specific tasks with minimal adjustments.

---

## Project Overview

This project implements BERT for sentiment analysis using the IMDB dataset. The goal is to showcase BERT's effectiveness compared to a simpler MLP (Multi-Layer Perceptron) model using GloVe embeddings. Key steps include:

1. **Pre-training and Fine-tuning BERT:** Adapting BERT for binary sentiment classification.
2. **MLP Implementation:** Training a lightweight model for comparison.
3. **Hyperparameter Experimentation:** Evaluating BERT's sensitivity to parameters like learning rate and batch size.
4. **Performance Comparison:** Analyzing accuracy, resource usage, and qualitative differences.

---

## Features

- **State-of-the-art BERT Implementation:** Sentiment analysis with high accuracy using a pre-trained BERT model.
- **MLP Alternative:** A simpler architecture using pre-trained GloVe embeddings.
- **Custom Dataset Handling:** Integration with the IMDB dataset.
- **Hyperparameter Tuning:** Exploration of different configurations to optimize performance.
- **Comprehensive Evaluation:** Accuracy and efficiency comparisons between BERT and MLP.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/google-research/bert
   cd bert
   ```

2. Install dependencies:
   ```bash
   pip install transformers datasets torch evaluate
   ```

3. Download GloVe embeddings (for MLP):
   ```bash
   wget http://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip
   ```

---

## Usage

### BERT Implementation

1. Preprocess the IMDB dataset:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("imdb").shuffle(seed=42)
   ```

2. Train BERT:
   ```python
   from transformers import Trainer, TrainingArguments
   trainer = Trainer(model=model, args=training_args, ...)
   trainer.train()
   ```

3. Evaluate the model:
   ```python
   results = trainer.evaluate()
   print(f"Evaluation Results: {results}")
   ```

4. Make predictions:
   ```python
   inputs = tokenizer("Your text here", return_tensors="pt", truncation=True)
   outputs = model(**inputs)
   predicted_class = torch.argmax(outputs.logits).item()
   ```

---

### MLP Implementation

1. Load GloVe embeddings:
   ```python
   def load_glove_embeddings():
       # Code to load embeddings
   ```

2. Train the MLP model:
   ```python
   for epoch in range(10):
       # Training loop
   ```

3. Evaluate MLP:
   ```python
   accuracy = correct / total
   print(f"MLP Accuracy: {accuracy}")
   ```

---

## Results

- **BERT Accuracy:** ~91.2% on the IMDB dataset.
- **MLP Accuracy:** ~72.8%, highlighting BERT's superior performance.
- **Efficiency:** BERT is computationally intensive during pretraining but efficient during fine-tuning.

---

## Observations and Limitations

1. **BERT:** Excels in capturing bidirectional context but requires significant computational resources.
2. **MLP:** Resource-friendly but lacks the contextual depth of BERT.
3. **Black-box Nature:** Interpretability of BERT models remains challenging.

---

## Future Directions

- Experimenting with newer transformer models like DeBERTa.
- Optimizing BERT for resource-constrained environments using quantization or pruning.
- Adapting domain-specific BERT models for specialized tasks.

---

## References

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv:1810.04805v2*.
- Gaur, M., Faldu, K., & Sheth, A. (2021). Semantics of the black-box: Can knowledge graphs help make deep learning systems more interpretable and explainable? *IEEE Internet Computing, 25(1), 51-59*.

## Authors

- Steve Kuruvilla - 200573392
- Shrasth Kumar - 200566998

---
