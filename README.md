# Transformer_From_Scratch
### Transformer from Scratch: English–German Translator 🌍🔤

## Overview
This project is a **from-scratch implementation** of a Transformer model applied to an English–German machine translation task. Designed using Object-Oriented Programming (OOP) principles, the project showcases my skills in building deep learning architectures from fundamental components. The Transformer model, built entirely from scratch, includes the core building blocks such as Multi-Head Self-Attention, Feed-Forward Networks, Layer Normalization, and Residual Connections. An interactive user interface built with Streamlit allows users to easily translate sentences and view the translation along with confidence scores.

## Key Features
- **Custom Transformer Architecture:**  
  - Developed foundational Transformer components including self-attention, feed-forward networks, and positional encodings.
  - Modular and extensible OOP design for easy experimentation and future modifications.
- **Embedding Layers:**  
  - Implemented custom embedding layers to handle word representations.
- **Data Handling & Preprocessing:**  
  - Custom scripts for dataset loading, text preprocessing, and vocabulary building.
- **Training & Evaluation:**  
  - Full training pipeline in PyTorch with checkpointing and performance monitoring.
- **User Interface:**  
  - An engaging Streamlit app for interactive translation demonstrations.
- **Comprehensive Documentation:**  
  - Well-structured project files and detailed notebooks for exploration and analysis.

## Project Structure
| File                   | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `building_blocks.py`   | Transformer layers (Attention, FFN, Residuals, mask generation)                        |
| `embeddings.py`        | Token + positional embedding implementation                            |
| `dataset.py`           | Bilingual dataset loader with bucketing                                 |
| `preprocess.py`        | Text cleaning, BPE tokenization, vocab building                        |
| `tools.py`             | Utilities (save_model, Load_model, translate_function)                               |
| `main.py`              | Training pipeline (model, optimizer, loss)                             |
| `main_notebook.ipynb`  | Experimental analysis (attention visualization, grad checks)           |
| `translator_app.py`    | Streamlit app with input textbox and translation display               |

## Technical Details
- **Architecture:**  
  - **Encoder & Decoder:** Built from scratch using multi-head self-attention, feed-forward networks, positional encoding, layer normalization, and residual connections.
  - **Attention Mechanism:** Uses scaled dot-product attention.
  - **Positional Encoding:** Implemented via learnable or fixed embeddings to incorporate sequence order.
- **Training:**  
  - **Loss Function:** Cross-Entropy Loss (with padding tokens ignored).
  - **Optimizer:** Adam optimizer with gradient clipping.
  - **Checkpointing:** Periodically saves model state along with vocabularies and configuration.
- **Data:**  
  - Utilizes an English–German parallel corpus. Kaggle datasets like [English To German]([https://kaggle.com/](https://www.kaggle.com/datasets/kaushal2896/english-to-german)]
  
## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/M-Esmat/Transformer_From_Scratch.git
   cd Transformer_From_Scratch

## Results
- **Training Loss:** Example loss values (e.g., *Epoch 600 | Loss: 0.0079*) indicate strong convergence.


## Potential Improvements / Future Work
- Experiment with more advanced Transformer variants (e.g. BERT-based models).
- Incorporate attention visualization in the Streamlit app.
- Enhance preprocessing to handle more complex language structures.
- Expand the dataset for more robust evaluation and generalization.
- Deploy the model as an API using services like FastAPI.

