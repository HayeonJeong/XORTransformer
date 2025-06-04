# XORTransformer

This project explores how to solve the XOR problem using a minimal Transformer architecture implemented in PyTorch.  

## 🔍 Objective
- Find the **simplest Transformer architecture** that can successfully solve the XOR problem.
- Experimentally analyze the necessity of each component in the self-attention mechanism:
  `D_k`, learnable weights, bias, positional encoding, softmax, and layer normalization.

## 📁 Directory Structure
- `train.py`: Training script
- `utils/xor_transformer.py`: Transformer model tailored for XOR
- `results/`: Contains output logs of each experiment
- `past_question/`: My personal notes and analysis while working on the assignment
- `report.md`: Final write-up summarizing the experiment results, key insights, and mathematical analysis of the XOR-transformer model

## ⚙️ Experiment Setup
- Input: Sequence of two scalar tokens (−1 or +1)
- Output: XOR result (0 or 1)
- Loss function: Binary Cross Entropy
- Optimizer: Adam
- Success criterion: If any random seed results in perfect prediction, the configuration is considered valid

## 🧪 Key Experiments
- ✅ **Baseline**: All components enabled (`D_k=4`, trainable weights/bias, positional encoding, softmax, layer norm)
- ✅ **Minimal working structure**: `D_k=2`, only **bias is trainable**, all other components disabled  
  → Total number of trainable parameters: **9**

## 📊 Key Findings
- The XOR problem can be solved **with only learnable bias**
- Fixing weights, removing softmax/PE/LayerNorm is acceptable
- Achieves a balance between model simplicity and performance

## 🧠 Mathematical Analysis
- The attention score and final output are mathematically analyzed to explain how the minimal model works

## 🚀 How to Run
Run `train.py` with your chosen configuration using the `--conditions` argument (JSON format):

```bash
python train.py --conditions '{"D_k": 4, "weights": true, "bias": true, "positional_encoding": true, "softmax": true, "layer_norm": true}'