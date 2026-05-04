# NanoGPT: A Minimalist Transformer Implementation
A clean, readable, and highly extensible implementation of a GPT-style Transformer, inspired by Andrej Karpathy's nanoGPT. This repository focuses on the core mechanics of the decoder-only architecture, making it an ideal starting point for experiments in Language Modeling and Generative AI.

## Technical Specifications

Architecture
- Model Type: Causal Decoder-only Transformer
- Layers: 12 Transformer Blocks
- Embedding Dimension ($n_{embd}$): 768
- Attention Heads: 12
- Head Dimension: 64
- Context Window ($block\_size$): 256
- TokensVocabulary Size: 50,257 (GPT-2 Tokenization)
- Parameters: ~124M (standard "GPT-2 Small" scale)

## Key Architectural Features
- Weight Tying: Shared weights between token embeddings (wte) and the language modeling head (lm_head).
- Positional Embeddings: Learned absolute positional embeddings (wpe).
- Normalization: Pre-norm formulation using LayerNorm.
- Activation: GELU non-linearity within the Feed-Forward Network (FFN).
- Attention: Scaled Dot-Product Attention with causal masking.
- Initialization: Custom normal distribution initialization ($\sigma = 0.02$) with special scaling for residual projections ($\frac{0.02}{\sqrt{2 \times n\_layer}}$).
