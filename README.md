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
