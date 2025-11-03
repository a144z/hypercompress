# Hypercompression Implementation Plan

## Current Issues
1. **BLT returns 1.00x** - Skips tied weights completely, falls back to projection that never runs properly
2. **Merge only blends weights** - Doesn't actually reduce parameter count (param_ratio stays ~2.5x)
3. **Branch execution order** - BLT runs after structural changes so finds no embeddings
4. **Metrics vs Reality** - High effective ratios but low actual param_ratio

## Root Cause
The pipeline computes "effective parameters" but doesn't actually restructure the model to achieve true compression. Only structural LRA actually changes the model architecture.

## Solution: Aggressive Structural Compression Pipeline

### Phase 1: Structural BLT (FIRST)
- Run BEFORE any other compression
- For tied weights: Create wrapper that shares the compressed embedding
- Replace: `Embedding(V, D)` → `Embedding(V, r) + Linear(r, D)` 
- Update lm_head to match: `Linear(D, V)` → `Linear(D, r) + Linear(r, V)`

### Phase 2: Structural LRA
- Replace all `Linear(in, out)` → `Linear(in, r) + Linear(r, out)`
- Use planned ranks from budget

### Phase 3: Weight Compression (LRA + KV + Sparsity)
- Apply to the ALREADY structurally compressed model
- These provide additional "effective" savings

### Phase 4: Merge
- Since structure is already changed, just blend remaining weights

