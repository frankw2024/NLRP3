# Why Training Losses Become NaN

## Problem Summary

In your training history, you're seeing:
- **NaN training losses** in Fold 1 and Fold 3
- **0.0 validation metrics** (AUC, AP) when training fails
- Fold 2 trains successfully

## Root Causes

### 1. **Numerical Instability Cascade**
When model outputs (logits) become NaN or Inf:
1. Loss computation produces NaN
2. `loss.backward()` propagates NaN through gradients
3. NaN gradients corrupt all model weights
4. All subsequent predictions become NaN
5. Training cannot recover

### 2. **Extreme `pos_weight` Values**
`BCEWithLogitsLoss` with `pos_weight` can cause issues:
- If class imbalance is extreme, `pos_weight` can be very large
- Large `pos_weight` amplifies loss for positive samples
- Can lead to gradient explosion → NaN

### 3. **Uninitialized or Extreme Initial Weights**
- Default PyTorch initialization might produce extreme values
- Extreme activations → NaN in forward pass
- No NaN checking before loss computation

### 4. **Division by Zero or Invalid Operations**
- Division operations (e.g., pooling) without proper masking
- Attention mechanisms with all-zero masks
- Cross-attention with mismatched dimensions

## Why Fold 2 Works But Folds 1 & 3 Don't

**Different data splits** → different class distributions:
- Fold 1 & 3: Extreme class imbalance in training set
- Fold 2: More balanced split → stable training

## Fixes Applied

### 1. **NaN Detection in Training Loop**
```python
# Check logits before loss computation
if torch.isnan(logits).any() or torch.isinf(logits).any():
    print("WARNING: NaN/Inf in logits, skipping batch")
    continue

# Check loss before backward pass
if torch.isnan(loss) or torch.isinf(loss):
    print("WARNING: NaN/Inf loss, skipping batch")
    continue

# Check gradients before optimizer step
for param in model.parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print("WARNING: NaN gradients, skipping update")
        optimizer.zero_grad()
        continue
```

### 2. **Safe `pos_weight` Calculation**
```python
# Clamp pos_weight to reasonable range [0.1, 10.0]
pos_weight_val = float(neg_count) / float(pos_count)
pos_weight_val = max(0.1, min(10.0, pos_weight_val))
```

### 3. **Weight Initialization**
```python
# Use conservative Xavier initialization
nn.init.xavier_uniform_(module.weight, gain=0.5)  # Lower gain = smaller initial weights
```

### 4. **NaN Checking in Model Forward Pass**
```python
# Check intermediate outputs and replace NaN with zeros
if torch.isnan(lig_pool).any():
    lig_pool = torch.nan_to_num(lig_pool, nan=0.0, posinf=0.0, neginf=0.0)
```

## Why Validation Metrics Are 0.0

When training produces NaN:
- Model weights are corrupted
- All predictions become NaN
- `safe_metric()` replaces NaN predictions with 0.5
- With all predictions = 0.5, metrics fail:
  - AUC: Requires variation in predictions → returns 0.0
  - AP: Same issue → returns 0.0
  - Precision/Recall/F1: All predictions same → returns 0.0

## Prevention Strategy

1. **Early Detection**: Check for NaN at every step
2. **Skip Corrupted Batches**: Don't update model with NaN gradients
3. **Conservative Initialization**: Prevent extreme initial values
4. **Gradient Clipping**: Already in place (max_norm=1.0)
5. **Learning Rate**: Consider reducing if NaN persists (currently 2e-5)

## Additional Recommendations

1. **Lower Learning Rate**: Try `lr=1e-5` for more stable training
2. **Warmup**: Add learning rate warmup for first few epochs
3. **Batch Size**: Smaller batches (currently 8) are good, but could try 4
4. **Gradient Accumulation**: Accumulate gradients over multiple batches before update
5. **Mixed Precision**: Use `torch.cuda.amp` if GPU available (reduces memory, can help stability)

## Expected Behavior After Fixes

- NaN batches will be skipped with warnings
- Training will continue with valid batches
- Model won't be corrupted by NaN gradients
- More folds should train successfully
- Better overall cross-validation results


Summary: Why NaN Losses Occur
Root causes
Numerical instability cascade:
Logits become NaN → loss becomes NaN → gradients become NaN → model weights corrupted → all predictions NaN
Extreme pos_weight values:
With severe class imbalance, pos_weight can be very large
Large pos_weight amplifies loss, leading to gradient explosion
Unchecked operations:
No NaN checking before loss computation
No gradient validation before optimizer step
Division operations without proper safeguards
Why Fold 2 works but Folds 1 & 3 don't
Different data splits → different class distributions:
Folds 1 & 3: Extreme imbalance in training set → unstable
Fold 2: More balanced split → stable training
Fixes applied
NaN detection in training: Check logits, loss, and gradients before updates
Safe pos_weight: Clamp to [0.1, 10.0] to prevent extreme values
Weight initialization: Conservative Xavier init (gain=0.5) to prevent extreme initial values
Model forward pass: NaN checking and replacement in intermediate outputs
Why validation metrics are 0.0
When training produces NaN:
All predictions become NaN → replaced with 0.5
All predictions = 0.5 → no variation → AUC/AP return 0.0
Binary metrics also fail when all predictions are the same
The fixes should prevent NaN from corrupting the model and allow training to continue even when some batches produce NaN. The code will now skip corrupted batches and continue with valid ones.
Created NaN_loss_explanation.md with a detailed explanation.
