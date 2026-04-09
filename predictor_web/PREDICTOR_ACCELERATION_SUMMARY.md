# Predictor inference acceleration summary

## Bottlenecks observed
- Predictor inference previously executed a strict per-sequence loop in `run_prediction`, with one tokenization call and one model forward pass for each sequence.
- This caused unnecessary Python overhead and low hardware utilization for uploads containing many sequences.
- Model/tokenizer loading was already cached with `lru_cache`, so repeated request initialization was not the dominant issue.

## Optimizations implemented
- Added batched inference utilities in `predict_service.py`:
  - batched text preparation for all sequences
  - batched tokenizer calls
  - batched model forward passes under `torch.inference_mode()`
- Kept existing model/tokenizer caching and checkpoint semantics unchanged.
- Reused the same predictor service for both predictor tab and generator post-scoring to avoid duplicate inference code paths.

## Not implemented intentionally (future options)
- No model architecture changes, quantization, TorchScript, ONNX, or TensorRT conversion (out of scope and could alter behavior/performance characteristics non-locally).
- No multi-GPU/distributed inference redesign.
- No asynchronous task queue or background worker system.

## Expected speedup source
- Main expected gain is from **batching** (fewer tokenizer/forward invocations).
- Secondary gain is reduced Python overhead in per-sequence loops.
- Existing **caching** remains important to avoid reloading tokenizer and checkpoints per request.
