uv run eval/durations.py \
  --dataset-root /Users/ben/dev/propred/data/LibriTTS \
  --split test-clean \
  --target-speaker 1069 \
  --C 0 \
  --dp-ckpt /Users/ben/dev/propred/lightning_logs/duration/smooth_log_50e/checkpoints/epoch=33-step=13940.ckpt \
  --smooth 1