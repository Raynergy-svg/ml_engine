# Training on Free Cloud GPU

Your ML engine is now ready to train on **Google Colab's free GPU** (Tesla T4, 16GB).

## Quick Start (3 steps)

### 1. Upload to Google Colab

**Option A: Direct Upload**
1. Go to https://colab.research.google.com
2. Upload `colab_train.ipynb`
3. Upload your `market_data/*.csv` files when prompted

**Option B: GitHub** (recommended)
```bash
# Push to GitHub first
git add .
git commit -m "Ready for cloud training"
git push
```
Then in Colab, just clone the repo (first cell handles this).

### 2. Enable GPU

In Colab menu:
- **Runtime** → **Change runtime type** → **GPU** → **Save**

### 3. Run All Cells

- **Runtime** → **Run all**

Training will start automatically and save checkpoints every time validation improves.

## What You Get (Free)

- **GPU**: Tesla T4 (16GB VRAM)
- **Session**: Up to 12 hours
- **Cost**: $0
- **Speed**: ~10-20x faster than CPU

## Auto-Resume Feature

If your session disconnects:
1. Restart the notebook
2. Run all cells again
3. Training continues from last checkpoint (no progress lost!)

## Download Your Model

After training, the last cell downloads `trained_models.zip`. Extract it to:
```
trained_data/models/
```

Then use locally:
```bash
.venv/bin/python main.py predict -t ORCL -p 1d -i 5m
```

## Alternative Free Options

| Platform | GPU | Free Hours | Notes |
|----------|-----|------------|-------|
| **Google Colab** | T4 | ~12h/session | Best for most users |
| **Kaggle** | P100/T4 | 30h/week | Good limits |
| **Paperspace** | M4000 | 6h/session | Easy setup |
| **Lightning AI** | Various | Limited | New option |

## Paid Upgrades (Optional)

If you need more:
- **Colab Pro**: $10/mo → Better GPUs (A100), 24h sessions, no idle timeout
- **Kaggle**: Free 30hrs/week is usually enough
- **AWS/GCP**: Pay per hour (more expensive)

---

Your training is configured to:
- ✓ Auto-resume from checkpoints
- ✓ Use mixed precision (faster)
- ✓ Save best model automatically
- ✓ Run 100 epochs (adjustable in notebook)
- ✓ Early stopping after 20 epochs no improvement

**Next:** Open `colab_train.ipynb` in Google Colab and hit "Run all"!
