#!/usr/bin/env python3
"""Central fine-tuning script.
Loads checkpoint `artifacts/semantic_lpr_async_fl.pt`, trains centrally on `data/` using
`UnifiedPlateDataset`, prioritises text loss, validates each epoch, saves checkpoint and logs.
"""
import json
from pathlib import Path
import time
import logging

import torch
from torch.utils.data import DataLoader, random_split

from src.model import SemanticLPRNet, semantic_lpr_loss
from src.dataset import UnifiedPlateDataset, decode_text
from src.dataset import MAX_PLATE_LEN
from src.dataset import CHARSET

LOG_PATH = Path('artifacts/training_log_utf8.txt')
CKPT_PATH = Path('artifacts/semantic_lpr_async_fl.pt')
SAVE_CKPT = Path('artifacts/semantic_lpr_finetuned.pt')


def setup_logger():
    logger = logging.getLogger('finetune')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(LOG_PATH, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_checkpoint(device):
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = SemanticLPRNet()
    model.load_state_dict(ckpt['model_state'])
    start_round = int(ckpt.get('round', 0))
    return model, start_round


def compute_exact_and_format(model, device, val_loader, max_samples=200):
    model.eval()
    total = 0
    exact = 0
    format_valid = 0
    sim_sum = 0.0
    samples = 0
    from src.semantic_pipeline import is_valid_plate_format
    from difflib import SequenceMatcher

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['image'].to(device)
            outputs = model(imgs, snr_db=18.0)
            logits = outputs['text_logits']  # [B, L, C]
            preds = logits.argmax(dim=-1).cpu()
            for i in range(preds.shape[0]):
                pred = ''.join([CHARSET[idx] if idx < len(CHARSET) else '' for idx in preds[i].tolist()]).strip()
                gt = batch['raw_text'][i]
                total += 1
                if pred == gt:
                    exact += 1
                if is_valid_plate_format(pred):
                    format_valid += 1
                # similarity
                sim = SequenceMatcher(None, pred, gt).ratio()
                sim_sum += sim
                samples += 1
                if samples >= max_samples:
                    break
            if samples >= max_samples:
                break
    return {
        'exact_pct': 100.0 * exact / samples if samples else 0.0,
        'format_pct': 100.0 * format_valid / samples if samples else 0.0,
        'avg_sim': sim_sum / samples if samples else 0.0,
        'samples': samples,
    }


def train():
    logger = setup_logger()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Device: {device}')

    model, start_round = load_checkpoint(device)
    model = model.to(device)
    logger.info(f'Loaded checkpoint round={start_round}')

    # dataset
    dataset = UnifiedPlateDataset('data', augment=True, ccpd_root=None)
    n = len(dataset)
    val_n = min(200, max(20, int(0.1 * n)))
    train_n = n - val_n
    train_set, val_set = random_split(dataset, [train_n, val_n], generator=torch.Generator().manual_seed(42))
    logger.info(f'Dataset samples: total={n}, train={train_n}, val={val_n}')

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    best_exact = 0.0
    target_exact = 100.0  # user requested 100% if possible
    max_epochs = 200
    early_stop_patience = 8
    stagnation = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0
        start_time = time.time()
        for batch in train_loader:
            imgs = batch['image'].to(device)
            target_imgs = batch['target_image'].to(device)
            text_targets = batch['text'].to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss, stats = semantic_lpr_loss(outputs, {'target_image': target_imgs, 'text': text_targets}, recon_weight=0.5, text_weight=5.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            steps += 1

            if steps % 50 == 0:
                logger.info(f'Epoch {epoch} Step {steps} Loss {epoch_loss/steps:.4f}')

        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / max(1, steps)
        logger.info(f'Epoch {epoch} completed in {epoch_time:.1f}s — avg loss {avg_loss:.4f}')

        # validate
        metrics = compute_exact_and_format(model, device, val_loader, max_samples=200)
        logger.info(f"Validation — exact {metrics['exact_pct']:.2f}% format {metrics['format_pct']:.2f}% avg_sim {metrics['avg_sim']:.3f} (N={metrics['samples']})")

        # scheduler
        scheduler.step(avg_loss)

        # save checkpoint
        torch.save({'model_state': model.state_dict(), 'round': start_round + epoch, 'args': {}}, SAVE_CKPT)
        logger.info(f'Checkpoint saved: {SAVE_CKPT} (epoch {epoch})')

        # early stopping
        if metrics['exact_pct'] > best_exact:
            best_exact = metrics['exact_pct']
            stagnation = 0
        else:
            stagnation += 1
        if metrics['exact_pct'] >= target_exact:
            logger.info('Target exact match reached — stopping')
            break
        if stagnation >= early_stop_patience:
            logger.info('No improvement — early stopping')
            break

    logger.info(f'Best exact: {best_exact:.2f}%')


if __name__ == '__main__':
    train()
