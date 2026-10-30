"""Train the Kronos tokenizer and predictor from scratch for one stage (PLAN.md section 5).

  uv run python -m m3_ml.kronos.train --stage 1
  uv run python -m m3_ml.kronos.train --stage 1 --reuse-tokenizer   # predictor only (PLAN.md bug log)

Checkpoints and the training log go to .local/kronos/stage{n}/.
"""
import argparse
import json
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from .config import MODEL, PREDICTOR_TRAIN, SEED, STAGES, TOKENIZER, TOKENIZER_TRAIN, VAL_WINDOWS, WORK, PROTOCOL_ID
from .dataset import WindowSampler
from .vendor import Kronos, KronosTokenizer
from .universe import WTI_INPUT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def tensors(x, stamps):
    return torch.from_numpy(x).to(DEVICE), torch.from_numpy(stamps).to(DEVICE)


def optimizer_and_schedule(model, cfg):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], betas=cfg['betas'], weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg['lr'], total_steps=cfg['max_steps'],
                                                pct_start=0.03, div_factor=10)
    return opt, sched


def run_loop(name, model, cfg, train, val_batches, train_step, val_loss, log, fixed_steps=None):
    """Stage 1 selects duration on historical validation; stage 2 never selects."""
    opt, sched = optimizer_and_schedule(model, cfg)
    best, best_state, bad, t0 = float('inf'), None, 0, time.time()
    steps = cfg['max_steps'] if fixed_steps is None else fixed_steps
    for step in range(1, steps + 1):
        model.train()
        x, stamps, _ = train.sample(cfg['batch'])
        loss = train_step(model, *tensors(x, stamps))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
        opt.step()
        sched.step()
        if fixed_steps is None and step % cfg['eval_every'] == 0:
            model.eval()
            with torch.no_grad():
                parts = [val_loss(model, *tensors(x, s)) for x, s in val_batches]
            means = {k: float(np.mean([p[k] for p in parts])) for k in parts[0]}
            v = means['val_loss']
            log.append({'part': name, 'step': step, 'train_loss': float(loss), **means,
                        'minutes': round((time.time() - t0) / 60, 2)})
            print(log[-1], flush=True)
            if v < best:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg['patience']:
                    break
    if fixed_steps is None:
        if best_state is None:
            raise ValueError("Training schedule did not reach a validation checkpoint.")
        model.load_state_dict(best_state)
        selected_step = min((r for r in log if r.get('part') == name), key=lambda r: r['val_loss'])['step']
    else:
        selected_step = fixed_steps
        best = None
    return best, selected_step


def tokenizer_step(model, x, stamps):
    (z_pre, z), bsq_loss, _, _ = model(x)
    return (F.mse_loss(z_pre, x) + F.mse_loss(z, x) + bsq_loss) / 2


def tokenizer_val(model, x, stamps):
    (_, z), _, _, _ = model(x)
    return {'val_loss': float(F.mse_loss(z, x))}


@contextmanager
def causal_cross_attention(model):
    """The vendor's fine-token cross-attention is causal only in training mode. In eval
    mode over a full sequence, position i would also attend to later positions, whose
    hidden states contain its own target. Switch that one module to training mode (its
    dropout probabilities are 0, so nothing else changes)."""
    attn = model.dep_layer.cross_attn
    assert attn.attn_dropout_p == 0 and attn.resid_dropout.p == 0
    was = attn.training
    attn.train()
    try:
        yield
    finally:
        attn.train(was)


def predictor_losses(tokenizer):
    def tokens(x):
        with torch.no_grad():
            s1, s2 = tokenizer.encode(x, half=True)
        return s1, s2

    def step(model, x, stamps):
        s1, s2 = tokens(x)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=DEVICE == 'cuda'):
            logits = model(s1[:, :-1], s2[:, :-1], stamps[:, :-1])
            loss, _, _ = model.head.compute_loss(logits[0].float(), logits[1].float(), s1[:, 1:], s2[:, 1:])
        return loss

    def val(model, x, stamps):
        # Teacher forcing on the coarse token makes the validation loss deterministic, and
        # causal cross-attention makes it the same prediction task as in training.
        s1, s2 = tokens(x)
        with causal_cross_attention(model), \
                torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=DEVICE == 'cuda'):
            logits = model(s1[:, :-1], s2[:, :-1], stamps[:, :-1], use_teacher_forcing=True, s1_targets=s1[:, 1:])
            loss, ce_s1, ce_s2 = model.head.compute_loss(logits[0].float(), logits[1].float(), s1[:, 1:], s2[:, 1:])
        return {'val_loss': float(loss), 'val_ce_s1': float(ce_s1), 'val_ce_s2': float(ce_s2)}

    return step, val


def main(stage, reuse_tokenizer=False):
    cfg = STAGES[stage]
    out = WORK / f'stage{stage}'
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    train = WindowSampler(cfg, 'train', seed=SEED)
    if stage == 1:
        val = WindowSampler(cfg, 'val', seed=SEED + 1)
        vx, vt, _ = val.sample(VAL_WINDOWS)
        val_batches = [(vx[i:i + 128], vt[i:i + 128]) for i in range(0, len(vx), 128)]
        durations = {}
    else:
        selection = json.loads((WORK / 'stage1' / 'metadata.json').read_text())
        if selection.get('protocol_id') != PROTOCOL_ID:
            raise ValueError("Stage 2 requires approved-protocol stage 1 duration selection.")
        durations = selection['selected_steps']
        val_batches = []
    log = [{'stage': stage, **cfg, 'train_tickers': len(train.series),
            'protocol_id': PROTOCOL_ID, 'reuse_tokenizer': reuse_tokenizer}]
    print(log[0], flush=True)

    tokenizer = KronosTokenizer(**TOKENIZER).to(DEVICE)
    if reuse_tokenizer:
        tokenizer.load_state_dict(torch.load(out / 'tokenizer.pt', map_location=DEVICE))
        reuse_meta = json.loads((out / 'metadata.json').read_text())
        if reuse_meta.get('protocol_id') != PROTOCOL_ID:
            raise ValueError("Cannot reuse a historical tokenizer under the new protocol.")
        durations['tokenizer'] = reuse_meta['selected_steps']['tokenizer']
    else:
        loss, duration = run_loop('tokenizer', tokenizer, TOKENIZER_TRAIN, train, val_batches,
                                  tokenizer_step, tokenizer_val, log,
                                  fixed_steps=durations.get('tokenizer'))
        log[0]['tokenizer_best_val_mse'] = loss
        durations['tokenizer'] = duration
        torch.save(tokenizer.state_dict(), out / 'tokenizer.pt')

    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    torch.manual_seed(SEED + 2)   # the predictor phase does not depend on the tokenizer phase's draws
    train.reseed(SEED + 2)
    model = Kronos(**MODEL).to(DEVICE)
    step, val_loss = predictor_losses(tokenizer)
    loss, duration = run_loop('predictor', model, PREDICTOR_TRAIN, train, val_batches,
                              step, val_loss, log, fixed_steps=durations.get('predictor'))
    log[0]['predictor_best_val_ce'] = loss
    durations['predictor'] = duration
    torch.save(model.state_dict(), out / 'predictor.pt')
    (out / 'train_log.json').write_text(json.dumps(log, indent=2, default=str) + '\n', encoding='utf-8')
    mean, std = train.statistics[WTI_INPUT]
    np.savez(out / 'normalization.npz', mean=mean, std=std)
    metadata = dict(protocol_id=PROTOCOL_ID, stage=stage, **cfg,
                    selected_steps=durations, normalization='frozen training ticker statistics')
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print('saved', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True)
    parser.add_argument('--reuse-tokenizer', action='store_true')
    args = parser.parse_args()
    main(args.stage, args.reuse_tokenizer)
