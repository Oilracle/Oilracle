"""Fixed Kronos experiment settings (PLAN.md sections 4-6)."""
from pathlib import Path
from m3_ml.common.protocol import TRAIN_END, VALIDATION_END, FIT_END, PROTOCOL_ID

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'm3_ml/kronos/checkpoints'
if not WORK.is_dir():
    WORK = ROOT / '.local/kronos_protocol'

# Kronos-mini and Kronos-Tokenizer-2k (official Hugging Face config.json files).
TOKENIZER = dict(d_in=6, d_model=256, n_heads=4, ff_dim=512, n_enc_layers=4, n_dec_layers=4,
                 ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, s1_bits=10, s2_bits=10,
                 beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=5)
MODEL = dict(s1_bits=10, s2_bits=10, n_layers=4, d_model=256, n_heads=4, ff_dim=512, ffn_dropout_p=0.2,
             attn_dropout_p=0.0, resid_dropout_p=0.2, token_dropout_p=0.0, learn_te=True)

LOOKBACK, FUTURE = 256, 64
WINDOW = LOOKBACK + FUTURE
CLIP = 5.0
FEATURES = ['open', 'high', 'low', 'close', 'volume', 'amount']

STAGES = {
    1: {'train_end': TRAIN_END, 'val_end': VALIDATION_END, 'corpus_cutoff': VALIDATION_END},
    2: {'train_end': FIT_END, 'val_end': FIT_END, 'corpus_cutoff': FIT_END},
}

SAMPLING = {'A': 0.5, 'B': 0.5}   # share of windows from each group
ENERGY_WEIGHT = 3.0               # relative weight of the 5 energy futures within group A
VAL_WINDOWS = 2048                # fixed validation windows per stage

TOKENIZER_TRAIN = dict(lr=2e-4, weight_decay=0.1, betas=(0.9, 0.95), batch=64, max_steps=30_000,
                       eval_every=1_000, patience=5, clip_grad=2.0)
PREDICTOR_TRAIN = dict(lr=5e-4, weight_decay=0.1, betas=(0.9, 0.95), batch=64, max_steps=60_000,
                       eval_every=2_000, patience=5, clip_grad=3.0)
SEED = 20261017

HORIZON = 60
PATHS_VALIDATION, PATHS_TEST = 200, 1000
SAMPLING_GRID = [{'T': t, 'top_p': p} for t in (0.8, 1.0) for p in (0.9, 1.0)]
