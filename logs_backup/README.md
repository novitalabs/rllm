# DeepSWE Training Logs Backup

Created: 2026-02-26

## Directory Structure

```
logs_backup/
├── ray_sessions/              # Compressed Ray worker logs
│   ├── session_2026-02-22_14-49-49_472914.tar.gz    (190K)  # Run 5 - Option A TP=8 attempt
│   ├── session_2026-02-22_15-43-40_745323.tar.gz    (6.7M)  # Run 5/6 - Option B early attempts
│   ├── session_2026-02-22_17-27-14_917894.tar.gz    (1.9M)  # Run 6 - batch=1 test
│   ├── session_2026-02-22_18-39-40_482380.tar.gz    (1.9M)  # Run 7 - crypto miner impacted
│   ├── session_2026-02-22_19-21-47_425549.tar.gz    (2.4M)  # Run 7 restart
│   ├── session_2026-02-22_20-17-23_804434.tar.gz    (8.1M)  # Run 7 continued (miner killed)
│   ├── session_2026-02-22_23-53-32.tar.gz           (35M)   # Run 8 START (steps 0-~60)
│   ├── session_2026-02-23_15-03-44.tar.gz           (72M)   # Run 8 RESTART (steps ~60-~140)
│   ├── session_2026-02-24_09-25-28.tar.gz           (101M)  # Run 8 FINAL (steps ~140-200, completed)
│   └── session_2026-02-26_11-43-21.tar.gz           (25M)   # Post-training evaluation session
│
├── hydra_configs/             # Hydra output configs for every launch attempt
│   └── outputs/
│       ├── 2026-02-10/        # 23 sessions: batch=8→2, tp=8 (single-node experiments)
│       ├── 2026-02-11/        # 4 sessions: batch=2, tp=8
│       ├── 2026-02-12/        # 15 sessions: batch=4, tp=8 (Option A experiments)
│       ├── 2026-02-13/        # 1 session: batch=4, tp=16 (Option B first attempt)
│       ├── 2026-02-14/        # 2 sessions: batch=4, tp=16
│       ├── 2026-02-17/        # 2 sessions: batch=4, tp=16 (Run 1-2)
│       ├── 2026-02-18/        # 2 sessions: batch=4, tp=16 (Run 2-3)
│       ├── 2026-02-19/        # 1 session: batch=4, tp=16 (Run 3)
│       ├── 2026-02-20/        # 2 sessions: batch=4, tp=16 (Run 3-4)
│       ├── 2026-02-21/        # 12 sessions: batch=4, tp=16 (Run 4-5 debug)
│       ├── 2026-02-22/        # 50 sessions: batch=4, tp=8/16 (Run 5-8 debug+production)
│       ├── 2026-02-23/        # 1 session: batch=4, tp=16 (Run 8 restart)
│       └── 2026-02-24/        # 1 session: batch=4, tp=16 (Run 8 final restart)
│
├── training_scripts/          # All training shell scripts
│   ├── train_deepswe_full.sh          # Final production training script (Option B)
│   ├── train_deepswe_docker.sh        # Docker-based single-node variant
│   ├── train_deepswe_4docker.sh       # 4-Docker container variant
│   ├── train_deepswe_multinode.sh     # Multi-node training script
│   ├── train_deepswe_multinode_r2egym.sh
│   ├── train_deepswe_r2egym.sh
│   ├── train_deepswe_eager.sh         # Eager execution variant
│   ├── train_deepswe_minimal.sh       # Minimal test config
│   └── train_deepswe_single.sh        # Single-node variant
│
├── experimental_scripts/      # Stashed experimental scripts (from git stash)
│   ├── train_deepswe_no_ckpt.sh       # No-checkpoint variant
│   ├── train_deepswe_tp8.sh           # TP=8 Option A variant
│   ├── train_deepswe_tp8_force_local.sh
│   └── train_minimal_test.sh
│
├── diagnostic_reports/        # Stashed diagnostic reports (from git stash)
│   ├── STATUS_REPORT.md
│   ├── debugging_summary.md
│   ├── final_diagnosis.md
│   ├── immediate_findings.md
│   ├── training_status.md
│   └── vllm_throughput_analysis.md
│
└── misc_logs/
    ├── vllm_server.log        # Early vLLM standalone server test (293K)
    └── monitor_output.log     # Early Docker monitor output (11K)
```

## Key Training Sessions

### Run 8 (Production Run - 200 Steps Completed)

The only run that completed the full 200 training steps:

| Ray Session | Hydra Config | Steps | Notes |
|------------|-------------|-------|-------|
| `session_2026-02-22_23-53-32` | `2026-02-22/23-54-23` | 0 → ~60 | Initial start |
| `session_2026-02-23_15-03-44` | `2026-02-23/15-04-21` | ~60 → ~140 | Restart after OOM |
| `session_2026-02-24_09-25-28` | `2026-02-24/09-25-56` | ~140 → 200 | Final segment, completed |

### Configuration Evolution

| Phase | Dates | batch_size | TP | Architecture |
|-------|-------|------------|-----|-------------|
| Early single-node | Feb 10-12 | 2-8 | 8 | Option A (TP=8, single node) |
| Cross-node TP | Feb 13-22 | 4 | 16 | Option B (TP=16, cross-node) |
| Production (Run 8) | Feb 22-25 | 4 | 16 | Option B final config |

## Notes

- Ray session logs in `/tmp/ray/` are ephemeral and will be lost on reboot
- Checkpoint data (2.2TB in `/home/claude/work/rllm-origin/checkpoints/`) is NOT included in this backup
- All `train_agent_ppo.log` files in hydra_configs are empty (0 bytes) - actual training logs are in Ray sessions
- Total backup size: ~257MB (compressed)
