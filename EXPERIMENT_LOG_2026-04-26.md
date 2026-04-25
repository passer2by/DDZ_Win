# Experiment Log - 2026-04-26

## Goal

Build the first practical supervised-learning baseline for Dou Dizhu policy training, validate the training loop, and identify the next bottleneck.

## Environment

- Local machine:
  - Windows 64-bit (`AMD64`)
  - Main local GPU mentioned by user: `RTX 4070`
- Cloud machine used for training:
  - `Ubuntu 22.04.4 LTS`
  - `Python 3.11.7`
  - `PyTorch 2.2.2`
  - `CUDA 12.4`
  - `GPU: NVIDIA GeForce RTX 3080 10GB`

## Code / training pipeline work completed

- Added training pipeline modules:
  - `training/export_data.py`
  - `training/dataset.py`
  - `training/model.py`
  - `training/train_policy.py`
  - `training/policy_agent.py`
  - `training/evaluate_policy.py`
- Added progress logging for:
  - self-play export
  - training
  - evaluation
- Added train / validation split to `training.train_policy`
- Added best-checkpoint saving:
  - `training/checkpoints/policy_mlp_best.pt`
- Added CLI arguments for:
  - export data
  - training
  - evaluation
- Fixed a training-feature issue:
  - removed `did_win` from encoded input features because it leaks future outcome information and is unavailable at inference time

## Major performance optimization

- Optimized `ddz/patterns.py`
- Replaced the old brute-force subset enumeration inside `find_patterns_from_hand()` with direct pattern construction by pattern type
- Result:
  - single-game local simulation measured around `0.0128s`
  - cloud export speed improved dramatically

### Measured export speed after optimization

- Command:
  - `python -m training.export_data --num-games 100`
- Result:
  - `100` games
  - `2105` samples
  - `real 0m2.226s`

### Rough export estimates derived from that measurement

- `200` games: about `4.5s`
- `500` games: about `11s`
- `1000` games: about `22s`
- `10000` games: about `3.7 min`
- `20000` games: feasible and practical

## Datasets produced

### Small / intermediate datasets

- `20` games -> `395` samples
- `200` games -> `4211` samples

### Main large dataset

- Command:
  - `python -m training.export_data --num-games 20000`
- Result:
  - `20000` games
  - `411262` samples
  - output file:
    - `training/data/heuristic_self_play.jsonl`

### Derived landlord-only dataset

- Extracted all landlord-role samples from the large dataset
- Result:
  - `141240` landlord samples
  - file:
    - `training/data/heuristic_self_play_landlord.jsonl`

## Main-model training results

### Baseline with `200` games / `4211` samples

- Training:
  - `5` epochs
- Best validation accuracy:
  - about `0.8836`
- Evaluation against heuristic opponents:
  - `20` games
  - `6/20` wins
  - total win rate `30%`

### Large-run training with `20000` games / `411262` samples

- Command:
  - `python -m training.train_policy --epochs 5`
- Effective split:
  - train samples: `329010`
  - validation samples: `82252`
- Approximate epoch time:
  - about `612 ~ 614` seconds per epoch
- Best validation accuracy:
  - `0.9321`
- Final validation accuracy:
  - `0.9294`
- Best checkpoint:
  - `training/checkpoints/policy_mlp_best.pt`
- Final checkpoint:
  - `training/checkpoints/policy_mlp.pt`

### Important training conclusion

- On the large dataset, the best validation result appeared around epoch `3`
- Training to epoch `5` still worked, but the best model should be taken from:
  - `policy_mlp_best.pt`
- Future large runs should probably start with:
  - `3` epochs as a strong baseline
  - optionally `5` epochs for comparison

## Evaluation results

### Unified model vs heuristic opponents

- Using `policy_mlp_best.pt`
- Evaluation:
  - `20` games
- Result:
  - `6/20` wins
  - total win rate `30%`
- Seat wins:
  - `[2, 2, 2]`

### Unified model as landlord only

- Test:
  - model fixed as landlord
  - `50` games
- Result:
  - `17/50` wins
  - landlord win rate `34%`

### Unified model as farmer only

- Test:
  - model fixed as farmer
  - `50` games
- Result:
  - farmer-side wins `28/50`
  - farmer win rate `56%`
- Important note:
  - this is farmer *side* win rate with a heuristic teammate, not pure solo carry rate

### Model self-play

- All three seats used the same trained model
- `50` games
- Result:
  - landlord wins: `21`
  - farmer wins: `29`
  - farmer-side win rate `58%`

## Landlord-specialized experiment

### Setup

- Trained a landlord-only model using:
  - `training/data/heuristic_self_play_landlord.jsonl`
- Output checkpoints:
  - `training/checkpoints/policy_mlp_landlord.pt`
  - `training/checkpoints/policy_mlp_landlord_best.pt`

### Evaluation

- Landlord-specialized model as landlord
- `50` games vs heuristic farmers
- Result:
  - `16/50` wins
  - landlord win rate `32%`

### Conclusion

- This was worse than the unified model's landlord result:
  - unified landlord win rate: `34%`
  - landlord-specialized win rate: `32%`
- Therefore:
  - landlord-only training is **not** currently the best next direction
  - this experiment is still valuable because it ruled out one candidate direction

## Artifacts saved locally

- Downloaded and unpacked cloud training artifacts locally
- Local archive:
  - `ddz_training_artifacts.tar.gz`
- Extracted folder:
  - `ddz_training_artifacts/`

Key local files:

- `ddz_training_artifacts/training/checkpoints/policy_mlp.pt`
- `ddz_training_artifacts/training/checkpoints/policy_mlp_best.pt`
- `ddz_training_artifacts/training/data/heuristic_self_play.jsonl`
- `ddz_training_artifacts/training/data/smoke.jsonl`

## Current best model

- Use:
  - `training/checkpoints/policy_mlp_best.pt`
- Reason:
  - best validation result on the large unified training run

## Current project state

- Training pipeline works end-to-end
- Data generation is now fast enough for large runs
- The main bottleneck is no longer brute-force pattern generation
- Unified model already has meaningful playing strength
- Current performance is improving with more data, but gains are slowing
- Simple landlord-only specialization did not outperform the unified model

## Most important conclusions from today

1. Pattern-generation optimization was highly successful.
2. Large-scale self-play data generation is now practical.
3. The unified policy model improved substantially with large data.
4. Best validation accuracy reached about `93.21%`.
5. Unified model total win rate vs heuristic opponents reached about `30%`.
6. Unified model landlord win rate vs heuristic opponents was about `34%`.
7. Farmer-side performance is not obviously the main weakness anymore.
8. Landlord-only specialization did not beat the unified model.
9. The next improvement probably should **not** be more role splitting first.
10. The next likely directions are:
    - stronger features
    - higher-capacity model
    - smarter training objective than plain imitation only

## Recommended next steps

### Recommended next experiment order

1. Keep the unified model as the main baseline.
2. Try a larger-capacity network before more role-specific splitting.
3. Compare:
   - wider MLP
   - deeper MLP
4. Re-evaluate against heuristic opponents with:
   - total win rate
   - landlord-only win rate
   - farmer-only side win rate

### Suggested concrete follow-up experiments

- Try unified model with larger hidden size:
  - `hidden_dim=512`
- Try unified model with slightly deeper architecture
- Keep using the large unified dataset first
- Keep using `policy_mlp_best.pt` style checkpoint selection

## Quick context to paste into a future conversation

Use this summary:

> We built and validated a full supervised-learning training pipeline for a Dou Dizhu policy model. We optimized `find_patterns_from_hand()` so export speed is no longer the main bottleneck. We generated a `20000`-game dataset with `411262` samples, trained a unified `PolicyMLP`, and got best validation accuracy `0.9321`. The unified model achieved about `30%` win rate over `20` games vs heuristic opponents, and about `34%` landlord win rate over `50` games when fixed as landlord. A landlord-only specialized model reached only `32%` landlord win rate, so specialization did not help. Current best checkpoint is `training/checkpoints/policy_mlp_best.pt`. Next likely direction is higher-capacity unified models or stronger features, not more role splitting first.
