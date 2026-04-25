# DDZ_Win

Dou Dizhu decision helper and training playground.

## Version

Current version: `0.2.0`

## Current scope

- Card and pattern modeling
- Basic rule comparison
- Legal play generation
- Heuristic decision recommendations
- Play history collection for later training
- Simple opponent memory for premium threats like bombs and rocket
- Test coverage for core patterns and rules

## Latest update

- Added a `DDZ` conda environment workflow and generated `requirements.txt`
- Added function-level comments/docstrings across the main project modules
- Added play-history recording to simulator, environment, and strategy inputs
- Added simple memory inference for unseen bombs and rocket
- Added seat-specific premium-threat memory for nearby players
- Extended focused tests for history tracking and strategy memory behavior

## Quick start

```bash
python main.py
python play_against_ai.py
python -m training.export_data
python -m training.train_policy
pytest -q tests -p no:cacheprovider
```
