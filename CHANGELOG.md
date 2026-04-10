# Changelog

## 0.2.0 - 2026-04-11

### Added

- Added `ddz.__version__` version metadata
- Added `requirements.txt` generated from the `DDZ` environment
- Added `play_history` support to `GameState`, `TurnView`, `EnvObservation`, and `SimulationResult`
- Added `HistoryRecord` to keep chronological action history, including passes
- Added simple premium-threat memory inference for unseen bombs and rocket
- Added seat-specific opponent premium-threat memory

### Changed

- Expanded code comments and function docstrings across the core modules
- Updated README to reflect the latest project scope and version
- Extended focused tests for strategy memory and environment history tracking

### Notes

- Current strategy now uses lightweight memory features, but it is still heuristic-based and does not yet perform full search or opponent hand inference.
