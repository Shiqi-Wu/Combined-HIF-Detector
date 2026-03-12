# Scripts Layout

This directory is organized by purpose and stability.

## `train/`

Stable training entrypoints:

- `train/lstm.sh`
- `train/koopman_phi.sh`
- `train/koopman_residual.sh`

## `eval/`

Stable evaluation entrypoints:

- `eval/lstm.sh`
- `eval/dynamic_known_control.sh`
- `eval/koopman_phi.sh`
- `eval/koopman_residual.sh`

## `experimental/`

Experimental workflows that are not part of the main comparison:

- `experimental/seq2seq_simple.sh`

The old stale scripts that referenced missing evaluators were removed.
