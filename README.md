# Worst-group accuracy under reseeding: audit code

Code accompanying an anonymous submission under triple-blind review. Author,
affiliation, funding and acknowledgement information is deliberately absent and will
be restored in the camera-ready version.

## What this does

Re-trains five group-robustness methods --- empirical risk minimisation, group
reweighting, group-DRO, adversarial debiasing and last-layer retraining --- over 20
seeds on two benchmarks, under a protocol that matches backbone, epoch budget, batch
size, model-selection rule and tuning budget across methods. 200 independent training
runs in total.

## Layout

```
wgaudit/     data layer, training, metrics
runner/      pipeline entry point and the job-based runner
configs/     experiment grid, method definitions, release-gate declaration
scripts/     acquisition, preparation, tuning, aggregation, statistics, figures
```

## Environment

```bash
bash runner/pipeline.sh wheels      # once, with network access
bash runner/pipeline.sh env         # offline install from the wheelhouse
source .venv/bin/activate
```

Dependencies are pinned in `requirements.txt`; `constraints.txt` is installed as a
pip constraint file inside the environment so that a later install cannot move them.

## Reproducing the results

One command per stage. Each prints what it produced and refuses to continue if a
precondition fails.

```bash
bash runner/pipeline.sh check       # confirm every data mirror is reachable
bash runner/pipeline.sh probe       # report which dataset split strategy is available
bash runner/pipeline.sh data        # download
bash runner/pipeline.sh prep        # decode once into a checksummed corpus
bash runner/pipeline.sh selftest    # prove the release gate separates clean from dirty
bash runner/pipeline.sh smoke       # one unit end to end
bash runner/pipeline.sh microfast   # cheapest unit per method: timing
bash runner/pipeline.sh microheavy  # heaviest unit per method: memory ceiling
bash runner/pipeline.sh tune        # 12 trials per (dataset, method), then frozen
JOBS=2 bash runner/pipeline.sh final   # the reported pass
```

`final` runs on a fresh output directory with resume disabled, then aggregates,
computes statistics and runs the release gate.

Expect roughly 60 GPU-hours for the reported pass on a single 80 GB device. The
device is saturated by one unit, so raising `JOBS` buys little; the pilot measured
1.03x throughput on the smaller benchmark and 1.27x on the larger at `JOBS=5`.

## Outputs

| Artefact | Path |
|---|---|
| Per-unit results | `runs/<name>/*.json` |
| Merged table | `runs/<name>/results.csv` |
| Statistics | `runs/<name>/stats/{omnibus,posthoc}.json` |
| Figures | `runs/<name>/figures/*.pdf` |

## What the release gate checks

A run is refused unless: resume was disabled and no unit was skipped; every
comparative claim has an independent dataset; all methods saw byte-identical
evaluation inputs and an identical training set; every declared seed is present; the
backbone, epochs, batch size and tuning budget match across methods; every unit used
exactly the frozen hyperparameters; and the statistics artefacts exist.
`scripts/gate_selftest.py` proves the gate exits 0 on a synthetic clean run and 1 on
a dirty one, and runs as part of the smoke stage.

## Statistical procedure

Paired comparisons use an exact enumerated sign-flip permutation test over all 2^20
sign assignments. This was adopted after the signed-rank test proved
version-dependent: its exact null assumes no ties among absolute paired differences,
accuracies on a fixed test set produce such ties, and the library switches between
the exact null and a tie-corrected approximation depending on its version. Library
versions are recorded alongside the results.

## Data

Both benchmarks are public and are used under their stated terms; one of them is
released for non-commercial research only and carries annotated protected
attributes. Only aggregate group-level metrics are produced; no derived attribute
predictions are included in this repository or its outputs.
