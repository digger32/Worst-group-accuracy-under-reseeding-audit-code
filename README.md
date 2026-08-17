# Anonymous code release

Code for an anonymous submission under triple-blind review. Author, affiliation and
funding information is deliberately absent and will be restored in the camera-ready
version if the paper is accepted.

## What this is

A seed-stability audit of five group-robustness methods on two benchmarks: 2 datasets
x 5 methods x 20 seeds = 200 independent training runs, under a protocol that matches
backbone, epoch budget, batch size, model-selection rule and tuning budget across
methods.

## Environment

```bash
bash runner/pipeline.sh wheels      # once, with network
bash runner/pipeline.sh env         # offline install from the wheelhouse
source .venv/bin/activate
```

## Reproducing the reported numbers

One command per stage; each prints what it produced.

```bash
bash runner/pipeline.sh check       # verify every mirror is reachable
bash runner/pipeline.sh probe       # which CelebA split strategy is available
bash runner/pipeline.sh data        # download
bash runner/pipeline.sh prep        # decode once into a checksummed corpus
bash runner/pipeline.sh selftest    # prove the release gate separates clean from dirty
bash runner/pipeline.sh smoke       # one unit end to end
bash runner/pipeline.sh microfast   # cheapest unit per method: timing
bash runner/pipeline.sh microheavy  # heaviest unit per method: memory ceiling
bash runner/pipeline.sh tune        # 12 trials per (dataset, method), frozen afterwards
JOBS=2 bash runner/pipeline.sh final   # the reported pass; ~60 GPU-hours
```

`final` runs on a fresh directory with resume disabled, then aggregates, computes
statistics and runs the release gate. Tables and figures are regenerated **only if
the gate passes**, so no number in the manuscript can come from an incomplete run.

## The reported results

| Artefact | Path |
|---|---|
| Per-unit outputs | `runs/final_<date>/*.json` |
| Merged table | `runs/final_<date>/results.csv` |
| Statistics | `runs/final_<date>/stats/{omnibus,posthoc}.json` |
| Figures | `runs/final_<date>/figures/*.pdf` |
| Manuscript tables and macros | `latex/generated/*.tex` |

Every number in the paper is generated from `results.csv` by
`scripts/make_tables.py`; none is typed by hand.

## What the release gate checks

The run is refused unless: resume was disabled and no unit was skipped; every
comparative claim has an independent dataset; all methods saw byte-identical
evaluation inputs and an identical training set; every declared seed is present; the
backbone, epochs, batch size and tuning budget match across methods; every unit used
exactly the frozen hyperparameters; and the statistics artefacts exist.
`scripts/gate_selftest.py` proves the gate returns 0 on a synthetic clean run and 1 on
a dirty one.

## Licences

Both benchmarks are public and are used under their stated terms; CelebA is released
for non-commercial research only. No derived attribute predictions are included.
