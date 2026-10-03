# Reproducible Gains, Irreproducible Rankings: audit code

Code, frozen configuration and results for

> S. Kurashkin, V. Tynchenko, A. Borodulin, V. Nelyub. *Reproducible Gains,
> Irreproducible Rankings: A 200-Run Audit of Bias-Mitigation Methods.* Workshop on
> Trustworthy Machine Learning for Fair, Private, Robust, and Explainable
> Decision-Making, IEEE ICDM Workshops (ICDMW), 2026. To appear.

## What this does

Re-trains five bias-mitigation methods for group robustness (empirical risk
minimisation, group reweighting, group-DRO, adversarial debiasing and last-layer
retraining) over 20 seeds on Waterbirds and CelebA, under a protocol that matches
backbone, epoch budget, batch size, model-selection rule and tuning budget across
methods: 200 training runs. Reproducibility is measured under reseeding of training,
with the data split and the selected hyperparameters held fixed.

## Layout

```
wgaudit/          data layer, training, metrics
runner/           pipeline entry point and the job-based runner
configs/          grid, method definitions, release-gate declaration, frozen configuration
scripts/          acquisition, preparation, tuning, aggregation, statistics, figures,
                  and the targeted experiment on adversarial debiasing
results/final/    the reported pass: per-run outputs, merged table, statistics
results/advcfg/   the targeted experiment: declaration, report and run logs
```

## Environment

```bash
bash runner/pipeline.sh wheels      # once, with network access
bash runner/pipeline.sh env         # offline install from the wheelhouse
source .venv/bin/activate
```

## Reproducing the results

```bash
bash runner/pipeline.sh check       # confirm every data mirror is reachable
bash runner/pipeline.sh probe       # report which dataset split strategy is available
bash runner/pipeline.sh data        # download
bash runner/pipeline.sh prep        # decode once into a checksummed corpus
bash runner/pipeline.sh selftest    # prove the release gate separates clean from dirty
bash runner/pipeline.sh smoke       # one unit end to end
JOBS=2 bash runner/pipeline.sh final   # the reported pass, about 60 GPU-hours
bash runner/pipeline.sh advcfg      # targeted experiment, about 5 GPU-hours
```

`configs/tuned.yaml` is the frozen configuration of the reported pass; the `tune`
stage skips every pair already present in it. `final` runs on a fresh output directory
with resume disabled, then aggregates, computes statistics and runs the release gate;
tables and figures are regenerated only if the gate passes.

## The targeted experiment on adversarial debiasing

`advcfg` trains the four next-best adversarial configurations of the CelebA tuning
run on five seeds each. Before the first unit starts, `scripts/adv_config_stability.py
prepare` writes `runs/advcfg/DECLARATION.json`: the question, the configurations, the
statistic, the threshold and its rationale, the reading rule, its error rates, and the
sentence to be printed for each possible outcome, with a digest. The report refuses to
run if the declaration was edited, if any unit started before it was written, or if
any unit used other hyperparameters or other inputs.

## What the release gate checks

A run is refused unless resume was disabled and no unit was skipped; every comparative
claim has an independent dataset; all methods saw byte-identical evaluation inputs and
an identical training set; every declared seed is present; the backbone, epochs, batch
size and tuning budget match across methods; every unit used exactly the frozen
hyperparameters; and the statistics files exist. `scripts/gate_selftest.py` proves the
gate exits 0 on a synthetic clean run and 1 on a dirty one.

## Statistical procedure

Paired comparisons use an exact enumerated sign-flip permutation test over all 2^20
sign assignments, adopted after the signed-rank test proved version-dependent under
ties. Orderings are compared with a tie-invariant definition, and pairwise decisions
use the probability of outperforming with an exact binomial interval.

## Data

Both benchmarks are public and are used under their stated terms; CelebA is released
for non-commercial research only and carries annotated protected attributes. Only
aggregate group-level metrics are produced.
