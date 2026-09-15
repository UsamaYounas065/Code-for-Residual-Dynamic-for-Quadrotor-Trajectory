# Code-for-Residual-Dynamic-for-Quadrotor-Trajectory

Reproduction code for the manuscript *An Evaluation Protocol for Learned Residual Dynamics in Quadrotor Trajectory Tracking* (Usama Younas & Jason Kurz).

This repository regenerates the simulation tables and the NeuroBEM open-loop prediction experiment reported in the paper.

## License

MIT License — see [`LICENSE`](LICENSE). Copyright (c) 2026 Usama Younas.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke test

```bash
python scripts/smoke_test_core.py
# or
python scripts/reproduce_tables.py --only smoke
```

Expect PD-only figure-eight RMSE ≈ 0.21 m (paper ≈ 0.2107).

## Reproduce manuscript tables

```bash
# Tables 1–3, open-loop prediction (Table 4 / Fig. 5c), Dryden trust sweep (Fig. 6c)
python scripts/reproduce_tables.py --only table1,table2,table3,pred,trust

# MPC / INDI tables
python scripts/reproduce_tables.py --only table5,table6,table7

# Everything
python scripts/reproduce_tables.py --all

# Faster plumbing check
python scripts/reproduce_tables.py --only table1,table2 --quick
```

Outputs are written under `results/` (`table1.csv` … `table7.csv`, `table4_prediction.csv`, `fig6c_trust_sweep.csv`, `reproduce_meta.json`). Reference CSVs from the paper runs are included for comparison.

**Seed protocols**

- Tables 1–2: rollout seeds 200–203, network seeds 1–3, 200 epochs
- Table 3: rollout seeds 300–305, network seeds 1–3; tuned \(k_i=\{4,2,2\}\)
- Fig. 6c trust sweep: Dryden network seed 1, rollout seeds 200–204
- Table 7 offset-free MPC uses `dob_tau=0.2` (not the PD+INDI \(\tau_f=0.02\,\mathrm{s}\))

## NeuroBEM prediction (Section 5.12)

Bulk flight CSVs are **not** shipped here (size). Download NeuroBEM `processed_data` from the authors’ release and place it as:

```text
data/neurobem/processed_data/*.csv
```

Dataset page: https://rpg.ifi.uzh.ch/NeuroBEM.html  
Zip: https://download.ifi.uzh.ch/rpg/NeuroBEM/

This repo already includes `data/neurobem/testset.txt`, `Flights.txt`, and the NeuroBEM readme excerpt.

```bash
python scripts/neurobem_prediction_eval.py --quick --plot
python scripts/neurobem_prediction_eval.py --epochs 60 --plot
python scripts/neurobem_prediction_eval.py --high-speed-only --speed-min 5 --plot
```

## Layout

```text
src/                 # quadrotor core + MPPI / offset-free MPC
scripts/             # reproduce_tables.py, NeuroBEM eval, smoke test
data/neurobem/       # test partition + docs (processed CSVs optional)
results/             # reference CSV outputs
figures/             # optional plots from local runs
```

## Citation

If you use this code or the NeuroBEM data, please cite the manuscript and:

Bauersfeld, Kaufmann, Foehn, Sun, Scaramuzza. *NeuroBEM: Hybrid aerodynamic quadrotor model*. Robotics: Science and Systems, 2021. https://doi.org/10.15607/RSS.2021.XVII.042
