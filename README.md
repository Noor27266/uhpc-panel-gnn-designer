# UHPC Panel GNN Designer

An interactive Streamlit app for the steel-framed UHPC wall panel. A graph neural network (GNN) surrogate of the FEM model predicts, in well under a second:

- the displacement of all 167 nodes,
- the stress ratio of all 244 members,
- the dead-load, wind-load and governing reactions at all 18 supports.

The panel is drawn in 3D with its UHPC plates, supports and members, coloured by stress ratio or displacement. The deformed shape can be shown.

![Panel view](screenshots/01_panel_stress_ratio.png)

## What the app does

| Tab | Content |
|---|---|
| 3D Panel | Interactive 3D model (rotate, zoom, hover), stress-ratio or displacement colouring, amplified deformed shape, critical node and member marked, stress-ratio histogram |
| Support reactions | Top 6 supports by FZ, all 18 supports (chart, table, CSV download), comparison with the MIDAS Gen reference |
| vs Baseline | Current design against the baseline B 120x80x6.0 + L 50x5 + Q235 at the same geometry |
| Lightest-design optimiser | Searches the box x angle x steel-grade library for the lightest design that meets the displacement and stress limits, then sends the chosen design to the 3D view |
| Model & validation | Architecture, test-set accuracy, validation of this app against the training scripts |

Inputs (sidebar):
- box section (291 sizes)
- angle section (212 sizes)
- steel grade Q235 / Q345
- span scale S, vertical spacing factor Z, horizontal spacing factor X
- design limits (default 20 mm and SR 1.0)

## Models

| File | Source | Use |
|---|---|---|
| `models/phase1_base.pt` | Step05 training, 20,000 FEM samples, best epoch 496 | All sections, S = 1.0 to 1.8, Z and X = 0.7 to 1.3 (default) |
| `models/phase1_finetuned.pt` | Step06 fine-tuning, 90 extra FEM samples | Baseline sections at long spans (S up to 2.0) and extreme spacing |
| `models/scaler_params.npz` | Step05 | Input and output scalers (shared by both models) |

Both models are a pure-PyTorch edge-aware GAT (3 layers x 4 heads, 772,623 parameters). torch-geometric is not needed.

## Validation

`python validate_models.py` checks this app's inference against the original results:

| Check | Reference | This app |
|---|---|---|
| 2,000 Step05 test samples, base model | `pred_max_disp_mm`, `pred_max_sr` | max difference 0.010 mm and 0.0005 (input rounding in the cache file) |
| MIDAS baseline, base model | 3.6184 mm, 0.6205 | 3.6184 mm, 0.6205 |
| Step07 test points after fine-tuning (S=2; Z=1.4; X=1.4; all three) | 25.2800 / 6.8592 / 8.9958 / 93.3713 mm | identical |

MIDAS Gen FEM for the baseline gives 3.4495 mm and 0.5981.

Section-property mapping used by the training data (Step04): box Iy = Ix_mm4, Iz = Iz_mm4; angle Iy = Iz_mm4, Iz = Ix_mm4.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on GitHub + Streamlit Community Cloud

1. Create a new GitHub repository, for example `uhpc-panel-gnn`.
2. Upload everything in this folder: `app.py`, `uhpc_core.py`, `requirements.txt`, `.streamlit/`, `data/`, `models/`, `README.md`. The `screenshots/` folder is optional. Every file is under 5 MB, so normal git works without Git LFS.
   ```bash
   git init
   git add .
   git commit -m "UHPC panel GNN designer"
   git branch -M main
   git remote add origin https://github.com/<your-user>/uhpc-panel-gnn.git
   git push -u origin main
   ```
3. Go to https://share.streamlit.io, choose **Create app**, select the repository, branch `main` and main file `app.py`.
4. Under **Advanced settings**, choose Python 3.11 and click **Deploy**.

The first build takes a few minutes. `requirements.txt` points pip to the CPU-only PyTorch wheels, which keeps the install small.

## Files

```
app.py                 Streamlit interface
uhpc_core.py           GNN model, geometry transform, inference, weight, optimiser
validate_models.py     Reproduces the Step05 / Step07 predictions
data/panel_geometry.json     nodes, members, plates, supports, MIDAS reference
data/box_sections.csv        box section library
data/angle_sections.csv      angle section library
data/validation_reference.json  reference predictions used by validate_models.py
models/                trained weights and scalers
.streamlit/config.toml theme
```

Predictions come from a surrogate model. Confirm the final design with FEM.
