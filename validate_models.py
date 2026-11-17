"""Check that the GUI reproduces the trained surrogate exactly.

1. All 2,000 Step05 test samples (predictions_cache.json) with the base model.
2. MIDAS baseline (B 120x80x6.0 + L 50x5 + Q235, S=Z=X=1) with the base model.
3. Step07 post-fine-tune test points with the fine-tuned model.
Run:  python validate_models.py
"""
import json
from pathlib import Path
import numpy as np
from uhpc_core import PanelSurrogate, build_meta, ROOT, find_file

ref = json.load(open(find_file("data", "validation_reference.json")))
out = {}

base = PanelSurrogate("phase1_base.pt")
rows = ref["step05_test_samples"]
ed, es = [], []
for s in rows:
    b = base.box[base.box.Section == s["box"]].iloc[0]; a = base.ang[base.ang.Section == s["angle"]].iloc[0]
    L = base.beam_lengths(base.coords(s["S"], s["Z"], s["X"]))
    d, sr, _ = base.predict_batch(build_meta(b, a, s["material"], s["S"], s["Z"], s["X"])[None], L)
    ed.append(d[0, :, 4].max() - s["pred_disp"]); es.append(sr[0].max() - s["pred_sr"])
ed = np.abs(ed); es = np.abs(es)
out["test_samples"] = dict(n=len(rows), max_abs_err_disp_mm=float(ed.max()), mean_abs_err_disp_mm=float(ed.mean()),
                           max_abs_err_sr=float(es.max()), mean_abs_err_sr=float(es.mean()))
print(f"[1] Step05 test set, {len(rows)} samples: max |dDisp| = {ed.max():.5f} mm, max |dSR| = {es.max():.5f}")

r = base.predict("B 120x80x6.0", "L 50x5", "Q235", 1, 1, 1)
m = ref["midas_baseline"]
out["baseline_base"] = dict(gui_disp=r["max_disp"], ref_disp=m["gnn_disp"], gui_sr=r["max_sr"], ref_sr=m["gnn_sr"])
print(f"[2] Baseline, base model: disp {r['max_disp']:.4f} (Step05 {m['gnn_disp']:.4f}), SR {r['max_sr']:.4f} (Step05 {m['gnn_sr']:.4f}); Python FEM {m['fem_disp']} mm / {m['fem_sr']}, MIDAS Gen 3.3015 mm / 0.6100")

ft = PanelSurrogate("phase1_finetuned.pt")
out["finetuned_points"] = []
for p in ref["step07_finetuned_points"]:
    r = ft.predict("B 120x80x6.0", "L 50x5", "Q235", p["S"], p["Z"], p["X"])
    out["finetuned_points"].append(dict(p, gui_disp=r["max_disp"], gui_sr=r["max_sr"]))
    print(f"[3] Fine-tuned {p['name']:<12} S={p['S']} Z={p['Z']} X={p['X']}: disp {r['max_disp']:.4f} (Step07 {p['gnn_disp']:.4f}), "
          f"SR {r['max_sr']:.4f} (Step07 {p['gnn_sr']:.4f})")
json.dump(out, open(ROOT / "validation_results.json", "w"), indent=2)
print("Saved validation_results.json")
