"""UHPC wall panel GNN surrogate: model definition, geometry and inference.

The model classes and the input pipeline follow Step05_train_gnn.py and
Step07_zero_shot_transfer.py of the UHPC-IDS software exactly, so the
predictions here are identical to the trained surrogate.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
RHO_STEEL = 7.85e-6          # kg/mm3
DISP_LIMIT = 20.0            # mm
SR_LIMIT = 1.0
BASELINE = {"box": "B 120x80x6.0", "angle": "L 50x5", "material": "Q235", "S": 1.0, "Z": 1.0, "X": 1.0}

# ---- reference values used to build the 26 meta features (Step04) ----
BOX_BASE_A, BOX_BASE_Iy, BOX_BASE_Wy = 2256.0, 4381632.0, 83981.0
ANG_BASE_A, ANG_BASE_Iy, ANG_BASE_Wy = 475.0, 112503.0, 6260.0

MODELS = {
    "Base GNN (Step05): all sections": "phase1_base.pt",
    "Fine-tuned GNN (Step06): long spans": "phase1_finetuned.pt",
}


# =============================================================================
# Model (identical to the training code)
# =============================================================================
class GATConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim_per_head, n_heads, edge_dim, dropout=0.05):
        super().__init__()
        self.H = n_heads; self.D = out_dim_per_head
        self.out_dim = n_heads * out_dim_per_head
        self.W_src = nn.Linear(in_dim, self.out_dim, bias=False)
        self.W_dst = nn.Linear(in_dim, self.out_dim, bias=False)
        self.W_edge = nn.Linear(edge_dim, self.out_dim, bias=False)
        self.attn = nn.Parameter(torch.Tensor(1, n_heads, out_dim_per_head))
        self.bias = nn.Parameter(torch.zeros(self.out_dim))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.out_dim)
        self.skip = nn.Linear(in_dim, self.out_dim, bias=False)
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x, edge_index, edge_feat):
        B, N_, _ = x.shape; E = edge_index.shape[1]
        src, dst = edge_index[0], edge_index[1]
        h_src = self.W_src(x); h_dst = self.W_dst(x); h_edge = self.W_edge(edge_feat)
        h_src_e = h_src[:, src, :].view(B, E, self.H, self.D)
        h_dst_e = h_dst[:, dst, :].view(B, E, self.H, self.D)
        h_edge_v = h_edge.view(B, E, self.H, self.D)
        e = F.leaky_relu((h_src_e + h_dst_e + h_edge_v) * self.attn, negative_slope=0.2).sum(-1)
        exp_e = torch.exp(e - e.max(dim=1, keepdim=True).values)
        alpha_sum = torch.zeros(B, N_, self.H, device=x.device)
        for h in range(self.H):
            alpha_sum[:, :, h].scatter_add_(1, dst.unsqueeze(0).expand(B, -1), exp_e[:, :, h])
        alpha_norm = self.drop(exp_e / (alpha_sum[:, dst, :] + 1e-9))
        msg = ((h_src_e + h_edge_v) * alpha_norm.unsqueeze(-1)).view(B, E, self.H * self.D)
        out = torch.zeros(B, N_, self.H * self.D, device=x.device)
        out.scatter_add_(1, dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, self.H * self.D), msg)
        return F.elu(self.norm(out + self.bias + self.skip(x)))


class MLP(nn.Module):
    def __init__(self, dims, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.LayerNorm(dims[i + 1]), nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UHPCGATModel(nn.Module):
    def __init__(self, cfg, n_nodes, n_support, edge_index):
        super().__init__()
        self.n_nodes = n_nodes; self.n_support = n_support
        self.register_buffer("edge_index", edge_index)
        G = cfg["global_embed"]; NH = cfg["node_hidden"]; GH = cfg["gat_hidden"]
        H = cfg["gat_heads"]; NL = cfg["node_out"]; EF = cfg["edge_feat_dim"]; dp = cfg["dropout"]
        self.global_enc = MLP([cfg["global_dim"], 64, G], dropout=dp)
        self.node_init = MLP([cfg["node_feat_dim"] + G, NH, NH], dropout=dp)
        self.gat_layers = nn.ModuleList()
        in_d = NH
        for _ in range(cfg["gat_layers"]):
            self.gat_layers.append(GATConvLayer(in_d, GH, H, EF, dropout=dp)); in_d = GH * H
        self.head_disp = MLP([NL, 128, 64, 5], dropout=dp / 2)
        self.head_sr = MLP([NL * 2 + EF, 256, 128, 1], dropout=dp / 2)
        self.head_react = MLP([NL, 128, 9], dropout=dp / 2)

    def forward(self, meta, node_f, edge_f, support_idx):
        B = meta.shape[0]; ei = self.edge_index
        g = self.global_enc(meta).unsqueeze(1).expand(-1, self.n_nodes, -1)
        x = self.node_init(torch.cat([node_f, g], dim=-1))
        for gat in self.gat_layers:
            x = gat(x, ei, edge_f)
        disp_p = self.head_disp(x)
        src, dst = ei[0], ei[1]
        sr_p = self.head_sr(torch.cat([x[:, src, :], x[:, dst, :], edge_f], dim=-1)).squeeze(-1)
        react_p = self.head_react(x[:, support_idx, :]).view(B, self.n_support, 3, 3)
        return disp_p, sr_p, react_p


# =============================================================================
# Sections
# =============================================================================
def load_sections():
    box = pd.read_csv(ROOT / "data" / "box_sections.csv")
    ang = pd.read_csv(ROOT / "data" / "angle_sections.csv")
    box = box[(box.H_mm > 0) & (box.B_mm > 0) & (box.t_mm > 0) & (box.A_mm2 > 0)].reset_index(drop=True)
    ang = ang[(ang.a_mm > 0) & (ang.t_mm > 0) & (ang.A_mm2 > 0)].reset_index(drop=True)
    # Property mapping verified against the Step05 test-set predictions (max error < 0.001 mm)
    box["Iy"] = box.Ix_mm4; box["Iz"] = box.Iz_mm4
    box["Wy"] = box.Iy / (box.H_mm / 2) * 1.15; box["Wz"] = box.Iz / (box.B_mm / 2) * 1.15
    ang["Iy"] = ang.Iz_mm4; ang["Iz"] = ang.Ix_mm4; ang["Wy"] = 6260.0; ang["Wz"] = 6260.0
    return box, ang


def build_meta(b, a, material, S, Z, X):
    """26 meta features, exactly as build_meta() in Step04."""
    return np.array([
        b.H_mm, b.B_mm, b.t_mm, b.A_mm2, b.Iy, b.Iz, b.Wy, b.Wz,
        a.a_mm, a.a_mm, a.t_mm, a.A_mm2, a.Iy, a.Iz, a.Wy, a.Wz,
        0.0 if material == "Q235" else 1.0,
        b.A_mm2 / BOX_BASE_A, b.Iy / BOX_BASE_Iy, b.Wy / BOX_BASE_Wy,
        a.A_mm2 / ANG_BASE_A, a.Iy / ANG_BASE_Iy, a.Wy / ANG_BASE_Wy,
        float(S), float(Z), float(X)], dtype=np.float64)


# =============================================================================
# Surrogate
# =============================================================================
class PanelSurrogate:
    def __init__(self, model_file="phase1_base.pt"):
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.geo = json.load(open(ROOT / "data" / "panel_geometry.json"))
        self.coords0 = np.array(self.geo["node_coords"], dtype=np.float64)
        self.beams = np.array(self.geo["beams"], dtype=np.int64)
        self.pids = np.array(self.geo["beam_pids"], dtype=np.int64)
        self.sup_idx = np.array(self.geo["support_idx"], dtype=np.int64)
        self.n_nodes = len(self.coords0); self.n_beams = len(self.beams); self.n_sup = len(self.sup_idx)
        self.center = self.coords0.mean(axis=0)

        z = np.load(ROOT / "models" / "scaler_params.npz")
        self.sc = {k: z[k] for k in z.files}
        self.L_mean = float(self.sc["L_mean"][0]); self.L_std = float(self.sc["L_std"][0])

        ckpt = torch.load(ROOT / "models" / model_file, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        self.model = UHPCGATModel(cfg, self.n_nodes, self.n_sup, torch.tensor(self.beams.T, dtype=torch.long))
        self.model.load_state_dict(ckpt["model_state"]); self.model.eval()

        cn = (self.coords0 - self.sc["coord_mean"]) / self.sc["coord_std"]
        fixed = np.array(self.geo["node_is_fixed"], dtype=np.float32).reshape(-1, 1)
        self.node_f = torch.tensor(np.concatenate([cn, fixed], axis=1), dtype=torch.float32).unsqueeze(0)
        self.pid_t = torch.tensor((self.pids - 1).astype(np.float32)).view(1, self.n_beams, 1)
        self.sup_t = torch.tensor(self.sup_idx, dtype=torch.long)
        self.box, self.ang = load_sections()

    # ---- geometry ----
    def coords(self, S, Z, X):
        c = self.coords0.copy()
        cx, cz = self.center[0], self.center[2]
        x = c[:, 0] * S; y = c[:, 1] * S; zc = c[:, 2]
        return np.column_stack([cx * S + (x - cx * S) * X, y, cz + (zc - cz) * Z])

    def beam_lengths(self, xyz):
        return np.linalg.norm(xyz[self.beams[:, 1]] - xyz[self.beams[:, 0]], axis=1)

    def weight(self, b, a, L):
        Lb = L[self.pids == 1].sum(); La = L[self.pids == 2].sum()
        wb = b.A_mm2 * Lb * RHO_STEEL; wa = a.A_mm2 * La * RHO_STEEL
        return wb + wa, wb, wa

    # ---- prediction ----
    @torch.no_grad()
    def predict_batch(self, metas, L):
        """metas: (B, 26) raw; L: (n_beams,) lengths of the transformed geometry."""
        B = metas.shape[0]
        ms = (metas - self.sc["meta_m"]) / self.sc["meta_s"]
        meta_t = torch.tensor(ms, dtype=torch.float32)
        Lt = torch.tensor(((L - self.L_mean) / self.L_std).astype(np.float32)).view(1, self.n_beams, 1).expand(B, -1, -1)
        ang_t = torch.tensor(ms[:, 20:23], dtype=torch.float32).view(B, 1, 3).expand(B, self.n_beams, 3)
        edge_t = torch.cat([Lt, self.pid_t.expand(B, -1, -1), ang_t], dim=-1)
        d, s, r = self.model(meta_t, self.node_f.expand(B, -1, -1), edge_t, self.sup_t)
        d = d.numpy() * self.sc["disp_s"].reshape(self.n_nodes, 5) + self.sc["disp_m"].reshape(self.n_nodes, 5)
        s = s.numpy() * self.sc["sr_s"] + self.sc["sr_m"]
        r = r.numpy()
        rx = {k: r[:, :, j, :] * self.sc[f"rxn{k}_s"].reshape(self.n_sup, 3) + self.sc[f"rxn{k}_m"].reshape(self.n_sup, 3)
              for j, k in enumerate(("DL", "WL", "Gov"))}
        return d, s, rx

    def predict(self, box_name, ang_name, material, S, Z, X):
        b = self.box[self.box.Section == box_name].iloc[0]; a = self.ang[self.ang.Section == ang_name].iloc[0]
        xyz = self.coords(S, Z, X); L = self.beam_lengths(xyz)
        d, s, rx = self.predict_batch(build_meta(b, a, material, S, Z, X)[None, :], L)
        w, wb, wa = self.weight(b, a, L)
        disp = d[0]; sr = s[0]
        return dict(xyz=xyz, L=L, disp=disp, sr=sr, rxn_DL=rx["DL"][0], rxn_WL=rx["WL"][0], rxn_gov=rx["Gov"][0],
                    max_disp=float(disp[:, 4].max()), max_sr=float(sr.max()),
                    i_disp=int(disp[:, 4].argmax()), i_sr=int(sr.argmax()),
                    weight=w, weight_box=wb, weight_ang=wa, box=b, ang=a)

    def _eval(self, bi, ai, mats, S, Z, X, L, batch=256):
        dd = np.zeros(len(bi)); ss = np.zeros(len(bi))
        for k in range(0, len(bi), batch):
            sl = slice(k, k + batch)
            metas = np.stack([build_meta(self.box.iloc[i], self.ang.iloc[j], m, S, Z, X)
                              for i, j, m in zip(bi[sl], ai[sl], mats[sl])])
            d, s, _ = self.predict_batch(metas, L)
            dd[sl] = d[:, :, 4].max(axis=1); ss[sl] = s.max(axis=1)
        return dd, ss

    def optimise(self, material_list, S, Z, X, disp_lim=DISP_LIMIT, sr_lim=SR_LIMIT,
                 max_eval=4000, n_feasible=20, slack=1.25, progress=None):
        """Two-stage lightest-design search.
        Stage 1 screens every box section with the baseline angle (L 50x5) and keeps the
        boxes whose response is within `slack` of the limits.
        Stage 2 sorts all kept-box x angle x material designs by steel weight and evaluates
        them in that order until n_feasible designs pass both limits."""
        xyz = self.coords(S, Z, X); L = self.beam_lengths(xyz)
        Lb = L[self.pids == 1].sum(); La = L[self.pids == 2].sum()
        a0 = int(np.where(self.ang.Section.values == BASELINE["angle"])[0][0])
        nb = len(self.box)
        bi = np.tile(np.arange(nb), len(material_list)); mats = np.repeat(material_list, nb)
        d1, s1 = self._eval(bi, np.full(len(bi), a0), mats, S, Z, X, L)
        keep = np.unique(bi[(d1 <= disp_lim * slack) & (s1 <= sr_lim * slack)])
        if progress: progress(0.15)
        if len(keep) == 0:
            return pd.DataFrame(), dict(screened=len(bi), kept_boxes=0, evaluated=0)
        B, A = np.meshgrid(keep, np.arange(len(self.ang)), indexing="ij")
        B = B.ravel(); A = A.ravel()
        w = (self.box.A_mm2.values[B] * Lb + self.ang.A_mm2.values[A] * La) * RHO_STEEL
        cand = pd.concat([pd.DataFrame(dict(bi=B, ai=A, material=m, weight=w)) for m in material_list])
        cand = cand.sort_values(["weight", "material"], kind="stable").reset_index(drop=True).iloc[:max_eval]
        done = []; n_ok = 0; step = 512
        for k in range(0, len(cand), step):
            c = cand.iloc[k:k + step].copy()
            c["max_disp"], c["max_sr"] = self._eval(c.bi.values, c.ai.values, c.material.values, S, Z, X, L)
            c["feasible"] = (c.max_disp <= disp_lim) & (c.max_sr <= sr_lim)
            done.append(c); n_ok += int(c.feasible.sum())
            if progress: progress(0.15 + 0.85 * min(1.0, max(n_ok / n_feasible, (k + len(c)) / len(cand))))
            if n_ok >= n_feasible: break
        res = pd.concat(done).reset_index(drop=True)
        res["box"] = self.box.Section.values[res.bi]; res["angle"] = self.ang.Section.values[res.ai]
        return res, dict(screened=len(bi), kept_boxes=len(keep), evaluated=len(res))
