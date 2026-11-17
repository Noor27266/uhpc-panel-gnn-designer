"""UHPC Panel GNN Designer: Streamlit interface.

Run locally:   streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_js_eval import streamlit_js_eval

from uhpc_core import (BASELINE, DISP_LIMIT, MODELS, SR_LIMIT, PanelSurrogate)

st.set_page_config(page_title="UHPC Panel GNN Designer", page_icon="🏗️", layout="wide",
                   initial_sidebar_state="expanded")

# =============================================================================
# Style
# =============================================================================
ACCENT = "#1f6feb"; OK = "#1a7f37"; BAD = "#cf222e"; INK = "#1f2328"; MUTED = "#59636e"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 2.6rem; padding-bottom: 1rem; max-width: 1600px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
div[data-testid="stElementContainer"]:has(iframe[title*="streamlit_js_eval"]) {{ position: absolute; height: 0; overflow: hidden; }}
@media (max-height: 850px), (max-width: 1400px) {{
  .hero p, .hero .tags {{ display: none; }}
  .hero {{ padding: 9px 18px; }}
  .hero h1 {{ font-size: 1.12rem; }}
  .kpi .sub {{ display: none; }}
  .kpi .val {{ font-size: 1.25rem; }}
  .block-container {{ padding-top: 2.2rem; }}
}}
.side-title {{ font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: #1f6feb;
  font-weight: 700; margin: 1.0rem 0 0.25rem 0; border-top: 1px solid rgba(128,128,128,0.25); padding-top: 0.7rem; }}
.hero {{ background: linear-gradient(120deg, #0d1b2a 0%, #1b3a5c 55%, #1f6feb 100%); border-radius: 16px;
  padding: 12px 22px; color: white; margin-bottom: 10px; gap: 16px; flex-wrap: wrap; display: flex; justify-content: space-between; align-items: center; }}
.hero h1 {{ font-size: 1.3rem; margin: 0; font-weight: 700; color: white; }}
.hero p {{ margin: 2px 0 0 0; color: #c9dcf2; font-size: 0.82rem; }}
.hero .tag {{ background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.25); border-radius: 999px;
  padding: 4px 11px; font-size: 0.75rem; color: #fff; margin: 2px 0 2px 6px; display: inline-block; white-space: nowrap; }}
.kpi {{ background: white; border: 1px solid #d8dee4; border-radius: 12px; padding: 9px 14px 8px 14px;
  box-shadow: 0 1px 2px rgba(31,35,40,0.04); height: 100%; }}
.kpi .lab {{ font-size: 0.74rem; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }}
.kpi .val {{ font-size: 1.4rem; font-weight: 700; color: {INK}; line-height: 1.25; font-family: 'Inter'; }}
.kpi .val small {{ font-size: 0.9rem; color: {MUTED}; font-weight: 500; }}
.kpi .sub {{ font-size: 0.76rem; color: {MUTED}; margin-top: 2px; }}
.bar {{ height: 5px; background: #eef1f4; border-radius: 4px; margin-top: 6px; overflow: hidden; }}
.bar > div {{ height: 100%; border-radius: 4px; }}
.pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }}
.pass {{ background: #dafbe1; color: {OK}; }} .fail {{ background: #ffebe9; color: {BAD}; }}
.warn {{ background: #fff8c5; color: #7d4e00; }}
.verdict {{ border-radius: 12px; padding: 7px 14px; font-weight: 600; font-size: 0.92rem; margin: 2px 0 2px 0; }}
.v-ok {{ background: #dafbe1; color: {OK}; border: 1px solid #aceebb; }}
.v-bad {{ background: #ffebe9; color: {BAD}; border: 1px solid #ffcecb; }}
.card {{ background: white; border: 1px solid #d8dee4; border-radius: 14px; padding: 14px 18px; }}
.small {{ font-size: 0.84rem; color: {MUTED}; }}
div[data-testid="stTabs"] button p {{ font-weight: 600; font-size: 0.95rem; }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Model
# =============================================================================
@st.cache_resource(show_spinner="Loading the GNN surrogate...")
def get_model(fname):
    return PanelSurrogate(fname)


@st.cache_data(show_spinner=False, max_entries=64)
def run_prediction(fname, box, ang, mat, S, Z, X):
    r = get_model(fname).predict(box, ang, mat, S, Z, X)
    r = dict(r); r["box"] = r["box"].to_dict(); r["ang"] = r["ang"].to_dict()
    return r


base_model = get_model(MODELS["Base GNN (Step05): all sections"])
BOXES = list(base_model.box.Section); ANGLES = list(base_model.ang.Section)

defaults = dict(box=BASELINE["box"], angle=BASELINE["angle"], material="Q235", S=1.0, Z=1.0, X=1.0)
for k, v in defaults.items():
    st.session_state.setdefault(k, v)
if "pending" in st.session_state:          # design sent from the optimiser or the baseline button
    st.session_state.update(st.session_state.pop("pending"))

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("### 🏗️ UHPC Panel Designer")
    st.markdown('<div class="small">Graph neural network surrogate of the '
                'steel-framed UHPC wall panel FEM model</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-title">Surrogate model</div>', unsafe_allow_html=True)
    model_label = st.selectbox("Model", list(MODELS), label_visibility="collapsed",
                               help="Base GNN: trained on 20,000 FEM samples covering every section in the library "
                                    "(S = 1.0 to 1.8). Fine-tuned GNN: re-trained on 90 extra FEM samples of the "
                                    "baseline sections at extreme spacing and S up to 2.0.")
    st.markdown('<div class="side-title">Sections and material</div>', unsafe_allow_html=True)
    box = st.selectbox("Main frame (box section)", BOXES, key="box")
    ang = st.selectbox("Secondary members (angle section)", ANGLES, key="angle")
    mat = st.radio("Steel grade", ["Q235", "Q345"], key="material", horizontal=True)
    st.markdown('<div class="side-title">Panel geometry</div>', unsafe_allow_html=True)
    S = st.slider("Span scale S", 1.0, 2.0, step=0.05, key="S", help="Scales the panel width and depth.")
    Z = st.slider("Vertical spacing factor Z", 0.7, 1.4, step=0.05, key="Z")
    X = st.slider("Horizontal spacing factor X", 0.7, 1.4, step=0.05, key="X")
    st.markdown('<div class="side-title">Design limits</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    disp_lim = c1.number_input("Disp. (mm)", 1.0, 100.0, DISP_LIMIT, 1.0)
    sr_lim = c2.number_input("Stress ratio", 0.1, 2.0, SR_LIMIT, 0.05)
    st.markdown("---")
    if st.button("↺  Load baseline design", width="stretch"):
        st.session_state["pending"] = dict(defaults); st.rerun()

fname = MODELS[model_label]

# browser window size, used to fit the 3D view to the screen
_vp = streamlit_js_eval(js_expressions="[window.parent.innerWidth, window.parent.innerHeight]", key="viewport")
if isinstance(_vp, list) and len(_vp) == 2 and _vp[1]:
    st.session_state["vp"] = _vp
VW, VH = st.session_state.get("vp", [1600, 900])
COMPACT = VH <= 850 or VW <= 1400
PANEL_H = int(max(260, VH - (412 if COMPACT else 442)))
res = run_prediction(fname, box, ang, mat, S, Z, X)
base = run_prediction(fname, BASELINE["box"], BASELINE["angle"], BASELINE["material"], S, Z, X)
ok_d = res["max_disp"] <= disp_lim; ok_s = res["max_sr"] <= sr_lim; ok = ok_d and ok_s

warn = []
if fname == "phase1_base.pt" and S > 1.8:
    warn.append("S is above the training range of the base model (1.0 to 1.8). Use the fine-tuned model for long spans.")
if fname == "phase1_base.pt" and (Z > 1.3 or X > 1.3):
    warn.append("Z or X is above the trained spacing range (0.7 to 1.3).")
if fname == "phase1_finetuned.pt" and (box != BASELINE["box"] or ang != BASELINE["angle"]):
    warn.append("The fine-tuned model was re-trained only on the baseline sections. For other sections use the base model.")

# =============================================================================
# Header and KPIs
# =============================================================================
st.markdown(f"""
<div class="hero"><div><h1>UHPC Wall Panel · GNN Structural Designer</h1></div>
<div class="tags"><span class="tag">{model_label.split(':')[0]}</span><span class="tag">{box} + {ang} + {mat}</span><span class="tag">S={S:.2f} · Z={Z:.2f} · X={X:.2f}</span></div></div>
""", unsafe_allow_html=True)
for w in warn:
    st.warning(w, icon="⚠️")


def kpi(col, label, value, unit, sub, frac=None, good=True):
    bar = ""
    if frac is not None:
        colr = OK if good else BAD
        bar = f'<div class="bar"><div style="width:{min(frac, 1) * 100:.0f}%;background:{colr}"></div></div>'
    col.markdown(f'<div class="kpi"><div class="lab">{label}</div><div class="val">{value} <small>{unit}</small></div>'
                 f'<div class="sub">{sub}</div>{bar}</div>', unsafe_allow_html=True)


dw = res["weight"] - base["weight"]
k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Max displacement", f"{res['max_disp']:.2f}", "mm",
    f"Limit {disp_lim:.0f} mm · node {base_model.geo['node_ids'][res['i_disp']]}", res["max_disp"] / disp_lim, ok_d)
kpi(k2, "Max stress ratio", f"{res['max_sr']:.3f}", "",
    f"Limit {sr_lim:.2f} · element {base_model.geo['beam_eids'][res['i_sr']]}", res["max_sr"] / sr_lim, ok_s)
kpi(k3, "Steel weight", f"{res['weight']:,.0f}", "kg",
    f"Box {res['weight_box']:,.0f} kg · angle {res['weight_ang']:,.0f} kg")
kpi(k4, "Change vs baseline", f"{dw:+,.0f}", "kg",
    f"{dw / base['weight'] * 100:+.1f} % · baseline {base['weight']:,.0f} kg")

v_cls = "v-ok" if ok else "v-bad"
v_txt = ("✔ Design PASSES both limits" if ok else
         "✘ Design FAILS: " + " and ".join(t for t, f in [("displacement", not ok_d), ("stress ratio", not ok_s)] if f))
st.markdown(f'<div class="verdict {v_cls}" style="margin-top:10px">{v_txt} '
            f'<span style="font-weight:500">· utilisation: displacement {res["max_disp"] / disp_lim * 100:.0f} %, '
            f'stress {res["max_sr"] / sr_lim * 100:.0f} %</span></div>', unsafe_allow_html=True)

# =============================================================================
# 3D panel
# =============================================================================
def style2d(fig, title, xt, yt, height, legend=False):
    """Uniform, readable style for the 2D charts (fixed dark text on white, room for the axis labels)."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=INK), x=0.02, xanchor="left", y=0.98, yanchor="top"),
        height=height, margin=dict(l=62, r=18, t=46, b=56), paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=12, color=INK), showlegend=legend,
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=1.0, xanchor="right", font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", font_size=12))
    for upd, t in ((fig.update_xaxes, xt), (fig.update_yaxes, yt)):
        upd(title=dict(text=t, font=dict(size=12, color=INK), standoff=8), automargin=True,
            tickfont=dict(size=11, color=MUTED), showline=True, linecolor="#c9d1d9", gridcolor="#eef1f4",
            zeroline=False, ticks="outside", ticklen=4, tickcolor="#c9d1d9")
    return fig


def panel_figure(r, field, deformed, amp, show_plates, show_ids, height=640):
    import plotly.colors as pc
    geo = base_model.geo
    xyz = np.array(r["xyz"]) / 1000.0
    disp = np.array(r["disp"])
    beams = base_model.beams; pids = base_model.pids
    if deformed:
        xyz = xyz + disp[:, :3] / 1000.0 * amp
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    fig = go.Figure()

    if show_plates:
        I, J, K = [], [], []
        for q in geo["plates"]:
            if len(q) == 4:
                I += [q[0], q[0]]; J += [q[1], q[2]]; K += [q[2], q[3]]
            elif len(q) == 3:
                I.append(q[0]); J.append(q[1]); K.append(q[2])
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, color="#c8d3df", opacity=0.45, flatshading=True,
                                hoverinfo="skip", name="UHPC plates", showlegend=True,
                                lighting=dict(ambient=0.9, diffuse=0.3, specular=0.05)))

    if field == "Stress ratio":
        val = np.array(r["sr"]); cmax = max(1.0, float(val.max())); title = "Stress ratio"
        cs = [[0, "#2166ac"], [0.35, "#4fa3d1"], [0.6, "#f2c94c"], [0.8, "#f08a24"], [1.0, "#b2182b"]]
    else:
        nd = disp[:, 4]; val = 0.5 * (nd[beams[:, 0]] + nd[beams[:, 1]]); cmax = max(float(val.max()), 1e-6)
        title = "Disp. (mm)"; cs = pc.sequential.Turbo
    nb = 16
    bins = np.clip((val / cmax * nb).astype(int), 0, nb - 1)
    cols = pc.sample_colorscale(cs, [(b + 0.5) / nb for b in range(nb)])
    for b in range(nb):
        idx = np.where(bins == b)[0]
        for pid, wdt in ((1, 6), (2, 3)):
            sel = idx[pids[idx] == pid]
            if len(sel) == 0:
                continue
            X_ = np.full(len(sel) * 3, None); Y_ = X_.copy(); Z_ = X_.copy()
            X_[0::3] = x[beams[sel, 0]]; X_[1::3] = x[beams[sel, 1]]
            Y_[0::3] = y[beams[sel, 0]]; Y_[1::3] = y[beams[sel, 1]]
            Z_[0::3] = z[beams[sel, 0]]; Z_[1::3] = z[beams[sel, 1]]
            fig.add_trace(go.Scatter3d(x=X_, y=Y_, z=Z_, mode="lines", line=dict(color=cols[b], width=wdt),
                                       hoverinfo="skip", showlegend=False))
    # hover targets at member mid-points (same colour as the member, so they blend into the line)
    mid = 0.5 * (xyz[beams[:, 0]] + xyz[beams[:, 1]])
    sec = np.where(pids == 1, r["box"]["Section"], r["ang"]["Section"])
    txt = [f"<b>Element {e}</b> ({'box' if p == 1 else 'angle'})<br>{s}<br>L = {l:.0f} mm<br>SR = {v:.3f}<br>"
           f"Disp = {dv:.2f} mm" for e, p, s, l, v, dv in
           zip(geo["beam_eids"], pids, sec, r["L"], r["sr"], 0.5 * (disp[beams[:, 0], 4] + disp[beams[:, 1], 4]))]
    fig.add_trace(go.Scatter3d(x=mid[:, 0], y=mid[:, 1], z=mid[:, 2], mode="markers",
                               marker=dict(size=2.5, color=[cols[b] for b in bins], opacity=1),
                               text=txt, hovertemplate="%{text}<extra></extra>", showlegend=False))
    # colour bar (dummy trace)
    fig.add_trace(go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]], mode="markers", hoverinfo="skip", showlegend=False,
                               marker=dict(size=0.1, color=[0], colorscale=cs, cmin=0, cmax=cmax, opacity=0,
                                           colorbar=dict(title=dict(text=title, side="right", font=dict(size=12, color=INK)),
                                                         thickness=12, len=0.75, x=0.99, y=0.48,
                                                         tickfont=dict(size=11, color=INK), outlinewidth=0))))
    # supports
    si = base_model.sup_idx
    fig.add_trace(go.Scatter3d(x=x[si], y=y[si], z=z[si], mode="markers", name="Supports",
                               marker=dict(size=4, symbol="diamond", color="#0d1b2a", line=dict(color="white", width=1)),
                               text=[f"<b>Support node {n}</b><br>FZ (DL) = {f:.2f} kN" for n, f in
                                     zip(geo["support_node_ids"], np.array(r["rxn_DL"])[:, 2])],
                               hovertemplate="%{text}<extra></extra>"))
    # critical points (values shown in the title line, not on the model)
    i_d = r["i_disp"]; i_s = r["i_sr"]
    fig.add_trace(go.Scatter3d(x=[x[i_d]], y=[y[i_d]], z=[z[i_d]], mode="markers", name="Max displacement",
                               marker=dict(size=7, color="#ff7b00", line=dict(color="white", width=2)),
                               hovertemplate=f"<b>Max displacement</b><br>node {geo['node_ids'][i_d]}<br>{r['max_disp']:.2f} mm<extra></extra>"))
    m = mid[i_s]
    fig.add_trace(go.Scatter3d(x=[m[0]], y=[m[1]], z=[m[2]], mode="markers", name="Max stress ratio",
                               marker=dict(size=7, color="#c00020", symbol="square", line=dict(color="white", width=2)),
                               hovertemplate=f"<b>Max stress ratio</b><br>element {geo['beam_eids'][i_s]}<br>{r['max_sr']:.3f}<extra></extra>"))
    if show_ids:
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="text", text=[str(n) for n in geo["node_ids"]],
                                   textfont=dict(size=9, color="#30363d"), hoverinfo="skip", showlegend=False))
    xr = float(np.ptp(x)); zr = float(np.ptp(z)); yr = float(np.ptp(y))
    ax = dict(showbackground=False, gridcolor="#e6eaef", zeroline=False, showspikes=False,
              tickfont=dict(size=10, color=MUTED), title_font=dict(size=12, color=INK))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=34, b=0), paper_bgcolor="white",
        title=dict(text=f"<span style='color:#ff7b00'>●</span> max displacement <b>{r['max_disp']:.2f} mm</b> (node {geo['node_ids'][i_d]})"
                        f"   <span style='color:#c00020'>■</span> max stress ratio <b>{r['max_sr']:.3f}</b> (element {geo['beam_eids'][i_s]})",
                   font=dict(size=13, color=INK), x=0.02, xanchor="left", y=0.985, yanchor="top"),
        scene=dict(xaxis=dict(title="X (m)", nticks=6, **ax), zaxis=dict(title="Z (m)", nticks=6, **ax),
                   yaxis=dict(title="", showticklabels=False, nticks=2, **ax),
                   aspectmode="manual", aspectratio=dict(x=2.3, y=2.3 * max(0.08, 3 * yr / xr), z=2.3 * zr / xr),
                   domain=dict(x=[0, 0.9], y=[0.07, 0.97]),
                   camera=dict(eye=dict(x=0.45, y=-1.3, z=0.38), center=dict(x=0, y=0, z=-0.05),
                               up=dict(x=0, y=0, z=1), projection=dict(type="orthographic"))),
        legend=dict(orientation="h", y=0.0, yanchor="bottom", x=0.02, font=dict(size=11, color=INK),
                    bgcolor="rgba(255,255,255,0.85)", itemsizing="constant"),
        hoverlabel=dict(bgcolor="white", font_size=12))
    return fig


tab3d, tabR, tabB, tabO, tabM = st.tabs(["🧱  3D Panel", "📌  Support reactions", "⚖️  vs Baseline",
                                         "🎯  Lightest-design optimiser", "ℹ️  Model & validation"])

with tab3d:
    c1, c2, c3, c4, c5 = st.columns([1.4, 1, 1.2, 1, 1])
    field = c1.radio("Colour members by", ["Stress ratio", "Displacement"], horizontal=True, label_visibility="collapsed")
    deformed = c2.toggle("Deformed shape", value=False)
    amp = c3.slider("Amplification", 10, 500, 100, 10, disabled=not deformed, label_visibility="collapsed", format="×%d")
    show_plates = c4.toggle("UHPC plates", value=True)
    show_ids = c5.toggle("Node numbers", value=False,
                         help="3D view: drag to rotate, scroll to zoom, hover a member or support for its values. "
                              "Thick lines: box main frame; thin lines: angle secondary members.")
    p1, p2, p3 = st.columns([2.15, 1, 1], gap="small")
    p1.plotly_chart(panel_figure(res, field, deformed, amp, show_plates, show_ids, PANEL_H), width="stretch", key="panel3d",
                    theme=None, config=dict(displaylogo=False, toImageButtonOptions=dict(format="png", scale=2, filename="uhpc_panel")))
    sr = np.array(res["sr"]); d = np.array(res["disp"])
    fig_h = go.Figure(go.Histogram(x=sr, xbins=dict(start=0, size=0.025), marker=dict(color=ACCENT, line=dict(color="white", width=0.5))))
    fig_h.add_vline(x=sr_lim, line_dash="dash", line_color=BAD, line_width=2)
    fig_h.add_annotation(x=sr_lim, y=1, yref="paper", text="limit", showarrow=False, xanchor="right", yanchor="top",
                         font=dict(size=11, color=BAD), xshift=-4)
    style2d(fig_h, "Member stress ratios", "Stress ratio", "Number of members", PANEL_H)
    fig_h.update_xaxes(range=[0, max(sr_lim, float(sr.max())) * 1.05])
    p2.plotly_chart(fig_h, width="stretch", theme=None, config=dict(displaylogo=False))
    comps = ["DX", "DY", "DZ", "DXY", "DXYZ"]; dm = np.abs(d).max(axis=0)
    fig_c = go.Figure(go.Bar(x=comps, y=dm, marker_color=["#8fb3d9"] * 4 + [ACCENT], text=[f"{v:.2f}" for v in dm],
                             textposition="outside", textfont=dict(size=11, color=INK), cliponaxis=False))
    style2d(fig_c, "Max displacement by component", "Component", "Displacement (mm)", PANEL_H)
    fig_c.update_yaxes(range=[0, float(dm.max()) * 1.18])
    p3.plotly_chart(fig_c, width="stretch", theme=None, config=dict(displaylogo=False))

# =============================================================================
# Reactions
# =============================================================================
def rxn_table(r):
    geo = base_model.geo
    DL = np.array(r["rxn_DL"]); WL = np.array(r["rxn_WL"]); GV = np.array(r["rxn_gov"])
    return pd.DataFrame({"Node": geo["support_node_ids"],
                         "FX DL": DL[:, 0], "FY DL": DL[:, 1], "FZ DL": DL[:, 2],
                         "FX WL": WL[:, 0], "FY WL": WL[:, 1], "FZ WL": WL[:, 2],
                         "FX gov": GV[:, 0], "FY gov": GV[:, 1], "FZ gov": GV[:, 2]})


with tabR:
    T = rxn_table(res)
    s1, s2, s3 = st.columns(3)
    kpi(s1, "Σ FZ dead load", f"{T['FZ DL'].sum():.1f}", "kN", "Total vertical reaction, DL")
    kpi(s2, "Σ FY wind load", f"{T['FY WL'].sum():.1f}", "kN", "Total horizontal reaction, WL")
    kpi(s3, "Peak |FZ| governing", f"{T['FZ gov'].abs().max():.1f}", "kN",
        f"node {int(T.loc[T['FZ gov'].abs().idxmax(), 'Node'])}")
    st.write("")
    r1, r2, r3 = st.tabs(["Top 6 by FZ (DL)", "All 18 support nodes", "MIDAS reference (6 nodes)"])
    fmt = {c: "{:.3f}" for c in T.columns if c != "Node"}
    with r1:
        top = T.reindex(T["FZ DL"].abs().sort_values(ascending=False).index).head(6)
        st.dataframe(top.style.format(fmt).background_gradient(subset=["FZ DL"], cmap="Blues"),
                     width="stretch", hide_index=True)
    with r2:
        fig_r = go.Figure()
        for c, colr in [("FZ DL", ACCENT), ("FY WL", "#f4a259"), ("FZ gov", "#59636e")]:
            fig_r.add_trace(go.Bar(x=T.Node.astype(str), y=T[c], name=c, marker_color=colr))
        style2d(fig_r, "Support reactions (DL, WL, governing)", "Support node", "Reaction (kN)", 360, legend=True)
        fig_r.update_layout(barmode="group"); fig_r.update_xaxes(type="category")
        st.plotly_chart(fig_r, width="stretch", theme=None)
        st.dataframe(T.style.format(fmt), width="stretch", hide_index=True, height=420)
        st.download_button("⬇ Download reactions (CSV)", T.to_csv(index=False).encode(), "uhpc_support_reactions.csv",
                           "text/csv")
    with r3:
        md = base_model.geo["midas"]
        pos = [base_model.geo["support_node_ids"].index(n) for n in md["nodes"]]
        mt = pd.DataFrame({"Node": md["nodes"], "MIDAS FZ DL": md["FZ_DL"], "GNN FZ DL": T["FZ DL"].values[pos],
                           "MIDAS FY WL": md["FY_WL"], "GNN FY WL": T["FY WL"].values[pos]})
        mt["FZ error %"] = (mt["GNN FZ DL"] - mt["MIDAS FZ DL"]) / mt["MIDAS FZ DL"].abs() * 100
        is_base = (box == BASELINE["box"] and ang == BASELINE["angle"] and mat == "Q235" and S == 1 and Z == 1 and X == 1)
        if not is_base:
            st.info("The MIDAS Gen reference values belong to the baseline design (B 120x80x6.0 + L 50x5 + Q235, "
                    "S = Z = X = 1). Load the baseline design for a like-for-like comparison.", icon="ℹ️")
        st.dataframe(mt.style.format({c: "{:.3f}" for c in mt.columns if c != "Node"} | {"FZ error %": "{:+.2f}"}),
                     width="stretch", hide_index=True)
        st.markdown(f'<div class="small">MIDAS Gen baseline: max displacement {md["fem_max_disp_mm"]} mm, '
                    f'max stress ratio {md["fem_max_sr"]}.</div>', unsafe_allow_html=True)

# =============================================================================
# Baseline comparison
# =============================================================================
with tabB:
    rows = [("Max displacement (mm)", base["max_disp"], res["max_disp"]),
            ("Max stress ratio", base["max_sr"], res["max_sr"]),
            ("Steel weight (kg)", base["weight"], res["weight"]),
            ("Box frame weight (kg)", base["weight_box"], res["weight_box"]),
            ("Angle members weight (kg)", base["weight_ang"], res["weight_ang"]),
            ("Σ FZ dead load (kN)", float(np.sum(np.array(base["rxn_DL"])[:, 2])), float(np.sum(np.array(res["rxn_DL"])[:, 2]))),
            ("Σ FY wind load (kN)", float(np.sum(np.array(base["rxn_WL"])[:, 1])), float(np.sum(np.array(res["rxn_WL"])[:, 1])))]
    cmp = pd.DataFrame(rows, columns=["Quantity", "Baseline", "Current design"])
    cmp["Change %"] = (cmp["Current design"] - cmp["Baseline"]) / cmp["Baseline"].abs() * 100
    a, b = st.columns([1.1, 1])
    with a:
        st.markdown(f"**Baseline:** {BASELINE['box']} + {BASELINE['angle']} + Q235 at the same S, Z, X  \n"
                    f"**Current:** {box} + {ang} + {mat}")
        st.dataframe(cmp.style.format({"Baseline": "{:,.3f}", "Current design": "{:,.3f}", "Change %": "{:+.1f}"}),
                     width="stretch", hide_index=True)
    with b:
        lab = ["Displacement", "Stress ratio", "Weight"]
        bv = [base["max_disp"] / disp_lim, base["max_sr"] / sr_lim, 1.0]
        cv = [res["max_disp"] / disp_lim, res["max_sr"] / sr_lim, res["weight"] / base["weight"]]
        fb = go.Figure([go.Bar(name="Baseline", x=lab, y=bv, marker_color="#8fb3d9", text=[f"{v:.2f}" for v in bv],
                               textposition="outside"),
                        go.Bar(name="Current", x=lab, y=cv, marker_color=ACCENT, text=[f"{v:.2f}" for v in cv],
                               textposition="outside")])
        fb.add_hline(y=1.0, line_dash="dash", line_color=BAD)
        style2d(fb, "Utilisation and weight ratio", "", "Ratio", 360, legend=True)
        fb.update_layout(barmode="group"); fb.update_traces(textfont=dict(size=11, color=INK), cliponaxis=False)
        st.plotly_chart(fb, width="stretch", theme=None)
    sb = np.array(base["sr"]); sc = np.array(res["sr"])
    fs = go.Figure(go.Scatter(x=sb, y=sc, mode="markers", marker=dict(size=6, color=np.where(base_model.pids == 1, ACCENT, "#f4a259")),
                              text=[f"element {e}" for e in base_model.geo["beam_eids"]],
                              hovertemplate="%{text}<br>baseline %{x:.3f}<br>current %{y:.3f}<extra></extra>"))
    mx = max(sb.max(), sc.max()) * 1.05
    fs.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color="#59636e", dash="dash"), hoverinfo="skip"))
    style2d(fs, "Member stress ratio: current vs baseline (blue box, orange angle)", "Baseline stress ratio",
            "Current stress ratio", 380)
    st.plotly_chart(fs, width="stretch", theme=None)

# =============================================================================
# Optimiser
# =============================================================================
with tabO:
    st.markdown("Find the **lightest box + angle + steel grade** combination that satisfies the limits at the "
                "current geometry (S, Z, X). Designs are screened in order of increasing steel weight with the GNN.")
    o1, o2, o3 = st.columns(3)
    mats = o1.multiselect("Steel grades", ["Q235", "Q345"], default=["Q235", "Q345"])
    margin = o2.slider("Safety margin on limits (%)", 0, 30, 10,
                       help="Covers the surrogate error: the search uses limits reduced by this margin.")
    n_want = o3.slider("Feasible designs to list", 5, 50, 15)
    if st.button("🚀  Run optimiser", type="primary", disabled=not mats):
        bar = st.progress(0.0, text="Screening sections...")
        m = get_model(fname)
        out, info = m.optimise(mats, S, Z, X, disp_lim * (1 - margin / 100), sr_lim * (1 - margin / 100),
                               n_feasible=n_want, progress=lambda f: bar.progress(f, text="Evaluating designs..."))
        bar.empty()
        st.session_state["opt"] = (out, info, (S, Z, X, disp_lim, sr_lim, margin, fname))
    if "opt" in st.session_state:
        out, info, cfg = st.session_state["opt"]
        if cfg[:3] != (S, Z, X) or cfg[6] != fname:
            st.info("Geometry or model changed since this search. Run the optimiser again to update.", icon="ℹ️")
        if out.empty or not out.feasible.any():
            st.error("No feasible design found within the evaluation budget.")
        else:
            feas = out[out.feasible].sort_values("weight").head(n_want).reset_index(drop=True)
            best = feas.iloc[0]
            q1, q2, q3, q4 = st.columns(4)
            kpi(q1, "Lightest design", f"{best.weight:,.0f}", "kg", f"{best.box} + {best.angle} + {best.material}")
            kpi(q2, "Saving vs baseline", f"{base['weight'] - best.weight:,.0f}", "kg",
                f"{(base['weight'] - best.weight) / base['weight'] * 100:.1f} % lighter")
            kpi(q3, "Its max displacement", f"{best.max_disp:.2f}", "mm", f"limit used {cfg[3] * (1 - cfg[5] / 100):.1f} mm")
            kpi(q4, "Its max stress ratio", f"{best.max_sr:.3f}", "", f"limit used {cfg[4] * (1 - cfg[5] / 100):.2f}")
            st.write("")
            t1, t2 = st.columns([1.05, 1])
            show = feas[["box", "angle", "material", "weight", "max_disp", "max_sr"]].rename(columns=dict(
                box="Box", angle="Angle", material="Grade", weight="Weight (kg)", max_disp="Max disp (mm)", max_sr="Max SR"))
            show.index = show.index + 1
            t1.dataframe(show.style.format({"Weight (kg)": "{:,.0f}", "Max disp (mm)": "{:.2f}", "Max SR": "{:.3f}"}),
                         width="stretch", height=420)
            pick = t1.selectbox("Send a design to the panel view", show.index,
                                format_func=lambda i: f"#{i}: {show.loc[i, 'Box']} + {show.loc[i, 'Angle']} + {show.loc[i, 'Grade']}")
            if t1.button("Apply selected design"):
                row = feas.iloc[pick - 1]
                st.session_state["pending"] = dict(box=row.box, angle=row.angle, material=row.material); st.rerun()
            fo = go.Figure()
            for f, colr, nm in [(False, "#d0d7de", "Fails"), (True, OK, "Passes")]:
                sub = out[out.feasible == f]
                fo.add_trace(go.Scattergl(x=sub.weight, y=sub.max_disp, mode="markers", name=nm,
                                          marker=dict(size=5, color=colr, opacity=0.7),
                                          text=sub.box + " + " + sub.angle + " + " + sub.material,
                                          hovertemplate="%{text}<br>%{x:,.0f} kg<br>%{y:.2f} mm<extra></extra>"))
            fo.add_hline(y=cfg[3] * (1 - cfg[5] / 100), line_dash="dash", line_color=BAD)
            style2d(fo, f"{info['evaluated']:,} designs evaluated ({info['kept_boxes']} box sections kept)",
                    "Steel weight (kg)", "Max displacement (mm)", 440, legend=True)
            fo.update_yaxes(range=[0, cfg[3] * 2])
            t2.plotly_chart(fo, width="stretch", theme=None)

# =============================================================================
# Model information
# =============================================================================
with tabM:
    a, b = st.columns([1, 1])
    with a:
        st.markdown("#### Surrogate model")
        st.markdown("""
- **Architecture:** edge-aware graph attention network, 3 GAT layers × 4 heads, 772,623 parameters
- **Graph:** 167 nodes, 244 beam members (box main frame + angle secondary members), 18 supports, 109 UHPC plates
- **Inputs:** 26 design features (box and angle section properties, steel grade, span scale S, spacing factors Z and X), node coordinates, member lengths
- **Outputs:** displacement of every node (DX, DY, DZ, DXY, DXYZ), stress ratio of every member, DL / WL / governing reactions at all supports
- **Training data:** 20,000 FEM samples, S ∈ {1.0, 1.2, 1.5, 1.8}, Z, X ∈ [0.7, 1.3]
- **Fine-tuned model:** heads re-trained on 90 FEM samples of the baseline sections at extreme spacing and S up to 2.0
        """)
        st.markdown("#### Test-set accuracy (2,000 unseen FEM samples)")
        st.dataframe(pd.DataFrame({
            "Quantity": ["Max node displacement (mm)", "Max stress ratio", "Σ FZ dead load (kN)", "Σ FY wind load (kN)",
                         "All node DXYZ (mm)", "All member SR", "All reactions, governing (kN)"],
            "MAE": [0.2218, 0.0161, 0.8735, 0.1476, 0.0700, 0.0060, 0.0747],
            "R²": [0.99947, 0.99949, 0.99968, 0.99975, 0.99955, 0.99947, 0.99978]}).style.format(
            {"MAE": "{:.4f}", "R²": "{:.5f}"}), width="stretch", hide_index=True)
    with b:
        st.markdown("#### Validation of this application")
        st.markdown("The inference pipeline of this app was checked against the original training scripts "
                    "(`python validate_models.py`):")
        st.dataframe(pd.DataFrame({
            "Check": ["2,000 Step05 test samples (base GNN)", "MIDAS baseline, base GNN",
                      "Step07 points after fine-tuning (4 cases)"],
            "Reference": ["Step05 predictions", "3.6184 mm / 0.6205", "Step07 GNN results"],
            "This app": ["max diff 0.010 mm / 0.0005 SR", "3.6184 mm / 0.6205", "identical to 4 decimals"]}),
            width="stretch", hide_index=True)
        st.markdown("#### MIDAS Gen baseline (FEM)")
        st.markdown("Baseline B 120x80x6.0 + L 50x5 + Q235: max displacement **3.4495 mm**, max stress ratio **0.5981**. "
                    "Base GNN: 3.6184 mm (+4.9 %), 0.6205 (+3.8 %).")

st.sidebar.markdown('<div class="small" style="margin-top:14px">Predictions are surrogate estimates; confirm the final '
                    'design with FEM.</div>', unsafe_allow_html=True)
