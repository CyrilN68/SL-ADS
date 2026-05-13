import pathlib
import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full", app_title="Admin — Détection & Qualification SL")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pathlib
    import sys
    return go, make_subplots, mo, np, pd, pathlib, sys


@app.cell
def _(mo):
    mo.md("# Vue Administrateur — Détection SL + Qualification SBN")


# ── IMPORT CONFIG (source unique de vérité pour les chemins) ─────────────────
# Toute modification de VERSION_SUFFIX dans config.py se propage ici.

@app.cell
def _(pathlib, sys):
    # Resolve the project root from this notebook's own location so the
    # notebook is portable across machines and OSes.
    # Layout: <repo_root>/src/sl_ads/notebooks/<this_file>.py
    _THIS = pathlib.Path(__file__).resolve()
    _PROJ_ROOT = _THIS.parents[3]
    _SRC_DIR = _PROJ_ROOT / "src"
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    # Force reload in case ``sl_ads.config`` was already cached from a
    # different sys.path entry.
    import importlib as _importlib
    if "sl_ads.config" in sys.modules:
        _importlib.reload(sys.modules["sl_ads.config"])
    try:
        from sl_ads.config import CONFIG as _CFG  # noqa: F401
        # Default to the canonical complete run shipped with the repo.
        # Replace this path in the UI input below by your own run_id
        # after running the pipeline locally.
        RESULTS_DIR_DEFAULT = str(_PROJ_ROOT / "results" / "2e12261d55a8f975")
        config_import_ok = True
    except Exception:
        RESULTS_DIR_DEFAULT = str(_PROJ_ROOT / "results")
        config_import_ok = False
    return RESULTS_DIR_DEFAULT, config_import_ok


# ── CHARGEMENT ───────────────────────────────────────────────────────────────

@app.cell
def _(RESULTS_DIR_DEFAULT, mo, pathlib):
    _RES = pathlib.Path(RESULTS_DIR_DEFAULT)
    # Auto-detect: certaines versions génèrent detection_results.csv, d'autres _INJECTED.csv
    _det_injected = _RES / "detection_results_INJECTED.csv"
    _det_plain    = _RES / "detection_results.csv"
    if _det_injected.exists():
        _det_default = str(_det_injected)
    elif _det_plain.exists():
        _det_default = str(_det_plain)
    else:
        _det_default = str(_det_injected)   # fallback : chemin par défaut historique
    det_input = mo.ui.text(
        value=_det_default,
        label="detection_results CSV", full_width=True,
    )
    sbn_input = mo.ui.text(
        value=str(_RES / "qualif_types_sbn.csv"),
        label="qualif_types_sbn CSV", full_width=True,
    )
    return det_input, sbn_input


@app.cell
def _(det_input, mo, sbn_input):
    mo.vstack([det_input, sbn_input])


@app.cell
def _(det_input, pd):
    try:
        _d = pd.read_csv(det_input.value)
        _d["timestamp"] = pd.to_datetime(_d["timestamp"], errors="coerce")
        df_det = _d
        det_ok = True
    except Exception:
        df_det = pd.DataFrame()
        det_ok = False
    return det_ok, df_det


@app.cell
def _(pd, sbn_input):
    try:
        _s = pd.read_csv(sbn_input.value)
        _s["timestamp"] = pd.to_datetime(_s["timestamp"], errors="coerce")
        df_sbn = _s
        sbn_ok = True
    except Exception:
        df_sbn = pd.DataFrame()
        sbn_ok = False
    return df_sbn, sbn_ok


# Auto-détection raw_data
@app.cell
def _(det_input, mo, pathlib):
    _dir = pathlib.Path(det_input.value).parent
    _cands = sorted(_dir.glob("raw_data_*.csv"))
    raw_input = mo.ui.text(
        value=str(_cands[0]) if _cands else "",
        label="raw_data CSV (30s)", full_width=True,
    )
    return raw_input,


@app.cell
def _(mo, raw_input):
    mo.vstack([raw_input])


@app.cell
def _(pd, raw_input):
    try:
        _r = pd.read_csv(raw_input.value)
        _r["timestamp"] = pd.to_datetime(_r["timestamp"], errors="coerce")
        raw_data = {k: g.sort_values("timestamp").reset_index(drop=True)
                    for k, g in _r.groupby("metric_key")}
        raw_ok = True
    except Exception:
        raw_data = {}
        raw_ok = False
    return raw_data, raw_ok


@app.cell
def _(det_ok, df_det, df_sbn, mo, raw_data, raw_ok, sbn_ok):
    _msgs = []
    if det_ok: _msgs.append(f"detection: **{len(df_det)} fenêtres**")
    else: _msgs.append("detection: ❌")
    if sbn_ok: _msgs.append(f"SBN: **{int(df_sbn['gate_open'].sum())} anomalies** / {len(df_sbn)}")
    else: _msgs.append("SBN: ❌")
    if raw_ok: _msgs.append(f"raw: **{len(raw_data)} métriques**")
    else: _msgs.append("raw: non chargé (données brutes 30s indisponibles)")
    mo.callout(mo.md(" | ".join(_msgs)), kind="success" if (det_ok and sbn_ok) else "warn")


# ── STRUCTURE MÉTRIQUES ──────────────────────────────────────────────────────

@app.cell
def _(df_det):
    _all = [c[:-len("_b_safe")] for c in df_det.columns if c.endswith("_b_safe")]
    leaf_metrics = [p for p in _all if not p.startswith("METHODE") and not p.startswith("FINAL")]
    final_key    = next((p for p in _all if p.startswith("FINAL")), None)

    ATTACK_TYPES = ["UDP_FLOOD","SYN_FLOOD","ICMP_FLOOD","DNS_AMP","HTTP_FLOOD",
                    "SLOWLORIS","PORT_SCAN","DATA_EXFIL","NETWORK_OUTAGE","BGP_HIJACK",
                    "BOTNET_CC","Autre_Anomalie"]

    # FIX BUG b_sbn_cols : filtrer réellement sur les colonnes présentes (sans "or True")
    actual_types = [t for t in ATTACK_TYPES if f"b_sbn_{t}" in (df_det.columns.tolist() if len(df_det) else [])]

    TYPE_COLORS = {
        "UDP_FLOOD":"#e74c3c","SYN_FLOOD":"#c0392b","ICMP_FLOOD":"#e67e22",
        "DNS_AMP":"#f39c12","HTTP_FLOOD":"#27ae60","SLOWLORIS":"#16a085",
        "PORT_SCAN":"#2980b9","DATA_EXFIL":"#8e44ad","NETWORK_OUTAGE":"#7f8c8d",
        "BGP_HIJACK":"#d35400","BOTNET_CC":"#1abc9c","Autre_Anomalie":"#bdc3c7",
    }

    # FIX ROOT BUG raw_data : les colonnes sont déjà au format "prophet_xxx" / "reconst_xxx"
    # (pas "P_xxx" / "R_xxx") — la fonction retourne directement la clé telle quelle.
    def det_to_raw(m):
        # Cas normal train_v10 : colonnes nommées prophet_xxx / reconst_xxx
        if m.startswith("prophet_") or m.startswith("reconst_"):
            return m
        # Compatibilité ascendante notation P_/R_ (anciennes versions)
        if m.startswith("P_"): return "prophet_" + m[2:]
        if m.startswith("R_"): return "reconst_"  + m[2:]
        return None

    raw_key_map = {m: det_to_raw(m) for m in leaf_metrics}
    return ATTACK_TYPES, TYPE_COLORS, actual_types, det_to_raw, final_key, leaf_metrics, raw_key_map


# Fusion sur timestamp commun
@app.cell
def _(df_det, df_sbn, pd):
    if len(df_det) > 0 and len(df_sbn) > 0:
        _sbn_cols = [c for c in df_sbn.columns if c != "timestamp"]
        df_merged = df_det.merge(
            df_sbn[["timestamp"] + _sbn_cols],
            on="timestamp", how="left",
        )
    else:
        df_merged = df_det.copy()
    return df_merged,


# ── CONTRÔLES ───────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("---\n## Contrôles")


@app.cell
def _(mo):
    is_playing   = mo.ui.switch(label="▶ Play", value=False)
    anomaly_only = mo.ui.switch(label="Anomalies uniquement", value=True)
    return anomaly_only, is_playing


@app.cell
def _(mo):
    threshold_sl = mo.ui.slider(0.05, 0.95, value=0.15, step=0.01,
                                label="Seuil détection", show_value=True)
    speed_sel = mo.ui.dropdown(
        options={"0.5 fps":"2000ms","1 fps":"1000ms","2 fps":"500ms",
                 "5 fps":"200ms","10 fps":"150ms"},
        value="1 fps", label="Vitesse",
    )
    return speed_sel, threshold_sl


@app.cell
def _(anomaly_only, is_playing, mo, speed_sel, threshold_sl):
    mo.hstack([is_playing, anomaly_only, threshold_sl, speed_sel], gap=4)


# ── ANIMATION ───────────────────────────────────────────────────────────────

@app.cell
def _(anomaly_only, df_merged, final_key, threshold_sl):
    _col = f"{final_key}_proj_atk" if final_key else None
    if anomaly_only.value and _col and _col in df_merged.columns:
        play_indices = list(df_merged.index[df_merged[_col] > threshold_sl.value])
    else:
        play_indices = list(range(len(df_merged)))
    n_frames = len(play_indices)
    return n_frames, play_indices


@app.cell
def _(is_playing, mo, speed_sel):
    timer = mo.ui.refresh(default_interval=speed_sel.value) if is_playing.value else mo.ui.refresh()
    return timer,


@app.cell
def _(timer):
    timer


@app.cell
def _(mo):
    get_pos, set_pos = mo.state(0)
    return get_pos, set_pos


# FIX : vérification is_playing.value avant d'incrémenter (évite désynchronisation slider)
@app.cell
def _(is_playing, n_frames, set_pos, timer):
    timer
    if n_frames > 0 and is_playing.value:
        set_pos(lambda p: (p + 1) % n_frames)


@app.cell
def _(mo, n_frames):
    frame_slider = mo.ui.slider(0, max(0, n_frames - 1), value=0,
                                label="Position manuelle", show_value=True, full_width=True)
    return frame_slider,


@app.cell
def _(frame_slider, mo):
    mo.vstack([frame_slider])


@app.cell
def _(frame_slider, is_playing, set_pos):
    if not is_playing.value:
        set_pos(frame_slider.value)


@app.cell
def _(frame_slider, get_pos, is_playing, n_frames, play_indices):
    if n_frames == 0:
        current_df_idx = 0
        current_pos = 0
    elif is_playing.value:
        current_pos = get_pos() % n_frames
        current_df_idx = play_indices[current_pos]
    else:
        current_pos = frame_slider.value
        current_df_idx = play_indices[current_pos] if play_indices else 0
    return current_df_idx, current_pos


# ══════════════════════════════════════════════════════════════════════════
# KPI BANNER — Métriques opérationnelles temps réel
# Inspiré des tableaux de bord SOC (Jiang et al. 2012 ; SANS Blue Team 2020)
# ══════════════════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md("---\n## Tableau de bord opérationnel (KPIs)")


@app.cell
def _(current_df_idx, df_merged, final_key, mo, np, pd, threshold_sl):
    def _kpi_card(label, value, caption=""):
        return mo.md(
            f"""<div style="background:#1e2a3a;border:1px solid #2c3e50;border-radius:8px;
            padding:14px 20px;min-width:160px;text-align:center">
            <div style="font-size:11px;color:#7f8c8d;text-transform:uppercase;
            letter-spacing:1px">{label}</div>
            <div style="font-size:28px;font-weight:700;color:#ecf0f1;margin:6px 0">{value}</div>
            <div style="font-size:11px;color:#95a5a6">{caption}</div></div>"""
        )

    if len(df_merged) == 0 or not final_key:
        _out = mo.callout(mo.md("Aucune donnée chargée."), kind="warn")
    else:
        _pc = f"{final_key}_proj_atk"
        _uc = f"{final_key}_u"
        _ts_col = "timestamp"

        _total_anom = int((df_merged[_pc] > threshold_sl.value).sum()) if _pc in df_merged.columns else 0
        _total_win  = len(df_merged)
        _pct_anom   = 100 * _total_anom / max(_total_win, 1)

        _cur_is_anom = bool(df_merged.iloc[current_df_idx][_pc] > threshold_sl.value) if _pc in df_merged.columns else False
        if _cur_is_anom and _pc in df_merged.columns:
            _i = current_df_idx
            while _i > 0 and df_merged.iloc[_i - 1][_pc] > threshold_sl.value:
                _i -= 1
            _ts_start = df_merged.iloc[_i][_ts_col]
            _ts_cur_v = df_merged.iloc[current_df_idx][_ts_col]
            try:
                _dur_min = (pd.Timestamp(_ts_cur_v) - pd.Timestamp(_ts_start)).total_seconds() / 60
            except Exception:
                _dur_min = 0.0
        else:
            _dur_min = 0.0

        if _pc in df_merged.columns:
            _flags = (df_merged[_pc] > threshold_sl.value).values
            _ep_lens = []
            _cnt = 0
            for _f in _flags:
                if _f: _cnt += 1
                elif _cnt > 0:
                    _ep_lens.append(_cnt)
                    _cnt = 0
            if _cnt > 0: _ep_lens.append(_cnt)
            _mttd_min   = float(np.median(_ep_lens)) * 5 if _ep_lens else 0.0
            _n_episodes = len(_ep_lens)
        else:
            _mttd_min = 0.0
            _n_episodes = 0

        if _uc in df_merged.columns and _total_anom > 0:
            _u_anom_mean = float(df_merged.loc[df_merged[_pc] > threshold_sl.value, _uc].mean())
            _u_str = f"{_u_anom_mean:.3f}"
        else:
            _u_str = "n/a"

        _status_color = "danger" if _cur_is_anom else "success"
        _status_txt   = "⚠ ATTAQUE EN COURS" if _cur_is_anom else "✓ TRAFIC NORMAL"

        _out = mo.vstack([
            mo.callout(mo.md(f"**{_status_txt}**  |  Épisode courant : **{_dur_min:.0f} min**"), kind=_status_color),
            mo.hstack([
                _kpi_card("Total anomalies",        str(_total_anom),     f"{_pct_anom:.1f}% des fenêtres"),
                _kpi_card("Épisodes détectés",       str(_n_episodes),     "fenêtres consécutives > seuil"),
                _kpi_card("Durée médiane épisode",   f"{_mttd_min:.0f} min", "estimation 5 min/fenêtre"),
                _kpi_card("Incertitude moy (anom)",  _u_str,               "u CBF sur fenêtres > seuil"),
            ], gap=3, justify="start"),
        ])
    _out


# ── BARRE DE STATUT FENÊTRE COURANTE ────────────────────────────────────────

@app.cell
def _(ATTACK_TYPES, TYPE_COLORS, current_df_idx, current_pos, df_merged, final_key, mo, n_frames, threshold_sl):
    if len(df_merged) > 0 and n_frames > 0:
        _r   = df_merged.iloc[current_df_idx]
        _ts  = _r.get("timestamp", "?")
        _pc  = f"{final_key}_proj_atk" if final_key else None
        _p   = float(_r[_pc]) if _pc and _pc in _r else 0.0
        _u   = float(_r.get(f"{final_key}_u", 0) or 0) if final_key else 0.0
        _anom = _p > threshold_sl.value

        _top1   = str(_r.get("top1_type", "")) if "top1_type" in _r.index else ""
        _top1_b = float(_r.get("top1_b", 0) or 0) if "top1_b" in _r.index else 0.0
        _u_sbn  = float(_r.get("u_sbn", 0) or 0) if "u_sbn" in _r.index else 0.0
        _nov    = float(_r.get("novelty_score", 0) or 0) if "novelty_score" in _r.index else 0.0
        _gate   = bool(_r.get("gate_open", False)) if "gate_open" in _r.index else False

        _det_str  = f"**ANOMALIE** proj={_p:.3f} u={_u:.3f}" if _anom else f"Normal proj={_p:.3f}"
        _qual_str = (f"| SBN: **{_top1}** b={_top1_b:.3f} u_sbn={_u_sbn:.3f} novelty={_nov:.2f}"
                     if _gate and _top1 and _top1 != "nan" else "| SBN: gate fermée")

        _out = mo.callout(
            mo.md(f"Fenêtre **{current_pos+1}/{n_frames}** | `{_ts}` | {_det_str} {_qual_str}"),
            kind="danger" if _anom else "neutral",
        )
    else:
        _out = mo.md("")
    _out


# ══════════════════════════════════════════════════════════════════════════
# BANDEAU HAUT — TIMELINES SUPERPOSÉES
# ══════════════════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md("---\n## Timelines : Détection + Qualification")


@app.cell
def _(ATTACK_TYPES, TYPE_COLORS, current_df_idx, df_merged, final_key, go, make_subplots, pd, threshold_sl):
    if len(df_merged) == 0:
        _f = go.Figure()
    else:
        _x   = df_merged["timestamp"] if "timestamp" in df_merged.columns else df_merged.index
        _fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.40, 0.35, 0.25],
            vertical_spacing=0.04,
            subplot_titles=["Signal de détection (proj_atk CBF)",
                            "Qualification SBN — belief masses par type",
                            "Incertitude : u_cbf vs u_sbn"],
        )

        _pc = f"{final_key}_proj_atk" if final_key else None
        _bc = f"{final_key}_b_atk"    if final_key else None
        if _pc and _pc in df_merged.columns:
            _fig.add_trace(go.Scatter(
                x=_x, y=df_merged[_pc], mode="lines", name="proj_atk",
                line=dict(color="#e74c3c", width=1.5),
                fill="tozeroy", fillcolor="rgba(231,76,60,0.15)"),
                row=1, col=1)
        if _bc and _bc in df_merged.columns:
            _fig.add_trace(go.Scatter(
                x=_x, y=df_merged[_bc], mode="lines", name="b_atk",
                line=dict(color="#e67e22", width=1, dash="dot")),
                row=1, col=1)
        _fig.add_hline(y=threshold_sl.value, line_color="yellow", line_dash="dash",
                       line_width=1.5, row=1, col=1)

        _gate_mask = df_merged.get("gate_open", pd.Series(False, index=df_merged.index)) == True
        _da  = df_merged[_gate_mask] if "gate_open" in df_merged.columns else df_merged
        _xa  = _da["timestamp"] if "timestamp" in _da.columns else _da.index
        for _i, _t in enumerate(ATTACK_TYPES):
            _c = f"b_sbn_{_t}"
            if _c in _da.columns:
                _fig.add_trace(go.Bar(
                    x=_xa, y=_da[_c].fillna(0), name=_t,
                    marker_color=TYPE_COLORS.get(_t, "#aaa"),
                    hovertemplate=f"<b>{_t}</b> %{{y:.3f}}<extra></extra>",
                ), row=2, col=1)

        _uc = f"{final_key}_u" if final_key else None
        if _uc and _uc in df_merged.columns:
            _fig.add_trace(go.Scatter(
                x=_x, y=df_merged[_uc], mode="lines", name="u_cbf",
                line=dict(color="#9b59b6", width=1.2)),
                row=3, col=1)
        if "u_sbn" in df_merged.columns:
            _fig.add_trace(go.Scatter(
                x=_x, y=df_merged["u_sbn"].fillna(0), mode="lines", name="u_sbn",
                line=dict(color="#3498db", width=1.2, dash="dot")),
                row=3, col=1)

        _cur = str(df_merged.iloc[current_df_idx].get("timestamp", current_df_idx))
        _fig.add_vline(x=_cur, line_color="white", line_width=2, line_dash="dash")

        _fig.update_layout(
            height=500, barmode="stack",
            plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", font=dict(color="white"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=50, r=20, t=60, b=20),
        )
        _fig.update_yaxes(range=[0, 1.05], showgrid=True, gridcolor="rgba(255,255,255,0.1)")
        _fig.update_xaxes(showgrid=False)
        _f = _fig
    _f


# ══════════════════════════════════════════════════════════════════════════
# PANNEAU CENTRAL — 3 COLONNES
# ══════════════════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md("---\n## Panneau central — Fenêtre courante")


@app.cell
def _(ATTACK_TYPES, TYPE_COLORS, current_df_idx, df_merged, final_key, go, leaf_metrics, make_subplots, threshold_sl):
    if len(df_merged) == 0:
        _f = go.Figure()
    else:
        _r = df_merged.iloc[current_df_idx]

        _fig = make_subplots(
            rows=1, cols=3,
            column_widths=[0.40, 0.28, 0.32],
            subplot_titles=[
                "Métriques individuelles (proj_atk)",
                "Certitude détection (CBF)",
                "Certitude qualification (SBN)",
            ],
            specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        )

        _pa    = [float(_r.get(f"{m}_proj_atk", 0) or 0) for m in leaf_metrics]
        _order = sorted(range(len(leaf_metrics)), key=lambda i: _pa[i], reverse=True)
        _lm_s  = [leaf_metrics[i] for i in _order]
        _pa_s  = [_pa[i] for i in _order]
        _cols  = []
        for _v in _pa_s:
            _rv = min(1.0, _v)
            _cols.append(f"rgba({int(231*_rv + 39*(1-_rv))},{int(76*_rv + 174*(1-_rv))},{int(60*_rv + 96*(1-_rv))},0.85)")
        _fig.add_trace(go.Bar(
            y=_lm_s, x=_pa_s, orientation="h", name="proj_atk",
            marker_color=_cols,
            hovertemplate="<b>%{y}</b><br>proj_atk=%{x:.3f}<extra></extra>",
        ), row=1, col=1)
        _fig.add_vline(x=threshold_sl.value, line_color="yellow", line_dash="dash",
                       line_width=1.5, row=1, col=1)

        if final_key:
            _fs = float(_r.get(f"{final_key}_b_safe", 0) or 0)
            _fq = float(_r.get(f"{final_key}_b_susp", 0) or 0)
            _fa = float(_r.get(f"{final_key}_b_atk",  0) or 0)
            _fu = float(_r.get(f"{final_key}_u",       0) or 0)
            _fp = float(_r.get(f"{final_key}_proj_atk",0) or 0)
            _fps= float(_r.get(f"{final_key}_proj_safe",0) or 0)
            _fpq= float(_r.get(f"{final_key}_proj_susp",0) or 0)
            for _v, _n, _c in [
                (_fs, "b_safe",  "#27ae60"), (_fq, "b_susp",  "#f39c12"),
                (_fa, "b_atk",   "#e74c3c"), (_fu, "u",        "#9b59b6"),
            ]:
                _fig.add_trace(go.Bar(x=["Beliefs"], y=[_v], name=_n,
                                      marker_color=_c, showlegend=False), row=1, col=2)
            for _v, _n, _c in [
                (_fps, "P(Safe)",  "#27ae60"), (_fpq, "P(Susp)", "#f39c12"),
                (_fp,  "P(Atk)",   "#e74c3c"),
            ]:
                _fig.add_trace(go.Bar(x=["Proj.Prob"], y=[_v], name=_n,
                                      marker_color=_c, showlegend=False), row=1, col=2)
            _is_a = _fp > threshold_sl.value
            _fig.add_annotation(
                x="Proj.Prob", y=1.2, xref="x2", yref="y2", showarrow=False,
                text=f"{'⚠ ANOMALIE' if _is_a else '✓ NORMAL'}<br>proj={_fp:.3f} | u={_fu:.3f}",
                font=dict(color="#e74c3c" if _is_a else "#2ecc71", size=10),
                bgcolor="rgba(0,0,0,0.6)",
            )

        _gate = bool(_r.get("gate_open", False)) if "gate_open" in _r.index else False
        if _gate:
            _sbn_vals = [float(_r.get(f"b_sbn_{t}", 0) or 0) for t in ATTACK_TYPES]
            _u_sbn    = float(_r.get("u_sbn", 0) or 0)
            _nov      = float(_r.get("novelty_score", 0) or 0)
            _top1     = str(_r.get("top1_type", "")) if "top1_type" in _r.index else ""
            _top1_b   = float(_r.get("top1_b", 0) or 0) if "top1_b" in _r.index else 0.0
            _ord2     = sorted(range(len(ATTACK_TYPES)), key=lambda i: _sbn_vals[i], reverse=True)
            _t_s      = [ATTACK_TYPES[i] for i in _ord2]
            _v_s      = [_sbn_vals[i] for i in _ord2]
            _c_s      = [TYPE_COLORS.get(t, "#aaa") for t in _t_s]
            _fig.add_trace(go.Bar(
                y=_t_s, x=_v_s, orientation="h", name="b_sbn",
                marker_color=_c_s,
                hovertemplate="<b>%{y}</b><br>b=%{x:.3f}<extra></extra>",
            ), row=1, col=3)
            _fig.add_trace(go.Bar(
                y=["u_sbn"], x=[_u_sbn], orientation="h",
                marker_color="#9b59b6", name="u_sbn", showlegend=False,
            ), row=1, col=3)
            _fig.add_annotation(
                x=0.5, y=len(ATTACK_TYPES) + 1.8, xref="x3", yref="y3", showarrow=False,
                text=f"Top-1: <b>{_top1}</b> ({_top1_b:.3f})<br>u={_u_sbn:.3f} | novelty={_nov:.2f}",
                font=dict(color="white", size=10), bgcolor="rgba(0,0,0,0.6)",
            )
        else:
            _fig.add_annotation(
                x=0.5, y=0.5, xref="x3 domain", yref="y3 domain", showarrow=False,
                text="Gate SBN fermée", font=dict(color="#7f8c8d", size=14),
            )

        _fig.update_layout(
            height=max(420, len(leaf_metrics) * 22 + 120),
            barmode="stack",
            plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", font=dict(color="white"),
            showlegend=False,
            margin=dict(l=170, r=20, t=60, b=20),
        )
        _fig.update_xaxes(range=[0, 1.05], showgrid=True,
                          gridcolor="rgba(255,255,255,0.1)", row=1, col=1)
        _fig.update_yaxes(showgrid=False)
        _fig.update_xaxes(range=[0, 1.3], showgrid=True,
                          gridcolor="rgba(255,255,255,0.1)", row=1, col=2)
        _fig.update_yaxes(range=[0, 1.4], row=1, col=2)
        _fig.update_xaxes(range=[0, 1.05], showgrid=True,
                          gridcolor="rgba(255,255,255,0.1)", row=1, col=3)
        _f = _fig
    _f


# ══════════════════════════════════════════════════════════════════════════
# PANNEAU BAS — DÉTAIL MÉTRIQUE (Opinion SL + Données brutes 30s)
# ══════════════════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md("---\n## Détail métrique — Opinion SL + Données brutes 30s")


@app.cell
def _(leaf_metrics, mo):
    metric_sel = mo.ui.dropdown(
        options={m: m for m in leaf_metrics},
        value=leaf_metrics[0] if leaf_metrics else None,
        label="Métrique",
    )
    return metric_sel,


@app.cell
def _(metric_sel, mo):
    mo.hstack([metric_sel])


@app.cell
def _(current_df_idx, df_merged, go, make_subplots, metric_sel, mo, pd, raw_data, raw_key_map):
    if len(df_merged) == 0 or not metric_sel.value:
        _f = go.Figure()
    else:
        _m  = metric_sel.value
        _rk = raw_key_map.get(_m)
        _r  = df_merged.iloc[current_df_idx]
        _ts_cur = _r.get("timestamp")
        _has_raw = bool(_rk and _rk in raw_data and len(raw_data[_rk]) > 0)

        if not _has_raw and _rk:
            mo.callout(
                mo.md(f"Raw data introuvable pour `{_rk}`. "
                      f"Clés disponibles : `{', '.join(list(raw_data.keys())[:5])}…`"),
                kind="warn"
            )

        _fig = make_subplots(
            rows=2 if _has_raw else 1, cols=1,
            shared_xaxes=True,
            row_heights=[0.42, 0.58] if _has_raw else [1.0],
            vertical_spacing=0.06,
            subplot_titles=(
                [f"Opinion SL — {_m}", f"Données brutes 30s — {_rk}"]
                if _has_raw else [f"Opinion SL — {_m}"]
            ),
        )

        _win = df_merged.iloc[max(0, current_df_idx - 59): current_df_idx + 1]
        _xo  = _win["timestamp"] if "timestamp" in _win.columns else _win.index

        for _suf, _name, _fc, _lc in [
            ("_b_safe", "Safe",    "rgba(39,174,96,0.55)",  "rgba(39,174,96,0)"),
            ("_b_susp", "Suspect", "rgba(243,156,18,0.55)", "rgba(243,156,18,0)"),
            ("_b_atk",  "Attack",  "rgba(231,76,60,0.70)",  "rgba(231,76,60,0)"),
        ]:
            _c = f"{_m}{_suf}"
            if _c in _win.columns:
                _fig.add_trace(go.Scatter(
                    x=_xo, y=_win[_c].fillna(0), name=_name, stackgroup="op",
                    fillcolor=_fc, line=dict(width=0.5, color=_lc),
                ), row=1, col=1)

        _cu = f"{_m}_u"
        if _cu in _win.columns:
            _fig.add_trace(go.Scatter(
                x=_xo, y=_win[_cu].fillna(0), name="u",
                line=dict(color="#3498db", width=1.5, dash="dot"),
            ), row=1, col=1)

        for _suf, _name, _color in [
            ("_proj_safe", "P(Safe)",    "#27ae60"),
            ("_proj_susp", "P(Suspect)", "#f39c12"),
            ("_proj_atk",  "P(Anomalie)","#e74c3c"),
        ]:
            _c = f"{_m}{_suf}"
            if _c in _win.columns:
                _fig.add_trace(go.Scatter(
                    x=_xo, y=_win[_c].fillna(0), name=_name,
                    line=dict(color=_color, width=1.2, dash="dash"),
                ), row=1, col=1)

        for _suf, _name, _color in [
            ("_a_safe", "a(Safe)",  "#27ae60"),
            ("_a_susp", "a(Susp)",  "#f39c12"),
            ("_a_atk",  "a(Atk)",   "#e74c3c"),
        ]:
            _c = f"{_m}{_suf}"
            if _c in _win.columns:
                _fig.add_trace(go.Scatter(
                    x=_xo, y=_win[_c].fillna(0), name=_name, opacity=0.5,
                    line=dict(color=_color, width=1, dash="longdash"),
                ), row=1, col=1)

        if _ts_cur is not None:
            _fig.add_vline(x=str(_ts_cur), line_color="white", line_width=2, line_dash="dash")

        if _has_raw:
            _raw_df = raw_data[_rk]
            try:
                _ts       = pd.Timestamp(_ts_cur)
                _ts_left  = pd.Timestamp(_xo.iloc[0]) if hasattr(_xo, "iloc") else _ts - pd.Timedelta("300min")
                _span     = _ts - _ts_left
                _ts_right = _ts + _span
                _raw = _raw_df[
                    (_raw_df["timestamp"] >= _ts_left) &
                    (_raw_df["timestamp"] <= _ts_right)
                ]
            except Exception:
                _raw = _raw_df.tail(3000)
            _xr = _raw["timestamp"]
            if "abs_error" in _raw.columns:
                _fig.add_trace(go.Scatter(
                    x=_xr, y=_raw["abs_error"].clip(lower=0),
                    name="Abs Error", fill="tozeroy",
                    fillcolor="rgba(231,76,60,0.12)",
                    line=dict(color="rgba(231,76,60,0.25)", width=0.5),
                ), row=2, col=1)
            if "real" in _raw.columns:
                _fig.add_trace(go.Scatter(
                    x=_xr, y=_raw["real"], name="Réel",
                    line=dict(color="rgba(200,200,200,0.85)", width=1),
                ), row=2, col=1)
            if "pred" in _raw.columns:
                _fig.add_trace(go.Scatter(
                    x=_xr, y=_raw["pred"], name="Prédiction (modèle)",
                    line=dict(color="#27ae60", width=1.3, dash="dash"),
                ), row=2, col=1)
            if _ts_cur is not None:
                _fig.add_vline(x=str(_ts_cur), line_color="white", line_width=2, line_dash="dash")

        _fig.update_layout(
            height=700 if _has_raw else 380,
            plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", font=dict(color="white"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
            margin=dict(l=60, r=20, t=80, b=60),
        )
        _fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
        _fig.update_xaxes(showgrid=False, tickangle=-30)
        _fig.update_yaxes(range=[0, 1.05], row=1, col=1)
        _f = _fig
    _f


# ══════════════════════════════════════════════════════════════════════════
# TABLE DES ALERTES — Épisodes détectés avec filtre par type
# Référence : Jiang et al. 2012 "Towards Usable Network Monitoring Interfaces"
# → 73% du temps opérateur passé sur vue tabulaire filtrée, pas sur graphes
# ══════════════════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md("---\n## Table des alertes — Épisodes détectés")


@app.cell
def _(ATTACK_TYPES, mo):
    type_filter = mo.ui.multiselect(
        options=["Tous"] + ATTACK_TYPES,
        value=["Tous"],
        label="Filtrer par type d'attaque",
    )
    return type_filter,


@app.cell
def _(mo, type_filter):
    mo.hstack([type_filter])


@app.cell
def _(df_merged, final_key, mo, pd, threshold_sl, type_filter):
    if len(df_merged) == 0 or not final_key:
        _out = mo.callout(mo.md("Aucune donnée chargée."), kind="warn")
    else:
        _pc = f"{final_key}_proj_atk"
        _uc = f"{final_key}_u"

        _episodes = []
        _in_ep    = False
        _ep_rows  = []

        for _i, _row in df_merged.iterrows():
            _p = float(_row.get(_pc, 0) or 0) if _pc in _row.index else 0.0
            if _p > threshold_sl.value:
                _in_ep = True
                _ep_rows.append(_row)
            else:
                if _in_ep and _ep_rows:
                    _ep_df = pd.DataFrame(_ep_rows)
                    try:
                        _td = (pd.Timestamp(_ep_df.iloc[-1]["timestamp"]) -
                               pd.Timestamp(_ep_df.iloc[0]["timestamp"])).total_seconds() / 60
                    except Exception:
                        _td = len(_ep_df) * 5
                    _ep_dict = {
                        "Début"        : str(_ep_df.iloc[0]["timestamp"])[:19],
                        "Fin"          : str(_ep_df.iloc[-1]["timestamp"])[:19],
                        "Durée (min)"  : round(_td, 1),
                        "max proj_atk" : round(float(_ep_df[_pc].max()), 4) if _pc in _ep_df.columns else "-",
                        "u_cbf moy"    : round(float(_ep_df[_uc].mean()), 4) if _uc in _ep_df.columns else "-",
                        "Type top-1"   : _ep_df["top1_type"].dropna().mode().iloc[0] if "top1_type" in _ep_df.columns and len(_ep_df["top1_type"].dropna()) > 0 else "?",
                        "b_top1 moy"   : round(float(_ep_df["top1_b"].mean()), 4) if "top1_b" in _ep_df.columns else "-",
                        "u_sbn moy"    : round(float(_ep_df["u_sbn"].mean()), 4) if "u_sbn" in _ep_df.columns else "-",
                        "novelty moy"  : round(float(_ep_df["novelty_score"].mean()), 4) if "novelty_score" in _ep_df.columns else "-",
                    }
                    _episodes.append(_ep_dict)
                    _ep_rows = []
                _in_ep = False

        if _in_ep and _ep_rows:
            _ep_df = pd.DataFrame(_ep_rows)
            try:
                _td = (pd.Timestamp(_ep_df.iloc[-1]["timestamp"]) -
                       pd.Timestamp(_ep_df.iloc[0]["timestamp"])).total_seconds() / 60
            except Exception:
                _td = len(_ep_df) * 5
            _episodes.append({
                "Début"        : str(_ep_df.iloc[0]["timestamp"])[:19],
                "Fin"          : str(_ep_df.iloc[-1]["timestamp"])[:19] + " ⚠ en cours",
                "Durée (min)"  : round(_td, 1),
                "max proj_atk" : round(float(_ep_df[_pc].max()), 4) if _pc in _ep_df.columns else "-",
                "u_cbf moy"    : round(float(_ep_df[_uc].mean()), 4) if _uc in _ep_df.columns else "-",
                "Type top-1"   : _ep_df["top1_type"].dropna().mode().iloc[0] if "top1_type" in _ep_df.columns and len(_ep_df["top1_type"].dropna()) > 0 else "?",
                "b_top1 moy"   : round(float(_ep_df["top1_b"].mean()), 4) if "top1_b" in _ep_df.columns else "-",
                "u_sbn moy"    : round(float(_ep_df["u_sbn"].mean()), 4) if "u_sbn" in _ep_df.columns else "-",
                "novelty moy"  : round(float(_ep_df["novelty_score"].mean()), 4) if "novelty_score" in _ep_df.columns else "-",
            })

        if not _episodes:
            _out = mo.callout(mo.md(f"Aucun épisode au-dessus du seuil ({threshold_sl.value:.2f})."), kind="neutral")
        else:
            _df_ep = pd.DataFrame(_episodes)
            if type_filter.value and "Tous" not in type_filter.value:
                _df_ep = _df_ep[_df_ep["Type top-1"].isin(type_filter.value)]
            _out = mo.vstack([
                mo.md(f"**{len(_df_ep)} épisode(s)** | seuil proj_atk > {threshold_sl.value:.2f}"),
                mo.ui.table(_df_ep, page_size=15),
            ])
    _out


# ══════════════════════════════════════════════════════════════════════════
# CORRÉLATION INTER-MÉTRIQUES — Détection d'attaques coordonnées
# Pearson sur proj_atk des 60 dernières fenêtres.
# Référence opérationnelle : colonnes simultanément rouges = attaque volumétrique
# (Chandola et al. 2009 ACM CSUR §3.2 ; Sommer & Paxson 2010 IEEE S&P)
# ══════════════════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md("---\n## Corrélation inter-métriques (proj_atk, 60 fenêtres glissantes)")


@app.cell
def _(current_df_idx, df_merged, go, leaf_metrics, np):
    if len(df_merged) == 0 or len(leaf_metrics) < 2:
        _f = go.Figure()
    else:
        _win  = df_merged.iloc[max(0, current_df_idx - 59): current_df_idx + 1]
        _cols = [f"{m}_proj_atk" for m in leaf_metrics if f"{m}_proj_atk" in _win.columns]
        _lm_s = [c.replace("_proj_atk", "") for c in _cols]

        if len(_cols) >= 2:
            _mat = _win[_cols].fillna(0).corr(method="pearson").values
            # Masque triangulaire sup pour lisibilité
            _mask = np.triu(np.ones_like(_mat, dtype=bool), k=1)
            _mat_disp = _mat.copy()
            _mat_disp[_mask] = float("nan")

            _fig = go.Figure(go.Heatmap(
                z=_mat_disp,
                x=_lm_s, y=_lm_s,
                colorscale=[
                    [0.0, "#2980b9"],   # corrélation négative — bleu
                    [0.5, "#1a1a2e"],   # nulle — fond
                    [1.0, "#e74c3c"],   # positive forte — rouge
                ],
                zmin=-1, zmax=1,
                colorbar=dict(title="r Pearson", thickness=12),
                hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
                text=[[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in _mat_disp],
                texttemplate="%{text}",
                textfont=dict(size=8),
            ))
            _fig.update_layout(
                height=max(400, len(_lm_s) * 28 + 100),
                plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", font=dict(color="white"),
                margin=dict(l=180, r=80, t=40, b=180),
                xaxis=dict(tickangle=-45),
            )
        else:
            _fig = go.Figure()
        _f = _fig
    _f


# ══════════════════════════════════════════════════════════════════════════
# BASELINE ROLLING — Comparaison semaine N vs N-1
# Standard IPFIX/sFlow (ntopng, ElastiFlow, RFC 5101).
# Détection d'anomalies calendaires : lundi courant vs 4 derniers lundis.
# ══════════════════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md("---\n## Baseline rolling — Semaine courante vs semaine précédente")


@app.cell
def _(leaf_metrics, mo):
    baseline_metric_sel = mo.ui.dropdown(
        options={m: m for m in leaf_metrics},
        value=leaf_metrics[0] if leaf_metrics else None,
        label="Métrique pour comparaison baseline",
    )
    baseline_weeks_sel = mo.ui.slider(1, 4, value=2, step=1,
                                      label="Nb semaines de référence",
                                      show_value=True)
    return baseline_metric_sel, baseline_weeks_sel


@app.cell
def _(baseline_metric_sel, baseline_weeks_sel, mo):
    mo.hstack([baseline_metric_sel, baseline_weeks_sel])


@app.cell
def _(baseline_metric_sel, baseline_weeks_sel, df_merged, go, pd, raw_data, raw_key_map):
    if len(df_merged) == 0 or not baseline_metric_sel.value:
        _f = go.Figure()
    else:
        _m   = baseline_metric_sel.value
        _rk  = raw_key_map.get(_m)
        _use_raw = bool(_rk and _rk in raw_data and len(raw_data[_rk]) > 0)

        # Source de données : raw 30s si dispo, sinon proj_atk fenêtré
        if _use_raw:
            _src = raw_data[_rk][["timestamp", "real"]].dropna()
            _src = _src.rename(columns={"real": "value"})
            _col_label = f"{_rk} — valeur réelle (30s)"
        else:
            _pc   = f"{_m}_proj_atk"
            if _pc in df_merged.columns:
                _src = df_merged[["timestamp", _pc]].dropna().rename(columns={_pc: "value"})
            else:
                _src = pd.DataFrame(columns=["timestamp", "value"])
            _col_label = f"{_m} — proj_atk (fenêtres)"

        _fig = go.Figure()

        if len(_src) > 0:
            _ts_max = _src["timestamp"].max()
            _ts_min = _src["timestamp"].min()

            # Semaine courante : 7 derniers jours
            _w_start = _ts_max - pd.Timedelta(days=7)
            _cur = _src[_src["timestamp"] >= _w_start]

            _fig.add_trace(go.Scatter(
                x=_cur["timestamp"], y=_cur["value"],
                mode="lines", name="Semaine courante",
                line=dict(color="#e74c3c", width=2),
            ))

            # Semaines de référence N-1, N-2, …
            _palette_ref = ["rgba(52,152,219,0.7)", "rgba(52,152,219,0.45)", "rgba(52,152,219,0.25)", "rgba(52,152,219,0.15)"]
            for _wk in range(1, min(baseline_weeks_sel.value + 1, 5)):
                _offset     = pd.Timedelta(weeks=_wk)
                _ref_start  = _w_start - _offset
                _ref_end    = _ts_max  - _offset
                _ref        = _src[(_src["timestamp"] >= _ref_start) & (_src["timestamp"] <= _ref_end)].copy()
                if len(_ref) == 0:
                    continue
                # Décaler les timestamps pour superposition visuelle
                _ref["timestamp_shifted"] = _ref["timestamp"] + _offset
                _fig.add_trace(go.Scatter(
                    x=_ref["timestamp_shifted"], y=_ref["value"],
                    mode="lines", name=f"N-{_wk}",
                    line=dict(color=_palette_ref[_wk - 1], width=1, dash="dot"),
                ))

        _fig.update_layout(
            height=350,
            title=dict(text=_col_label, font=dict(size=12)),
            plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", font=dict(color="white"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=60, r=20, t=60, b=60),
            xaxis=dict(showgrid=False, tickangle=-30),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        )
        _f = _fig
    _f


if __name__ == "__main__":
    app.run()