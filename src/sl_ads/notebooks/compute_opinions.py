import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full", app_title="Compute Opinions Replay — Pipeline SL 3 niveaux")


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
    mo.md("# Compute Opinions Replay — Pipeline SL 3 niveaux")


# ── IMPORT CONFIG (source unique de vérité pour les chemins) ─────────────────

@app.cell
def _(pathlib, sys):
    # PATCH TASK-33 / MIN-01 (audit_tmp, 2026-04-26)
    # ──────────────────────────────────────────────────────────────────────
    # Le chemin pointait vers ``actual_version`` (ancien dossier Crocodile)
    # au lieu de ``actual_ version_claude_autre dataset`` (le dossier
    # courant utilisé par toutes les autres entrées du pipeline). On essaie
    # d'abord le dossier courant, puis on retombe sur ``actual_version``
    # pour rétrocompat.
    _PROJ_ROOT = pathlib.Path(r"C:\Users\cyril\PycharmProjects\IDS_SL_Bresil_v1")
    _CURRENT_DIR = _PROJ_ROOT / "actual_ version_claude_autre dataset"
    _LEGACY_DIR  = _PROJ_ROOT / "actual_version"
    if _CURRENT_DIR.exists():
        sys.path.insert(0, str(_CURRENT_DIR))
    elif _LEGACY_DIR.exists():
        sys.path.insert(0, str(_LEGACY_DIR))
    try:
        # Phase H: prefer sl_ads.config; fall back to legacy ``config`` shim.
        try:
            from sl_ads.config import CONFIG as _CFG
        except Exception:
            from config import CONFIG as _CFG
        _version_name = _CFG["VERSION_NAME"]
        RESULTS_DIR_DEFAULT = str(_PROJ_ROOT / "results" / f"resultats_{_version_name}")
        config_import_ok = True
    except Exception:
        RESULTS_DIR_DEFAULT = str(_PROJ_ROOT / "results")
        config_import_ok = False
    return RESULTS_DIR_DEFAULT, config_import_ok


# ── CHARGEMENT FICHIERS ─────────────────────────────────────────────────────

@app.cell
def _(RESULTS_DIR_DEFAULT, mo, pathlib):
    _RES = pathlib.Path(RESULTS_DIR_DEFAULT)
    det_input = mo.ui.text(
        value=str(_RES / "detection_results_INJECTED.csv"),
        label="detection_results CSV", full_width=True,
    )
    return det_input,


@app.cell
def _(det_input, mo):
    mo.vstack([det_input])


@app.cell
def _(det_input, pd):
    try:
        _df = pd.read_csv(det_input.value)
        _df["timestamp"] = pd.to_datetime(_df["timestamp"], errors="coerce")
        df = _df
        load_ok = True
    except Exception as _e:
        df = pd.DataFrame()
        load_ok = False
    return df, load_ok


@app.cell
def _(df, load_ok, mo):
    if load_ok:
        _out = mo.callout(mo.md(f"detection_results : **{len(df)} fenêtres** | {len(df.columns)} colonnes"), kind="success")
    else:
        _out = mo.callout(mo.md("Erreur chargement detection_results"), kind="danger")
    _out


# Auto-détection raw_data CSV
@app.cell
def _(det_input, mo, pathlib):
    _dir = pathlib.Path(det_input.value).parent
    _cands = sorted(_dir.glob("raw_data_*.csv"))
    _default = str(_cands[0]) if _cands else ""
    raw_input = mo.ui.text(value=_default, label="raw_data CSV", full_width=True)
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
def _(mo, raw_data, raw_ok):
    if raw_ok:
        _n = sum(len(v) for v in raw_data.values())
        _out = mo.callout(mo.md(f"raw_data : **{len(raw_data)} métriques** | {_n:,} points 30s"), kind="success")
    else:
        _out = mo.callout(mo.md("raw_data non chargé — panneau bas désactivé"), kind="warn")
    _out


# ── STRUCTURE MÉTRIQUES ─────────────────────────────────────────────────────

@app.cell
def _(df):
    _all = [c[:-len("_b_safe")] for c in df.columns if c.endswith("_b_safe")]
    leaf_metrics = [p for p in _all if not p.startswith("METHODE") and not p.startswith("FINAL")]
    method_keys  = [p for p in _all if p.startswith("METHODE")]
    final_key    = next((p for p in _all if p.startswith("FINAL")), None)

    # FIX ROOT BUG : les colonnes sont au format "prophet_xxx"/"reconst_xxx" dans train_v10
    def det_to_raw(m):
        if m.startswith("prophet_") or m.startswith("reconst_"):
            return m
        if m.startswith("P_"):   return "prophet_" + m[2:]
        if m.startswith("R_"):   return "reconst_"  + m[2:]
        return None

    raw_key_map = {m: det_to_raw(m) for m in leaf_metrics}
    return det_to_raw, final_key, leaf_metrics, method_keys, raw_key_map


# ── CONTRÔLES ───────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("---\n## Contrôles")


@app.cell
def _(mo):
    is_playing   = mo.ui.switch(label="▶ Play", value=False)
    anomaly_only = mo.ui.switch(label="Anomalies uniquement", value=False)
    return anomaly_only, is_playing


@app.cell
def _(mo):
    threshold_sl = mo.ui.slider(0.05, 0.95, value=0.15, step=0.01,
                                label="Seuil proj_atk", show_value=True)
    speed_sel = mo.ui.dropdown(
        options={"0.5 fps": "2000ms", "1 fps": "1000ms", "2 fps": "500ms",
                 "5 fps": "200ms", "10 fps": "150ms"},
        value="1 fps", label="Vitesse",
    )
    return speed_sel, threshold_sl


@app.cell
def _(anomaly_only, is_playing, mo, speed_sel, threshold_sl):
    mo.hstack([is_playing, anomaly_only, threshold_sl, speed_sel], gap=4)


# ── ANIMATION ───────────────────────────────────────────────────────────────

@app.cell
def _(anomaly_only, df, final_key, threshold_sl):
    _col = f"{final_key}_proj_atk" if final_key else None
    if anomaly_only.value and _col and _col in df.columns:
        play_indices = list(df.index[df[_col] > threshold_sl.value])
    else:
        play_indices = list(range(len(df)))
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


# FIX : incrément uniquement quand is_playing est actif
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


@app.cell
def _(current_df_idx, current_pos, df, final_key, mo, n_frames, threshold_sl):
    if len(df) > 0 and n_frames > 0:
        _row = df.iloc[current_df_idx]
        _ts  = _row.get("timestamp", "?")
        _pc  = f"{final_key}_proj_atk" if final_key else None
        _uc  = f"{final_key}_u" if final_key else None
        _p   = float(_row[_pc]) if _pc and _pc in _row else 0.0
        _u   = float(_row[_uc]) if _uc and _uc in _row else 0.0
        _anom = _p > threshold_sl.value
        mo.callout(
            mo.md(f"Fenêtre **{current_pos+1}/{n_frames}** | `{_ts}` | "
                  f"{'**ANOMALIE**' if _anom else 'Normal'} proj_atk={_p:.3f} | u={_u:.3f}"),
            kind="danger" if _anom else "neutral",
        )


# ══════════════════════════════════════════════════════════════════════════
# VUE 1 — SIGNAL FINAL
# ══════════════════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md("---\n## Vue 1 — Signal final : FINAL_SYSTEM_CBF")


@app.cell
def _(current_df_idx, df, final_key, go, make_subplots, threshold_sl):
    if len(df) == 0 or not final_key:
        _f = go.Figure()
    else:
        _pc  = f"{final_key}_proj_atk"
        _bc  = f"{final_key}_b_atk"
        _uc  = f"{final_key}_u"
        _x   = df["timestamp"] if "timestamp" in df.columns else df.index

        _fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.68, 0.32], vertical_spacing=0.04,
                             subplot_titles=["proj_atk (décision) + seuil", "Incertitude u"])

        if _pc in df.columns:
            _fig.add_trace(go.Scatter(x=_x, y=df[_pc], mode="lines", name="proj_atk",
                                      line=dict(color="#e74c3c", width=1.5),
                                      fill="tozeroy", fillcolor="rgba(231,76,60,0.15)"),
                           row=1, col=1)
        if _bc in df.columns:
            _fig.add_trace(go.Scatter(x=_x, y=df[_bc], mode="lines", name="b_atk",
                                      line=dict(color="#e67e22", width=1, dash="dot")),
                           row=1, col=1)
        _fig.add_hline(y=threshold_sl.value, line_color="yellow", line_dash="dash",
                       line_width=1.5, annotation_text=f"seuil={threshold_sl.value:.2f}",
                       row=1, col=1)

        if _uc in df.columns:
            _fig.add_trace(go.Scatter(x=_x, y=df[_uc], mode="lines", name="u",
                                      line=dict(color="#9b59b6", width=1),
                                      fill="tozeroy", fillcolor="rgba(155,89,182,0.2)"),
                           row=2, col=1)

        _cur = str(df.iloc[current_df_idx].get("timestamp", current_df_idx))
        _fig.add_vline(x=_cur, line_color="white", line_width=2, line_dash="dash")

        _fig.update_layout(height=380, plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
                           font=dict(color="white"),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02),
                           margin=dict(l=50, r=20, t=50, b=20))
        _fig.update_yaxes(range=[0, 1.05], showgrid=True, gridcolor="rgba(255,255,255,0.1)")
        _fig.update_xaxes(showgrid=False)
        _f = _fig
    _f


# ══════════════════════════════════════════════════════════════════════════
# VUE 2 — CASCADE PIPELINE
# ══════════════════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md("---\n## Vue 2 — Cascade pipeline (fenêtre courante)")


@app.cell
def _(current_df_idx, df, final_key, go, leaf_metrics, make_subplots, method_keys, threshold_sl):
    if len(df) == 0:
        _f = go.Figure()
    else:
        _r = df.iloc[current_df_idx]

        _pa = [float(_r.get(f"{m}_proj_atk", 0) or 0) for m in leaf_metrics]
        _order = sorted(range(len(leaf_metrics)), key=lambda i: _pa[i], reverse=True)
        _lm_s = [leaf_metrics[i] for i in _order]
        _pa_s = [_pa[i] for i in _order]

        _fig = make_subplots(
            rows=1, cols=3, column_widths=[0.55, 0.25, 0.20],
            subplot_titles=["Niveau 1 — Métriques (proj_atk, triées)",
                            "Niveau 2 — WBF intra-méthode",
                            "Niveau 3 — CBF final"],
        )

        _colors = [f"rgba({int(231*(v**0.5))},{int(76*(1-v))},{int(60*(1-v))},0.85)"
                   for v in _pa_s]
        _fig.add_trace(go.Bar(
            y=_lm_s, x=_pa_s, orientation="h", name="proj_atk",
            marker_color=_colors,
            hovertemplate="<b>%{y}</b><br>proj_atk=%{x:.3f}<extra></extra>",
        ), row=1, col=1)
        _fig.add_vline(x=threshold_sl.value, line_color="yellow", line_dash="dash",
                       line_width=1.5, row=1, col=1)

        _ml = [mk.replace("METHODE_1_", "").replace("METHODE_2_", "") for mk in method_keys]
        for _vals, _name, _col in [
            ([float(_r.get(f"{mk}_b_safe", 0) or 0) for mk in method_keys], "b_safe", "#27ae60"),
            ([float(_r.get(f"{mk}_b_susp", 0) or 0) for mk in method_keys], "b_susp", "#f39c12"),
            ([float(_r.get(f"{mk}_b_atk",  0) or 0) for mk in method_keys], "b_atk",  "#e74c3c"),
            ([float(_r.get(f"{mk}_u",       0) or 0) for mk in method_keys], "u",      "#9b59b6"),
        ]:
            _fig.add_trace(go.Bar(x=_ml, y=_vals, name=_name,
                                  marker_color=_col,
                                  hovertemplate=f"{_name}=%{{y:.3f}}<extra></extra>"),
                           row=1, col=2)

        if final_key:
            _fs = float(_r.get(f"{final_key}_b_safe", 0) or 0)
            _fq = float(_r.get(f"{final_key}_b_susp", 0) or 0)
            _fa = float(_r.get(f"{final_key}_b_atk",  0) or 0)
            _fu = float(_r.get(f"{final_key}_u",       0) or 0)
            _fp = float(_r.get(f"{final_key}_proj_atk",0) or 0)
            for _v, _n, _c in [(_fs,"b_safe","#27ae60"),(_fq,"b_susp","#f39c12"),
                                (_fa,"b_atk","#e74c3c"),(_fu,"u","#9b59b6")]:
                _fig.add_trace(go.Bar(x=["CBF"], y=[_v], name=_n, marker_color=_c,
                                      showlegend=False), row=1, col=3)
            _is_a = _fp > threshold_sl.value
            _fig.add_annotation(
                x="CBF", y=1.2, xref="x3", yref="y3", showarrow=False,
                text=f"{'⚠ ANOMALIE' if _is_a else '✓ NORMAL'}<br>proj={_fp:.3f}",
                font=dict(color="#e74c3c" if _is_a else "#2ecc71", size=11),
                bgcolor="rgba(0,0,0,0.6)",
            )

        _fig.update_layout(
            height=max(380, len(leaf_metrics) * 24 + 120),
            barmode="stack",
            plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", font=dict(color="white"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=180, r=20, t=80, b=20),
        )
        _fig.update_xaxes(range=[0, 1.05], showgrid=True,
                          gridcolor="rgba(255,255,255,0.1)", row=1, col=1)
        _fig.update_yaxes(range=[0, 1.35], showgrid=True,
                          gridcolor="rgba(255,255,255,0.1)", row=1, col=2)
        _fig.update_yaxes(range=[0, 1.35], showgrid=True,
                          gridcolor="rgba(255,255,255,0.1)", row=1, col=3)
        _f = _fig
    _f


# ══════════════════════════════════════════════════════════════════════════
# VUE 3 — HEATMAP proj_atk
# ══════════════════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md("---\n## Vue 3 — Heatmap proj_atk (17 métriques × 30 fenêtres glissantes)")


@app.cell
def _(current_df_idx, df, go, leaf_metrics):
    if len(df) == 0:
        _f = go.Figure()
    else:
        _win = df.iloc[max(0, current_df_idx - 29): current_df_idx + 1]
        _x = _win["timestamp"].astype(str) if "timestamp" in _win.columns else _win.index.astype(str)
        _z = []
        for _m in leaf_metrics:
            _c = f"{_m}_proj_atk"
            _z.append(_win[_c].fillna(0).tolist() if _c in _win.columns else [0]*len(_win))

        _fig = go.Figure(go.Heatmap(
            z=_z, x=list(_x), y=leaf_metrics,
            colorscale=[[0,"#0d3b66"],[0.25,"#27ae60"],[0.55,"#f39c12"],[1,"#e74c3c"]],
            zmin=0, zmax=1,
            colorbar=dict(title="proj_atk", thickness=12),
            hovertemplate="<b>%{y}</b><br>%{x}<br>proj_atk=%{z:.3f}<extra></extra>",
        ))
        _cur = str(df.iloc[current_df_idx].get("timestamp", current_df_idx))
        _fig.add_vline(x=_cur, line_color="white", line_width=2, line_dash="dash")
        _fig.update_layout(
            height=max(350, len(leaf_metrics) * 24 + 120),
            plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", font=dict(color="white"),
            margin=dict(l=210, r=80, t=30, b=80),
            xaxis=dict(tickangle=-40),
        )
        _f = _fig
    _f


# ══════════════════════════════════════════════════════════════════════════
# VUE 4 — HEATMAP conflict_K
# ══════════════════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md("---\n## Vue 4 — Heatmap conflict_K (ruptures détectées par le filtre temporel)")


@app.cell
def _(current_df_idx, df, go, leaf_metrics):
    if len(df) == 0:
        _f = go.Figure()
    else:
        _win = df.iloc[max(0, current_df_idx - 29): current_df_idx + 1]
        _x = _win["timestamp"].astype(str) if "timestamp" in _win.columns else _win.index.astype(str)
        _z = []
        for _m in leaf_metrics:
            _c = f"{_m}_conflict_K"
            _z.append(_win[_c].fillna(0).tolist() if _c in _win.columns else [0]*len(_win))

        _fig = go.Figure(go.Heatmap(
            z=_z, x=list(_x), y=leaf_metrics,
            colorscale=[[0,"#0d1b2a"],[0.35,"#2980b9"],[0.7,"#f39c12"],[1,"#e74c3c"]],
            zmin=0, zmax=1,
            colorbar=dict(title="conflict_K", thickness=12),
            hovertemplate="<b>%{y}</b><br>%{x}<br>K=%{z:.3f}<extra></extra>",
        ))
        _cur = str(df.iloc[current_df_idx].get("timestamp", current_df_idx))
        _fig.add_vline(x=_cur, line_color="white", line_width=2, line_dash="dash")
        _fig.update_layout(
            height=max(350, len(leaf_metrics) * 24 + 120),
            plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", font=dict(color="white"),
            margin=dict(l=210, r=80, t=30, b=80),
            xaxis=dict(tickangle=-40),
        )
        _f = _fig
    _f


# ══════════════════════════════════════════════════════════════════════════
# VUE 5+6 — DÉTAIL MÉTRIQUE : Opinion SL + Raw data
# ══════════════════════════════════════════════════════════════════════════

@app.cell
def _(mo):
    mo.md("---\n## Vue 5+6 — Détail métrique : Opinion SL + Données brutes 30s")


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
def _(current_df_idx, df, go, make_subplots, metric_sel, mo, pd, raw_data, raw_key_map):
    if len(df) == 0 or not metric_sel.value:
        _f = go.Figure()
    else:
        _m   = metric_sel.value
        _rk  = raw_key_map.get(_m)
        _row = df.iloc[current_df_idx]
        _ts_cur = _row.get("timestamp")

        _has_raw = bool(_rk and _rk in raw_data and len(raw_data[_rk]) > 0)

        # Diagnostic si raw attendu mais absent
        if not _has_raw and _rk and len(raw_data) > 0:
            _avail = list(raw_data.keys())[:8]
            mo.callout(
                mo.md(f"⚠ Clé raw `{_rk}` introuvable. Clés dispo : `{', '.join(_avail)}{'…' if len(raw_data)>8 else ''}`"),
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

        _win = df.iloc[max(0, current_df_idx - 59): current_df_idx + 1]
        _xo  = _win["timestamp"] if "timestamp" in _win.columns else _win.index

        for _suf, _name, _fc, _lc in [
            ("_b_safe", "Safe",    "rgba(39,174,96,0.55)",  "rgba(39,174,96,0)"),
            ("_b_susp", "Suspect", "rgba(243,156,18,0.55)", "rgba(243,156,18,0)"),
            ("_b_atk",  "Attack",  "rgba(231,76,60,0.70)",  "rgba(231,76,60,0)"),
        ]:
            _c = f"{_m}{_suf}"
            if _c in _win.columns:
                _fig.add_trace(go.Scatter(
                    x=_xo, y=_win[_c].fillna(0),
                    name=_name, stackgroup="op",
                    fillcolor=_fc, line=dict(width=0.5, color=_lc),
                    hovertemplate=f"{_name}=%{{y:.3f}}<extra></extra>",
                ), row=1, col=1)

        _cu = f"{_m}_u"
        if _cu in _win.columns:
            _fig.add_trace(go.Scatter(
                x=_xo, y=_win[_cu].fillna(0), name="u (incertitude)",
                line=dict(color="#3498db", width=1.5, dash="dot"),
                hovertemplate="u=%{y:.3f}<extra></extra>",
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
                    hovertemplate=f"{_name}=%{{y:.3f}}<extra></extra>",
                ), row=1, col=1)

        for _suf, _name, _color in [
            ("_a_safe", "a(Safe) adapt.",    "#27ae60"),
            ("_a_susp", "a(Susp) adapt.",    "#f39c12"),
            ("_a_atk",  "a(Atk) adapt.",     "#e74c3c"),
        ]:
            _c = f"{_m}{_suf}"
            if _c in _win.columns:
                _fig.add_trace(go.Scatter(
                    x=_xo, y=_win[_c].fillna(0), name=_name, opacity=0.55,
                    line=dict(color=_color, width=1, dash="longdash"),
                ), row=1, col=1)

        if _ts_cur is not None:
            _fig.add_vline(x=str(_ts_cur), line_color="white", line_width=2, line_dash="dash")

        if _has_raw:
            _raw_df = raw_data[_rk]
            try:
                _ts      = pd.Timestamp(_ts_cur)
                _ts_left = pd.Timestamp(_xo.iloc[0]) if hasattr(_xo, "iloc") else _ts - pd.Timedelta("300min")
                _span    = _ts - _ts_left
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
                    name="Abs Error", mode="lines",
                    fill="tozeroy", fillcolor="rgba(231,76,60,0.12)",
                    line=dict(color="rgba(231,76,60,0.25)", width=0.5),
                ), row=2, col=1)

            if "real" in _raw.columns:
                _fig.add_trace(go.Scatter(
                    x=_xr, y=_raw["real"],
                    name="Réel", mode="lines",
                    line=dict(color="rgba(200,200,200,0.85)", width=1),
                ), row=2, col=1)

            if "pred" in _raw.columns:
                _fig.add_trace(go.Scatter(
                    x=_xr, y=_raw["pred"],
                    name="Prédiction (modèle)", mode="lines",
                    line=dict(color="#27ae60", width=1.3, dash="dash"),
                ), row=2, col=1)

            if _ts_cur is not None:
                _fig.add_vline(x=str(_ts_cur), line_color="white", line_width=2, line_dash="dash")

        _fig.update_layout(
            height=700 if _has_raw else 400,
            plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", font=dict(color="white"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
            margin=dict(l=60, r=20, t=80, b=60),
        )
        _fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
        _fig.update_xaxes(showgrid=False, tickangle=-30)
        _fig.update_yaxes(range=[0, 1.05], row=1, col=1)
        _f = _fig
    _f


if __name__ == "__main__":
    app.run()