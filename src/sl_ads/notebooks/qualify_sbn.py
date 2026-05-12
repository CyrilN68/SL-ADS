import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full", app_title="SBN Replay — Qualification d'anomalies")


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
    mo.md("# SBN Replay — Qualification d'anomalies réseau")


# ── IMPORT CONFIG (source unique de vérité pour les chemins) ─────────────────

@app.cell
def _(pathlib, sys):
    # PATCH TASK-33 / MIN-01 (audit_tmp, 2026-04-26): align path with current dir.
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


@app.cell
def _(RESULTS_DIR_DEFAULT, mo, pathlib):
    _RES = pathlib.Path(RESULTS_DIR_DEFAULT)
    file_input = mo.ui.text(
        value=str(_RES / "qualif_types_sbn.csv"),
        label="Fichier CSV qualif_types_sbn", full_width=True,
    )
    return file_input,


@app.cell
def _(file_input, mo):
    mo.vstack([file_input])


@app.cell
def _(file_input, pd):
    try:
        _df = pd.read_csv(file_input.value)
        _df["timestamp"] = pd.to_datetime(_df["timestamp"], errors="coerce")
        df = _df
        load_ok = True
        load_msg = f"**{len(df)} fenêtres** — {int(df['gate_open'].sum())} anomalies ({df['gate_open'].mean()*100:.1f}%)"
    except Exception as _e:
        df = pd.DataFrame()
        load_ok = False
        load_msg = f"Erreur : `{_e}`"
    return df, load_msg, load_ok


@app.cell
def _(load_msg, load_ok, mo):
    mo.callout(mo.md(load_msg), kind="success" if load_ok else "danger")


@app.cell
def _(df):
    ATTACK_TYPES = [
        "UDP_FLOOD", "SYN_FLOOD", "ICMP_FLOOD", "DNS_AMP", "HTTP_FLOOD",
        "SLOWLORIS", "PORT_SCAN", "DATA_EXFIL", "NETWORK_OUTAGE", "BGP_HIJACK",
        "BOTNET_CC", "Autre_Anomalie",
    ]
    b_cols     = [f"b_sbn_{t}"     for t in ATTACK_TYPES if f"b_sbn_{t}"     in df.columns]
    b_raw_cols = [f"b_sbn_raw_{t}" for t in ATTACK_TYPES if f"b_sbn_raw_{t}" in df.columns]
    actual_types = [t for t in ATTACK_TYPES if f"b_sbn_{t}" in df.columns]
    has_raw_cols = len(b_raw_cols) > 0   # FIX : flag explicite utilisé pour désactiver le switch

    TYPE_COLORS = {
        "UDP_FLOOD": "#e74c3c", "SYN_FLOOD": "#c0392b", "ICMP_FLOOD": "#e67e22",
        "DNS_AMP": "#f39c12", "HTTP_FLOOD": "#27ae60", "SLOWLORIS": "#16a085",
        "PORT_SCAN": "#2980b9", "DATA_EXFIL": "#8e44ad", "NETWORK_OUTAGE": "#7f8c8d",
        "BGP_HIJACK": "#d35400", "BOTNET_CC": "#1abc9c", "Autre_Anomalie": "#bdc3c7",
    }
    return ATTACK_TYPES, TYPE_COLORS, actual_types, b_cols, b_raw_cols, has_raw_cols


# ── CONTRÔLES ──────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("---\n## Contrôles de lecture")


@app.cell
def _(mo):
    is_playing   = mo.ui.switch(label="▶ Play", value=False)
    anomaly_only = mo.ui.switch(label="Anomalies uniquement", value=True)
    show_raw     = mo.ui.switch(label="Raw (avant temporal/UM)", value=False)
    return anomaly_only, is_playing, show_raw


@app.cell
def _(anomaly_only, has_raw_cols, is_playing, mo, show_raw):
    _items = [is_playing, anomaly_only, show_raw]
    if show_raw.value and not has_raw_cols:
        _out = mo.vstack([
            mo.hstack(_items, gap=4),
            mo.callout(
                mo.md("Colonnes b_sbn_raw_* absentes. "
                      "Relancez qualify_anomaly_sbn_v2.py pour les generer."),
                kind="warn"
            )
        ])
    else:
        _out = mo.hstack(_items, gap=4)
    _out


@app.cell
def _(mo):
    speed_sel = mo.ui.dropdown(
        options={"0.5 fps": "2000ms", "1 fps": "1000ms", "2 fps": "500ms",
                 "5 fps": "200ms", "10 fps": "150ms"},
        value="1 fps", label="Vitesse",
    )
    return speed_sel,


@app.cell
def _(mo, speed_sel):
    mo.hstack([speed_sel])


# ── TIMER ───────────────────────────────────────────────────────────────────

@app.cell
def _(is_playing, mo, speed_sel):
    if is_playing.value:
        timer = mo.ui.refresh(default_interval=speed_sel.value)
    else:
        timer = mo.ui.refresh()
    return timer,


@app.cell
def _(timer):
    timer


# ── ÉTAT POSITION ───────────────────────────────────────────────────────────

@app.cell
def _(mo):
    get_pos, set_pos = mo.state(0)
    return get_pos, set_pos


@app.cell
def _(df, anomaly_only):
    if anomaly_only.value and len(df) > 0:
        play_indices = list(df.index[df["gate_open"] == True])
    else:
        play_indices = list(range(len(df)))
    n_frames = len(play_indices)
    return n_frames, play_indices


# FIX : incrément uniquement quand is_playing est actif
@app.cell
def _(is_playing, n_frames, set_pos, timer):
    timer
    if n_frames > 0 and is_playing.value:
        set_pos(lambda p: (p + 1) % n_frames)


@app.cell
def _(mo, n_frames):
    _max = max(0, n_frames - 1)
    frame_slider = mo.ui.slider(
        0, _max, value=0,
        label="Position manuelle (pause uniquement)",
        show_value=True, full_width=True,
    )
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


# ── BARRE DE STATUT ─────────────────────────────────────────────────────────

@app.cell
def _(current_df_idx, current_pos, df, mo, n_frames):
    if len(df) == 0 or n_frames == 0:
        _out = mo.md("Aucune donnée.")
    else:
        _row    = df.iloc[current_df_idx]
        _ts     = _row.get("timestamp", "?")
        _gate   = bool(_row.get("gate_open", False))
        _top1   = str(_row.get("top1_type", ""))
        _top1_b = _row.get("top1_b", 0)
        _u      = _row.get("u_sbn", 0)

        if _gate and _top1 and _top1 != "nan":
            _detail = f"**{_top1}** (b={float(_top1_b):.3f}) | u={float(_u):.3f}"
        elif _gate:
            _detail = "Gate ouverte — type non classifié"
        else:
            _detail = "Gate fermée — pas d'anomalie"

        _out = mo.callout(
            mo.md(f"Fenêtre **{current_pos + 1} / {n_frames}** | `{_ts}` | {_detail}"),
            kind="danger" if _gate else "neutral",
        )
    _out


# ── TIMELINE GLOBALE ────────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("---\n## Vue globale — Timeline")


@app.cell
def _(actual_types, b_cols, current_df_idx, df, go, make_subplots):
    if len(df) == 0:
        _fig_tl = go.Figure()
    else:
        _fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.15, 0.55, 0.30],
            vertical_spacing=0.04,
            subplot_titles=["Gate", "Belief masses (fenêtres anomales)", "u_sbn  /  novelty"],
        )
        _x = df["timestamp"] if "timestamp" in df.columns else df.index

        _fig.add_trace(
            go.Scatter(x=_x, y=df["gate_open"].astype(float),
                       mode="lines", fill="tozeroy",
                       line=dict(color="rgba(231,76,60,0.9)", width=1),
                       fillcolor="rgba(231,76,60,0.3)", name="Gate"),
            row=1, col=1,
        )

        _da = df[df["gate_open"]].copy()
        _xa = _da["timestamp"] if "timestamp" in _da.columns else _da.index
        _palette = ["#e74c3c","#c0392b","#e67e22","#f39c12","#27ae60",
                    "#16a085","#2980b9","#8e44ad","#7f8c8d","#d35400","#1abc9c","#bdc3c7"]
        for _i, (_col, _typ) in enumerate(zip(b_cols, actual_types)):
            if _col in _da.columns:
                _fig.add_trace(
                    go.Bar(x=_xa, y=_da[_col].fillna(0), name=_typ,
                           marker_color=_palette[_i % len(_palette)],
                           hovertemplate=f"<b>{_typ}</b> %{{y:.3f}}<extra></extra>"),
                    row=2, col=1,
                )

        for _col2, _col_name, _color in [
            ("u_sbn", "u_sbn", "violet"),
            ("novelty_score", "novelty", "orange"),
        ]:
            if _col2 in _da.columns:
                _fig.add_trace(
                    go.Scatter(x=_xa, y=_da[_col2].fillna(0), mode="markers",
                               name=_col_name,
                               marker=dict(color=_color, size=4, opacity=0.7)),
                    row=3, col=1,
                )

        _cur_ts = str(df.iloc[current_df_idx].get("timestamp", current_df_idx))
        for _r in [1, 2, 3]:
            _fig.add_vline(x=_cur_ts, line_color="white", line_width=2,
                           line_dash="dash", row=_r, col=1)

        _fig.update_layout(
            height=480, barmode="stack",
            plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", font=dict(color="white"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=40, r=20, t=60, b=20),
        )
        _fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
        _fig.update_xaxes(showgrid=False)
        _fig_tl = _fig

    _fig_tl


# ── DÉTAIL FENÊTRE COURANTE ─────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("---\n## Détail de la fenêtre courante")


@app.cell
def _(actual_types, b_cols, b_raw_cols, current_df_idx, df, go, has_raw_cols, make_subplots, mo, show_raw):
    if len(df) == 0:
        _fig_det = go.Figure()
    else:
        _row  = df.iloc[current_df_idx]
        _gate = bool(_row.get("gate_open", False))

        # FIX : si show_raw demandé mais colonnes absentes, utiliser b_cols par défaut
        _use_raw_flag = show_raw.value and has_raw_cols
        _use_cols  = b_raw_cols if _use_raw_flag else b_cols
        _suffix    = " (raw — avant temporal/UM)" if _use_raw_flag else ""
        _u_col     = "u_sbn_raw" if _use_raw_flag else "u_sbn"

        _fig = make_subplots(
            rows=1, cols=2, column_widths=[0.65, 0.35],
            subplot_titles=[f"Belief masses{_suffix}", "Contrainte SL  Σb + u"],
            specs=[[{"type": "xy"}, {"type": "pie"}]],
        )
        _palette = ["#e74c3c","#c0392b","#e67e22","#f39c12","#27ae60",
                    "#16a085","#2980b9","#8e44ad","#7f8c8d","#d35400","#1abc9c","#bdc3c7"]

        if _gate:
            _beliefs = [float(_row.get(c, 0) or 0) for c in _use_cols]
            _u = float(_row.get(_u_col, 0) or 0)
            _top1 = str(_row.get("top1_type", ""))

            _bar_colors = []
            for _i in range(len(actual_types)):
                _hex = _palette[_i % len(_palette)]
                _r2, _g2, _b2 = int(_hex[1:3],16), int(_hex[3:5],16), int(_hex[5:7],16)
                if actual_types[_i] == _top1:
                    _bar_colors.append(_hex)
                else:
                    _bar_colors.append(f"rgba({_r2},{_g2},{_b2},0.4)")

            _fig.add_trace(
                go.Bar(
                    x=actual_types, y=_beliefs,
                    marker_color=_bar_colors,
                    marker_line_color="white",
                    marker_line_width=[3 if t == _top1 else 0 for t in actual_types],
                    text=[f"{v:.3f}" if v > 0.01 else "" for v in _beliefs],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>b = %{y:.4f}<extra></extra>",
                    name="Belief",
                ),
                row=1, col=1,
            )

            _sum_b = sum(_beliefs)
            _fig.add_trace(
                go.Pie(
                    labels=["Σb (croyances)", "u (incertitude)"],
                    values=[max(_sum_b, 1e-9), max(_u, 1e-9)],
                    hole=0.5,
                    marker=dict(colors=["#3498db", "#e74c3c"]),
                    textinfo="label+percent",
                ),
                row=1, col=2,
            )
            _fig.add_annotation(
                text=f"Σ={_sum_b+_u:.4f}", x=0.82, y=0.5,
                xref="paper", yref="paper", showarrow=False,
                font=dict(size=13, color="white"),
            )
        else:
            _fig.add_annotation(
                text="Gate fermée — aucune anomalie",
                x=0.3, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=18, color="#7f8c8d"),
            )

        _fig.update_layout(
            height=370, plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
            font=dict(color="white"), showlegend=False,
            margin=dict(l=40, r=20, t=50, b=80),
        )
        _fig.update_yaxes(range=[0, 1.1], showgrid=True,
                          gridcolor="rgba(255,255,255,0.1)", row=1, col=1)
        _fig.update_xaxes(tickangle=-40, row=1, col=1)
        _fig_det = _fig

    _fig_det


# ── HISTORIQUE GLISSANT ─────────────────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("---\n## Historique glissant (30 dernières fenêtres anomales)")


@app.cell
def _(TYPE_COLORS, actual_types, b_cols, current_df_idx, df, go):
    if len(df) == 0:
        _fig_hist = go.Figure()
    else:
        _past_anom = df.iloc[: current_df_idx + 1]
        _past_anom = _past_anom[_past_anom["gate_open"]].tail(30)
        _fig = go.Figure()

        if len(_past_anom) > 0:
            _x = _past_anom["timestamp"] if "timestamp" in _past_anom.columns else _past_anom.index
            for _col, _typ in zip(b_cols, actual_types):
                if _col in _past_anom.columns:
                    _fig.add_trace(go.Bar(
                        x=_x, y=_past_anom[_col].fillna(0),
                        name=_typ, marker_color=TYPE_COLORS.get(_typ, "#aaa"),
                        hovertemplate=f"<b>{_typ}</b> %{{y:.3f}}<extra></extra>",
                    ))
            if "u_sbn" in _past_anom.columns:
                _fig.add_trace(go.Scatter(
                    x=_x, y=_past_anom["u_sbn"].fillna(0),
                    mode="lines+markers", name="u_sbn",
                    marker=dict(color="white", size=5),
                    line=dict(color="white", width=1.5, dash="dot"),
                    yaxis="y2",
                ))
        else:
            _fig.add_annotation(text="Aucune anomalie encore dans l'historique",
                                x=0.5, y=0.5, xref="paper", yref="paper",
                                showarrow=False, font=dict(size=14, color="#7f8c8d"))

        _fig.update_layout(
            height=300, barmode="stack",
            plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", font=dict(color="white"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=50, t=30, b=70),
            yaxis=dict(range=[0, 1], title="Belief", showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis2=dict(range=[0, 1], title="u_sbn", overlaying="y", side="right", showgrid=False),
            xaxis=dict(tickangle=-40),
        )
        _fig_hist = _fig

    _fig_hist


# ── SCATTER NOVELTY vs INCERTITUDE ──────────────────────────────────────────

@app.cell
def _(mo):
    mo.md("---\n## Scatter Novelty vs Incertitude")


@app.cell
def _(TYPE_COLORS, actual_types, current_df_idx, df, go):
    if len(df) == 0 or "novelty_score" not in df.columns:
        _fig_sc = go.Figure()
    else:
        _past_anom = df.iloc[: current_df_idx + 1]
        _past_anom = _past_anom[_past_anom["gate_open"]]
        _fig = go.Figure()

        if len(_past_anom) > 0:
            for _typ in actual_types:
                _sub = _past_anom[_past_anom["top1_type"] == _typ]
                if len(_sub) == 0:
                    continue
                _fig.add_trace(go.Scatter(
                    x=_sub["novelty_score"].fillna(0),
                    y=_sub["u_sbn"].fillna(0),
                    mode="markers", name=_typ,
                    marker=dict(color=TYPE_COLORS.get(_typ, "#aaa"), size=8,
                                opacity=0.8, line=dict(color="white", width=0.5)),
                    hovertemplate=f"<b>{_typ}</b><br>novelty=%{{x:.3f}}<br>u=%{{y:.3f}}<extra></extra>",
                ))
            _fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.25)")
            _fig.add_vline(x=0.5, line_dash="dash", line_color="rgba(255,255,255,0.25)")
            _fig.add_annotation(x=0.82, y=0.92, text="Novel + incertain",
                                showarrow=False, font=dict(color="orange", size=10))
            _fig.add_annotation(x=0.18, y=0.08, text="Connu + confiant",
                                showarrow=False, font=dict(color="lightgreen", size=10))
        else:
            _fig.add_annotation(text="Aucune anomalie encore", x=0.5, y=0.5,
                               xref="paper", yref="paper", showarrow=False,
                               font=dict(size=14, color="#7f8c8d"))

        _fig.update_layout(
            height=340, plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
            font=dict(color="white"),
            xaxis=dict(title="Novelty score", range=[-0.05, 1.05],
                      showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Incertitude u_sbn", range=[-0.05, 1.05],
                      showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=60, r=20, t=30, b=60),
        )
        _fig_sc = _fig

    _fig_sc


if __name__ == "__main__":
    app.run()