import pandas as pd
import numpy as np
import joblib
import sys
import os
import time
import tracemalloc
from sl_ads.paths import get_version_names, get_model_path  # Phase H
from sl_ads.train.model_guardrails import validate_required_model_coverage

from sl_ads.preprocessing_utils import preprocess_metrics  # Phase H
from sl_ads.calendar.regime import regime_of  # PATCH H2 (calendar-aware EVT dispatch)

# Priorité haute pour éviter la préemption sur E-cores (Intel hybrid architectures)
try:
    import psutil as _psutil
    _psutil.Process(os.getpid()).nice(_psutil.HIGH_PRIORITY_CLASS)
except Exception:
    pass  # non-bloquant : Windows seulement, peut nécessiter des droits admin

try:
    from sl_ads.config import CONFIG  # Phase H
except ImportError:
    print("❌ CRITIQUE : Impossible de charger sl_ads.config.")
    sys.exit(1)

# --- PARAMÈTRES DE SORTIE (centralisés depuis config.py) ---
VERSION_NAME, _ = get_version_names(CONFIG)

EVIDENCE_CSV     = f"evidence_{VERSION_NAME}.csv"
RAW_DATA_CSV     = f"raw_data_{VERSION_NAME}.csv"
OUTPUT_DIR       = os.path.join("../results", f"resultats_{VERSION_NAME}")
MODEL_PATH       = get_model_path(CONFIG, up_levels=1)
WINDOW_SIZE      = CONFIG.get('WINDOW_SIZE', 10)
T_TRAPEZE_RATIO  = CONFIG.get('T_TRAPEZE_RATIO', 0.1)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_instantaneous_evidence(signed_residual: float,
                                    t_susp: float,
                                    t_atk: float,
                                    t_trapeze_base: float,
                                    direction: str) -> tuple[float, float, float]:
    """
    Mapping trapézoïdal directionnel résidu signé → triplet d'évidence (p, s, n).

    Extension de fuzzy_trapezoid_relative avec support du filtrage directionnel
    et de seuils asymétriques issus de train_v10 (pos/neg/both).

    Parameters
    ----------
    signed_residual : float  — résidu signé e_t = y_t − ŷ_t
    t_susp          : float  — seuil de suspicion (début zone Suspect plafond)
    t_atk           : float  — seuil d'anomalie (début zone Attack plafond)
    t_trapeze_base  : float  — début de la rampe Safe→Suspect
                               (= T_TRAPEZE_RATIO × t_susp, cf. config.py)
    direction       : str    — 'pos' : seuls résidus > 0 sont suspects
                               'neg' : seuls résidus < 0 sont suspects
                               'both': seuils asymétriques pos/neg (routé en amont)
                               None / autre : symétrique sur |résidu|

    Returns
    -------
    (p, s, n) : tuple[float, float, float], p + s + n = 1.0

    Invariant
    ---------
    Pour une fenêtre de W pas, la somme P = Σp, S = Σs, N = Σn vérifie
    P + S + N = W. Le vecteur [P, S, N] constitue le vecteur de preuves r
    de la bijection évidence→opinion (Jøsang 2016, Def. 3.9).
    """
    # Filtrage directionnel : un résidu dans le mauvais sens est preuve de normalité
    if direction == 'pos':
        if signed_residual <= 0.0:
            return (1.0, 0.0, 0.0)
        e = signed_residual
    elif direction == 'neg':
        if signed_residual >= 0.0:
            return (1.0, 0.0, 0.0)
        e = abs(signed_residual)
    else:
        # Symétrique : l'amplitude seule compte
        e = abs(signed_residual)

    # Mapping trapézoïdal linéaire par morceaux
    if e < t_trapeze_base:
        return (1.0, 0.0, 0.0)
    elif e < t_susp:
        width = t_susp - t_trapeze_base
        alpha = (e - t_trapeze_base) / width if width > 0 else 1.0
        return (1.0 - alpha, alpha, 0.0)
    elif e < t_atk:
        width = t_atk - t_susp
        alpha = (e - t_susp) / width if width > 0 else 1.0
        return (0.0, 1.0 - alpha, alpha)
    else:
        return (0.0, 0.0, 1.0)

# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def compute_evidence():
    """
    Calcule les triplets de preuves brutes (P, S, N) par fenêtre temporelle et par
    indicateur prédictif (Prophet + reconstruction linéaire).

    Pour chaque fenêtre de WINDOW_SIZE pas, le résidu signé (y_t − ŷ_t) est converti
    en évidence instantanée via un mapping trapézoïdal directionnel, puis accumulé.
    Le vecteur résultant r = [P, S, N] satisfait P + S + N = n (avec n = taille
    effective de la fenêtre) et constitue l'entrée de la bijection évidence→opinion
    (Jøsang 2016, Def. 3.9) dans l'étape suivante (compute_opinions_v3.py).

    ─── PATCH M-06 / F09 (2026-04-21) : ROBUSTESSE DES FENÊTRES PARTIELLES ─────
    La dernière fenêtre d'une trace peut être plus petite que WINDOW_SIZE (ex:
    3 points au lieu de 10 si la trace s'arrête au milieu d'une fenêtre). On ne
    la drop pas : elle est **acceptée avec sa taille réduite n < WINDOW_SIZE**.

    Conséquence mathématique documentée :
        - L'invariant sur les fenêtres pleines est P + S + N = WINDOW_SIZE.
        - Sur la dernière fenêtre partielle : P + S + N = n < WINDOW_SIZE.
        - L'incertitude résultante u = W / (W + n) est **systématiquement plus
          élevée** que sur les fenêtres pleines (Jøsang Def. 3.9 : moins de
          preuves → plus d'incertitude). C'est sémantiquement correct : on a
          moins d'information donc on est moins confiant.

    Choix de conception :
        - Pas de padding (qui ajouterait du signal synthétique non observé) ;
        - Pas de drop silencieux (qui perdrait les dernières minutes d'activité
          potentiellement critiques dans un pipeline temps-réel) ;
        - Incertitude plus élevée est la représentation SL-correcte de
          "dernière fenêtre observée partiellement".

    Les évaluateurs downstream (evaluate_injection_v2, evaluate_qualify_sbn)
    n'ont pas à traiter ces fenêtres différemment : la bijection SL fait
    émerger l'incertitude additionnelle naturellement.

    Sorties :
        evidence_{VERSION}.csv   — triplets P/S/N + RMSE + IW par fenêtre
        raw_data_{VERSION}.csv   — résidus instantanés par pas (debug)
        metadata_{VERSION}.csv   — profil par indicateur (type, R², seuils, direction)
    """
    print(f"--- COMPUTE EVIDENCE ({VERSION_NAME}) ---")
    tracemalloc.start()
    t_global_start = time.perf_counter()

    # Chargement données
    print("-> Loading dataset...")
    df = pd.read_csv(CONFIG['file_path'])
    df['ds'] = pd.to_datetime(df['timestamp'])
    # Gestion des valeurs manquantes — politique UNIQUE partagée avec train_v10.py
    # (voir preprocessing_utils.preprocess_metrics) :
    #   1. forward-fill LIMITÉ (limit=NAN_FFILL_LIMIT, ~5 min) pour les trous courts ;
    #   2. les trous longs restent NaN (signalement aval) ;
    #   3. fillna(0) est volontairement ÉVITÉ sur les métriques réseau : 0 bytes
    #      n'est pas équivalent à "absence de mesure" (le modèle apprend alors des
    #      résidus artificiels sur les périodes non-instrumentées).
    # Cette politique doit rester IDENTIQUE à celle de train_v10.py pour éviter
    # tout désalignement train/inférence (constat #2 de l'audit scientifique).
    df = df.sort_values('ds')
    df = preprocess_metrics(df, limit_ffill=CONFIG.get("NAN_FFILL_LIMIT", 10))

    holidays_data = CONFIG.get('HOLIDAYS_LIST', [])
    holidays_df   = pd.DataFrame(holidays_data)
    is_cal_wknd   = df['ds'].dt.dayofweek >= 5
    if not holidays_df.empty:
        holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
        is_holiday   = df['ds'].dt.normalize().isin(holidays_df['ds'].dt.normalize())
    else:
        is_holiday = pd.Series(False, index=df.index)
    df['on_weekend'] = (is_cal_wknd | is_holiday).astype(int)
    df['on_weekday'] = (1 - df['on_weekend']).astype(int)

    # Chargement modèles
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Modèle '{MODEL_PATH}' introuvable. Lancez train_v10.py d'abord.")
        return
    print(f"-> Loading models from {MODEL_PATH}...")
    models_pkg = joblib.load(MODEL_PATH)
    validate_required_model_coverage(
        models_pkg,
        CONFIG,
        failures=models_pkg.get("_training_failures", []),
    )

    # Vérification anti-fuite train/test.
    # train_v10.py stocke _meta_split_date dans le pkg lors du dump.
    # Si la split_date du modèle diffère de celle de la config courante, les
    # métriques de détection pourraient être invalides (évaluation sur données
    # déjà vues à l'entraînement, ou split incohérent).
    _pkg_split = models_pkg.get('_meta_split_date')
    _cfg_split = str(CONFIG.get('split_date', ''))
    if _pkg_split is None:
        print("[WARN] _meta_split_date absent du pkg (artefact pré-v10). "
              "Vérification anti-fuite ignorée.")
    elif _pkg_split != _cfg_split:
        print(f"❌ CRITIQUE : split_date incohérente — modèle entraîné avec "
              f"split_date='{_pkg_split}', config courante='{_cfg_split}'.\n"
              f"   Relancez train_v10.py avec la config actuelle ou corrigez split_date.")
        return

    # Split test set
    if CONFIG.get("TUNING_MODE", False):
        start = pd.to_datetime(CONFIG['TUNING_START'])
        end   = pd.to_datetime(CONFIG['TUNING_END'])
        print(f"-> TUNING MODE: {start} to {end}")
        test_df = df[(df['ds'] >= start) & (df['ds'] <= end)].copy().reset_index(drop=True)
    else:
        split_date = pd.to_datetime(CONFIG['split_date'])
        test_df    = df[df['ds'] > split_date].copy().reset_index(drop=True)

    # Identification des métriques (exclure la clé EDP)
    prophet_keys = [k for k in models_pkg if k.startswith('prophet_') and isinstance(models_pkg[k], dict)]
    reconst_keys = [k for k in models_pkg if k.startswith('reconst_') and isinstance(models_pkg[k], dict)]
    all_metric_keys = prophet_keys + reconst_keys

    # R² clampé à [0, 1] : un R² négatif (modèle pire que la moyenne) est traité
    # comme poids nul pour ne pas inverser la pondération downstream (trust_discount).
    metric_weights = {
        key: max(0.0, float(models_pkg[key].get('r2_score', 0.5)))
        for key in all_metric_keys
    }

    # Validation des seuils avant la boucle — un seuil absent rend tout le run invalide.
    # On vérifie en amont pour éviter un plantage après plusieurs heures de calcul.
    for key in all_metric_keys:
        pkg = models_pkg[key]
        _thr = pkg.get('thresholds', {})
        if 't_susp' not in pkg and 'suspect' not in _thr:
            print(f"❌ CRITIQUE [{key}] : aucun seuil t_susp trouvé dans le pkg. "
                  f"Relancez train_v10.py.")
            return
        if 't_atk' not in pkg and 'attack' not in _thr:
            print(f"❌ CRITIQUE [{key}] : aucun seuil t_atk trouvé dans le pkg. "
                  f"Relancez train_v10.py.")
            return

    evidence_rows = []
    raw_rows      = []
    total_rows    = len(test_df)
    proc_times    = []

    print(f"-> Analyzing {total_rows} frames in windows of {WINDOW_SIZE}...")

    # ==========================================================================
    # BOUCLE PAR FENÊTRE
    # ==========================================================================
    # PATCH M-06 / F09 (2026-04-21) : on accepte la dernière fenêtre partielle
    # (len(batch) < WINDOW_SIZE) en la laissant passer : l'incertitude SL u
    # résultante sera plus élevée (moins de preuves → plus d'u), ce qui est la
    # représentation SL-correcte d'une observation partielle. Voir docstring
    # compute_evidence() pour la justification complète.
    _n_partial_windows = 0
    for i in range(0, total_rows, WINDOW_SIZE):
        t_start = time.perf_counter()
        batch   = test_df.iloc[i:i + WINDOW_SIZE].copy()
        if len(batch) < 1:
            break
        if len(batch) < WINDOW_SIZE:
            _n_partial_windows += 1

        current_date = batch['ds'].iloc[-1]
        ev_row = {'timestamp': current_date}

        for key in all_metric_keys:
            pkg       = models_pkg[key]
            model     = pkg['model']
            clean_key = key.replace("->", "_to_")

            vals_real, vals_pred = [], []

            # Prédiction
            if pkg['type'] == 'prophet':
                fcst        = model.predict(batch)
                metric_name = key.replace('prophet_', '')
                vals_real   = batch[metric_name].values
                vals_pred   = fcst['yhat'].values
                # Largeur moyenne de l'intervalle de confiance Prophet sur la fenêtre.
                # Sauvegardée dans le CSV evidence pour usage optionnel downstream.
                iw = np.mean(fcst['yhat_upper'].values - fcst['yhat_lower'].values)
                ev_row[f"{clean_key}_iw"] = max(float(iw), 1e-6)

            elif pkg['type'] == 'reconstruction':
                target_name = key.split('_from_')[0].replace('reconst_', '')
                feat_name   = key.split('_from_')[1]
                vals_real   = batch[target_name].values
                vals_pred   = model.predict(batch[[feat_name]].values)
                ev_row[f"{clean_key}_iw"] = np.nan

            vals_real_arr = np.array(vals_real)
            vals_pred_arr = np.array(vals_pred)

            # RMSE fenêtre (pour C3 online_rmse)
            rmse_win = np.sqrt(np.mean((vals_real_arr - vals_pred_arr) ** 2))
            ev_row[f"{clean_key}_rmse"] = max(float(rmse_win), 1e-6)

            # Données brutes (30s) pour CSV de debug
            abs_errors = np.abs(vals_real_arr - vals_pred_arr)
            for j in range(len(batch)):
                raw_rows.append({
                    'timestamp':  batch['ds'].iloc[j],
                    'metric_key': key,
                    'real':       vals_real[j],
                    'pred':       vals_pred[j],
                    'abs_error':  abs_errors[j]
                })

            # PATCH H2 — calendar-aware EVT dispatch.
            # If the pkg carries a ``thresholds_per_regime`` block (produced
            # by ``calibrate_thresholds_per_regime_v2`` when training was
            # run with ``CALENDAR_EVT_ENABLED=True``), look up the bucket
            # for ``current_date`` and source the thresholds from there.
            # Otherwise use the legacy scalar fields.  ``direction`` is
            # always read from the legacy field (it is metric-level, not
            # regime-level).
            _per_regime = pkg.get('thresholds_per_regime')
            if _per_regime is not None and isinstance(_per_regime, dict) and \
                    'thresholds_per_regime' in _per_regime:
                _holidays  = CONFIG.get('HOLIDAYS_LIST')
                _bucket    = regime_of(current_date, holidays=_holidays)
                _bucket_th = _per_regime['thresholds_per_regime'].get(_bucket)
                if _bucket_th:
                    pkg_thresh = _bucket_th
                else:  # pragma: no cover — defensive
                    pkg_thresh = pkg
            else:
                pkg_thresh = pkg

            # Lecture des seuils (validés avant la boucle, jamais absents ici)
            _thresholds = pkg.get('thresholds', {})
            t_susp = pkg_thresh.get('t_susp', _thresholds.get('suspect', pkg.get('t_susp')))
            t_atk  = pkg_thresh.get('t_atk',  _thresholds.get('attack',  pkg.get('t_atk')))
            # t_trapeze_base : début de la zone de transition Safe→Suspect.
            # Contrôlé par T_TRAPEZE_RATIO dans config.py (défaut 0.1 = 10 % de t_susp).
            t_trapeze_base = pkg_thresh.get('t_trapeze_base', T_TRAPEZE_RATIO * t_susp)
            direction      = pkg.get('direction', None)

            t_susp_pos         = pkg_thresh.get('t_susp_pos',         t_susp)
            t_atk_pos          = pkg_thresh.get('t_atk_pos',          t_atk)
            t_trapeze_base_pos = pkg_thresh.get('t_trapeze_base_pos', t_trapeze_base)
            t_susp_neg         = pkg_thresh.get('t_susp_neg',         t_susp)
            t_atk_neg          = pkg_thresh.get('t_atk_neg',          t_atk)
            t_trapeze_base_neg = pkg_thresh.get('t_trapeze_base_neg', t_trapeze_base)

            P, S, N = 0.0, 0.0, 0.0
            # Composantes directionnelles (frame 5-états, direction='both' uniquement).
            # Invariant : S = S_pos + S_neg, N = N_pos + N_neg (Jøsang §3.5.4 coarsening).
            # P (safe) est partagé entre les deux directions → non splitté.
            S_pos, N_pos, S_neg, N_neg = 0.0, 0.0, 0.0, 0.0
            for j in range(len(batch)):
                signed_r = vals_real_arr[j] - vals_pred_arr[j]
                if direction == 'both':
                    if signed_r >= 0:
                        p, s, n = compute_instantaneous_evidence(signed_r, t_susp_pos, t_atk_pos, t_trapeze_base_pos,'pos')
                        S_pos += s;
                        N_pos += n
                    else:
                        p, s, n = compute_instantaneous_evidence(signed_r, t_susp_neg, t_atk_neg, t_trapeze_base_neg,'neg')
                        S_neg += s;
                        N_neg += n
                else:
                    p, s, n = compute_instantaneous_evidence(signed_r, t_susp, t_atk, t_trapeze_base, direction)
                P += p
                S += s
                N += n

            ev_row[f"{clean_key}_P"] = P
            ev_row[f"{clean_key}_S"] = S
            ev_row[f"{clean_key}_N"] = N
            if direction == 'both':
                ev_row[f"{clean_key}_S_pos"] = S_pos
                ev_row[f"{clean_key}_N_pos"] = N_pos
                ev_row[f"{clean_key}_S_neg"] = S_neg
                ev_row[f"{clean_key}_N_neg"] = N_neg

        evidence_rows.append(ev_row)
        proc_times.append(time.perf_counter() - t_start)

    # PATCH M-06 / F09 : informer l'utilisateur sur les fenêtres partielles
    # acceptées (uniquement la queue en pratique : la dernière fenêtre si
    # total_rows % WINDOW_SIZE != 0).
    if _n_partial_windows > 0:
        print(f"   INFO: {_n_partial_windows} partial window(s) accepted (size < {WINDOW_SIZE}). "
              f"Higher SL uncertainty on these rows is mathematically correct.")

    # ==========================================================================
    # SAUVEGARDE
    # ==========================================================================
    curr_ram, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    df_evidence = pd.DataFrame(evidence_rows)

    meta_rows = []
    for key in all_metric_keys:
        pkg       = models_pkg[key]
        clean_key = key.replace("->", "_to_")
        r2_w      = metric_weights.get(key, 0.5)
        _thr      = pkg.get('thresholds', {})
        meta_rows.append({
            'metric_key':        key,
            'clean_key':         clean_key,
            'type':              pkg['type'],
            'r2_weight':         r2_w,
            'threshold_suspect': pkg.get('t_susp', _thr.get('suspect', 0)),
            'threshold_attack':  pkg.get('t_atk',  _thr.get('attack',  0)),
            'direction':         pkg.get('direction', 'sym'),
            'kurtosis':          pkg.get('kurtosis', float('nan')),
            'cv':                pkg.get('cv',       float('nan')),
        })

    meta_path = os.path.join(OUTPUT_DIR, f"metadata_{VERSION_NAME}.csv")
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)

    ev_path = os.path.join(OUTPUT_DIR, EVIDENCE_CSV)
    df_evidence.to_csv(ev_path, index=False)

    raw_path = os.path.join(OUTPUT_DIR, RAW_DATA_CSV)
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)

    total_time      = time.perf_counter() - t_global_start
    avg_window_time = np.mean(proc_times) if proc_times else 0

    print(f"\n{'='*60}")
    print(f"  Evidence computed in {total_time:.1f}s ({len(evidence_rows)} windows)")
    print(f"  Avg window time: {avg_window_time*1000:.1f}ms")
    print(f"  Peak RAM: {peak_ram / (1024**2):.1f} MB")
    print(f"  📄 Evidence  : {ev_path}")
    print(f"  📄 Raw data  : {raw_path}")
    print(f"  📄 Metadata  : {meta_path}")
    print(f"{'='*60}\n")
    print("✅ Evidence computation complete.")


if __name__ == "__main__":
    compute_evidence()
