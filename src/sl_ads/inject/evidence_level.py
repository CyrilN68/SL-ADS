"""
inject_at_evidence_level.py — Injection d'attaques dans le CSV d'evidence
==========================================================================
VERSION v3

PRINCIPE D'INJECTION :
    Ce module opère directement sur le CSV de preuves brutes (evidence_{VERSION}.csv)
    produit par compute_evidence_v2.py. Il substitue les triplets (P, S, N) des
    fenêtres temporelles concernées par des valeurs de signature précalibrées,
    permettant d'évaluer la détectabilité et la qualifiabilité du système SL-ADS
    sur un dataset (RedeRio) sans attaques réelles étiquetées.

CONVENTION DES TUPLES DE SIGNATURE :
    (ev_attack, ev_suspect, ev_normal)
    ev_attack  → stocké dans col_N (= evidence "attack" dans le pipeline SL)
    ev_normal  → stocké dans col_P (= evidence "safe" dans le pipeline SL)
    ev_suspect → stocké dans col_S
    L'inversion P↔N est intentionnelle : voir inject_attack_into_evidence().

    Les valeurs brutes dans le catalogue sont des poids relatifs (pas des preuves
    absolues). Avant injection, chaque signature est normalisée de sorte que
    P + S + N = WINDOW_SIZE, maintenant l'invariant de la bijection SL (Def. 3.9).

PROFIL DE RAMP (montée progressive) :
    L'injection utilise une interpolation linéaire entre l'état normal (P=WINDOW_SIZE,
    S=0, N=0) et la signature normalisée d'attaque, pondérée par un profil alpha
    trapézoïdal (Jøsang 2016, Def. 3.9 ; inspiration CIC-IDS2017).
    Propriété : P+S+N = WINDOW_SIZE est maintenu à tout alpha ∈ [0,1].

PIPELINE :
    evidence_{VERSION}.csv
            ↓  ce script
    evidence_{VERSION}_attacks.csv   (+colonne injection_label, +injection_ramp_alpha)
            ↓  compute_opinions_v3.py
    detection_results_INJECTED.csv  ✅

BASES SCIENTIFIQUES NOUVELLES MÉTRIQUES :
    icmp   : Moustafa & Slay (2015) UNSW-NB15 — icmp_count feat. top-5 DoS
    udp    : Sharafaldin et al. (2018) CIC-IDS2017 — udp_count feat. discriminante
    tcp    : Sharafaldin et al. (2018) Table III — tcp feat. SYN flood
    fin    : Roesch (1999) Snort — fin/syn ratio signature half-open
    fin←syn: Roesch (1999) ; Lippmann et al. (2000) MIT Lincoln Lab eval.
    tcp←pkt: Mirsky et al. (2018) Kitsune — tcp_pkt_ratio feat.
    udp←flo: Cloudflare DDoS Threat Report 2024 — udp_flow_ratio
    ntp_amp : van Rijswijk-Deij (2014) APNIC/IMC — BAF NTP=556.9 ; Czyz 2014 IMC
    ssh_bf  : Najafabadi (2015) ICMLA ; Hynek (2020) IFIP SEC ; Hofstede (2014) SSHCure
    dns_tun : Sharma (2018) Procedia CS ; Habibi (2019) IEEE IM ; MDPI Elec. (2023)

NOTE ASYMMETRIC_THRESHOLD_METRICS :
    Les nouvelles métriques Prophet (icmp, udp, tcp, fin) ont été entraînées
    avec dir='sym' car elles ne figurent pas encore dans ASYMMETRIC_THRESHOLD_METRICS
    du config. Pour un futur retrain, ajouter dans CONFIG['ASYMMETRIC_THRESHOLD_METRICS']:
        "icmp"  : "pos",   # ICMP flood = excès ICMP positif
        "udp"   : "pos",   # UDP  flood = excès UDP positif
        "tcp"   : "pos",   # SYN  flood = excès TCP positif
        "fin"   : "sym",   # SYN flood : fin baisse ; Slowloris : fin monte puis baisse
        "fin_syn"   : "sym",   # résidu fin-from-syn : négatif = half-open
        "tcp_packets": "sym",  # résidu tcp-from-packets
        "udp_flows"  : "sym",  # résidu udp-from-flows
"""
import shutil
import sys
import pandas as pd
import numpy as np
import os
from sl_ads.paths import get_version_names, get_results_dir  # Phase H

from sl_ads.config import INJECTED_ATTACK_CATALOG as _CANONICAL_ORDER  # Phase H
_CANONICAL_NAMES = [a['name'] for a in _CANONICAL_ORDER]

try:
    from sl_ads.config import CONFIG  # Phase H
except ImportError:
    print("❌ CRITIQUE : Impossible de charger sl_ads.config.")
    sys.exit(1)

# ==============================================================================
# PARAMÈTRES — chemins centralisés depuis config.py
# ==============================================================================
VERSION_NAME, VERSION_NAME_MODIF = get_version_names(CONFIG)
RESULTS_DIR = get_results_dir(CONFIG, up_levels=1)
INPUT_EVIDENCE_CSV = os.path.join(RESULTS_DIR, f"evidence_{VERSION_NAME}.csv")
OUTPUT_EVIDENCE_CSV = os.path.join(RESULTS_DIR, f"evidence_{VERSION_NAME_MODIF}.csv")
# Taille de fenêtre temporelle — doit correspondre à celle de compute_evidence_v2
# (WINDOW_SIZE pas de 30 s pour RedeRio → fenêtres de 5 min).
# Utilisé pour normaliser les signatures (maintien de l'invariant P+S+N = WINDOW_SIZE).
WINDOW_SIZE = CONFIG.get('WINDOW_SIZE', 10)


# ==============================================================================
# CATALOGUE D'ATTAQUES
#
# Structure de chaque entrée :
#   "prophet_{metric}"             → columns prophet_{metric}_P / _S / _N
#   "reconst_{target}_from_{feat}" → columns reconst_{target}_from_{feat}_P / _S / _N
#
# Le script détecte automatiquement les colonnes disponibles —
# les métriques absentes du CSV sont silencieusement ignorées.
# ==============================================================================

ATTACK_CATALOG = [

    # ──────────────────────────────────────────────────────────────────────────
    # 0. UNKNOWN_ANOMALY_CONTROL — pattern de contrôle hors-catalogue
    # But : vérifier que qualify_anomaly.py signale u_qualif > 0.40
    # Design : aucun groupe sémantique ne doit dominer clairement.
    #   - Volume modéré (pas DDoS)
    #   - ICMP/UDP/TCP légèrement élevés mais non-discriminants
    #   - Flux ouverts (comme Slowloris) MAIS bytes non nuls (≠ Slowloris)
    #   - Reconstruction non cassée (≠ floods volumétriques)
    # Aucune correspondance MITRE ATT&CK : pattern synthétique de contrôle.
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "UNKNOWN_ANOMALY_CONTROL",
        "type": "Anomalie de contrôle — Pattern inconnu",
        "start": "2025-12-20 10:00:00",
        "duration_h": 2.0,
        "intensity": "high",
        "ramp_frac": 0.10,
        "signature": {
            # ── Métriques existantes ──────────────────────────────────────────
            "prophet_bytes":                       (12.0, 6.0, 1.0),  # Vol:Anom (comme DDoS)
            "prophet_packets":                     (1.0,  3.0, 8.0),  # Pkts:Safe ← anti-flood
            "prophet_flows":                       (10.0, 5.0, 2.0),  # Connexions:Anom
            "prophet_syn":                         (8.0,  4.0, 2.0),  # SYN:Anom
            "prophet_entropy_src_ip":              (10.0, 4.0, 1.0),  # Ent_ip:Anom
            "prophet_entropy_src_port":            (1.0,  3.0, 8.0),  # Ent_sport:Safe
            "prophet_entropy_dst_port":            (1.0,  3.0, 8.0),  # Ent_dport:Safe ← anti-scan
            "prophet_avg_pkt_size":                (1.0,  3.0, 8.0),  # Pkt:Safe ← anti-ICMP/DNS
            "reconst_bytes_from_packets":          (1.0,  4.0, 7.0),  # Reconstruction:Safe
            "reconst_bytes_from_entropy_src_port": (1.0,  3.0, 8.0),  # Reconstruction2:Safe
            # ── Nouvelles métriques protocole ────────────────────────────────
            # Aucun protocole ne domine → pattern non classifiable
            "prophet_icmp":                        (3.0,  3.0, 5.0),  # ICMP légèrement ↑ (ambigu)
            "prophet_udp":                         (2.0,  3.0, 6.0),  # UDP légèrement ↑ (ambigu)
            "prophet_tcp":                         (5.0,  4.0, 3.0),  # TCP modéré (ambigu)
            "prophet_fin":                         (1.0,  3.0, 7.0),  # FIN bas (pas HTTP_FLOOD)
            "reconst_fin_from_syn":                (3.0,  4.0, 5.0),  # ratio fin/syn légèrement anormal
            "reconst_tcp_from_packets":            (3.0,  4.0, 5.0),  # tcp/pkt légèrement anormal
            "reconst_udp_from_flows":              (2.0,  3.0, 6.0),  # udp/flow légèrement ↑
        }
    },

    # ──────────────────────────────────────────────────────────────────────────
    # 1. UDP FLOOD DDoS — volumétrique intense
    # Ref : Cloudflare DDoS Threat Report 2024, §3.1 ; CIC-IDS2017 DoS UDP
    # Caractéristiques :
    #   bytes↑↑↑, packets↑↑↑ (trafic UDP massif sans état)
    #   udp↑↑↑ (discriminateur vs ICMP_FLOOD)
    #   icmp≈normal (discriminateur vs ICMP_FLOOD — signal Safe actif)
    #   syn≈0, fin≈0 (UDP sans TCP handshake)
    #   tcp↓ relatif (tcp/packets chute car UDP domine)
    #   reconstruction cassée (bytes/packets ratio anormal : paquets UDP < Ethernet MTU)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "UDP_FLOOD_DDOS",
        "type": "DDoS Volumétrique — UDP Flood",
        "start": "2025-11-16 14:00:00",
        "duration_h": 4.0,
        "intensity": "extreme",
        "ramp_frac": 0.10,
        "signature": {
            # ── Métriques existantes ──────────────────────────────────────────
            "prophet_bytes":                       (40.0, 8.0,  0.5),
            "prophet_packets":                     (35.0, 7.0,  0.5),
            "prophet_flows":                       (5.0,  10.0, 3.0),
            "prophet_syn":                         (0.5,  2.0,  8.0),  # UDP → pas de SYN
            "prophet_entropy_src_ip":              (2.0,  4.0,  5.0),
            "prophet_entropy_src_port":            (8.0,  5.0,  2.0),  # ports source random
            "prophet_entropy_dst_port":            (1.0,  3.0,  6.0),
            "prophet_avg_pkt_size":                (6.0,  8.0,  3.0),
            "reconst_bytes_from_packets":          (30.0, 8.0,  0.5),
            "reconst_bytes_from_entropy_src_port": (20.0, 6.0,  1.0),
            # ── Nouvelles métriques protocole ────────────────────────────────
            # Ref Sharafaldin 2018 : udp_pkt_count feature discriminante UDP flood
            "prophet_udp":                         (35.0, 7.0,  0.5),  # UDP ↑↑↑ — discriminateur clé
            # Signal Safe actif pour ICMP : enseigne au classifieur que ce n'est PAS ICMP_FLOOD
            "prophet_icmp":                        (0.0,  1.0,  9.0),  # ICMP normal — discriminateur ICMP
            # TCP non affecté par UDP flood
            "prophet_tcp":                         (0.0,  1.0,  9.0),  # TCP normal (pas de SYN)
            # FIN absent (UDP sans connexion TCP)
            "prophet_fin":                         (0.0,  1.0,  9.0),  # FIN ≈ 0 (pas de TCP)
            # fin/syn : ni syn ni fin → résidu RANSAC ≈ 0 → neutre
            "reconst_fin_from_syn":                (1.0,  2.0,  8.0),  # ratio fin/syn neutre
            # tcp/packets : tcp reste stable, packets↑↑ → ratio tcp/packets chute
            # Ref Mirsky 2018 Kitsune : tcp_pkt_ratio est perturbé par UDP flood
            "reconst_tcp_from_packets":            (15.0, 5.0,  1.0),  # tcp/pkt ↓↓ (UDP noie TCP)
            # udp/flows : UDP ↑↑↑ , flows légèrement ↑ → ratio explose
            "reconst_udp_from_flows":              (25.0, 6.0,  0.5),  # udp/flow ↑↑↑
        }
    },

    # ──────────────────────────────────────────────────────────────────────────
    # 2. SYN FLOOD DDoS — TCP half-open connections
    # Ref : CIC-IDS2017 Sharafaldin 2018 Table III ; MITRE ATT&CK T1498.001
    # Caractéristiques :
    #   syn↑↑↑ (explosion SYN sans réponse ACK)
    #   fin≈0 (aucune connexion complétée → demi-ouvertes abandonnées)
    #   fin/syn → 0 : signature SYN flood classique (Roesch 1999 Snort)
    #   avg_pkt_size↓ (SYN seuls ~60 bytes vs trafic normal ~640 bytes/pkt sur RedeRio)
    #   icmp=0, udp=0 (pure TCP)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "SYN_FLOOD_DDOS",
        "type": "DDoS TCP — SYN Flood",
        "start": "2025-11-21 02:30:00",
        "duration_h": 0.75,
        "intensity": "extreme",
        "ramp_frac": 0.15,
        "signature": {
            # ── Métriques existantes ──────────────────────────────────────────
            "prophet_bytes":                       (4.0,  8.0,  2.0),  # bytes modérés (SYN sans payload)
            "prophet_packets":                     (18.0, 8.0,  0.5),
            "prophet_flows":                       (8.0,  6.0,  2.0),
            "prophet_syn":                         (45.0, 5.0,  0.0),  # SYN ↑↑↑ — signal primaire
            "prophet_entropy_src_ip":              (12.0, 5.0,  0.5),  # IPs spoofées
            "prophet_entropy_src_port":            (10.0, 4.0,  1.0),
            "prophet_entropy_dst_port":            (1.0,  2.0,  7.0),
            "prophet_avg_pkt_size":                (12.0, 5.0,  1.0),  # paquets SYN ~60B → anomalie taille
            "reconst_bytes_from_packets":          (8.0,  7.0,  2.0),
            "reconst_bytes_from_entropy_src_port": (5.0,  6.0,  2.0),
            # ── Nouvelles métriques protocole ────────────────────────────────
            # tcp↑↑↑ car SYN flood = TCP uniquement
            # Ref Sharafaldin 2018 : tcp_count feature discriminante SYN flood
            "prophet_tcp":                         (25.0, 5.0,  0.5),  # TCP ↑↑↑ (tous SYN)
            # fin≈0 : aucune connexion complétée (half-open)
            # Prophet temporal : fin reste bas → signal faible (baseline déjà bas sur RedeRio)
            "prophet_fin":                         (0.0,  1.0,  9.0),  # FIN ≈ 0 (aucun ACK-FIN)
            # ICMP et UDP absents
            "prophet_icmp":                        (0.0,  1.0,  9.0),  # ICMP normal
            "prophet_udp":                         (0.0,  1.0,  9.0),  # UDP normal
            # reconst_fin_from_syn : signal fort — syn↑↑↑ mais fin≈0 → résidu négatif massif
            # Ref Roesch 1999 Snort : fin/syn → 0 est signature SYN flood canonique
            # Ref Lippmann et al. 2000 MIT Lincoln Lab : ratio complétion TCP = feature top-3
            "reconst_fin_from_syn":                (30.0, 5.0,  0.0),  # fin << syn → FORT signal
            # tcp/packets : tcp ↑↑↑ + packets ↑ → ratio monte (tous TCP)
            "reconst_tcp_from_packets":            (20.0, 5.0,  0.5),  # tcp/pkt ↑↑
            # udp/flows : UDP normal, flows ↑ → ratio udp/flows ↓
            "reconst_udp_from_flows":              (0.0,  1.0,  9.0),  # udp/flow normal
        }
    },

    # ──────────────────────────────────────────────────────────────────────────
    # [NOUVELLE] BOTNET C&C BEACONING — connexions périodiques vers serveur C&C
    # Ref : Binsalleeh et al. (2010) IEEE SP&E — beaconing NetFlow features
    # Ref : Garcia et al. (2014) DIMVA — CTU-13 dataset (réseau académique)
    # Ref : MITRE ATT&CK T1071.001 — C&C via HTTP/HTTPS
    #
    # Caractéristiques :
    #   flows↑ modéré : beaconing = connexions TCP courtes et répétées
    #   bytes↓↓, packets↓ : heartbeat minimal (petits messages de présence)
    #   avg_pkt_size↓ : petits paquets (~100-500B) — signature "flat BPF"
    #   entropy_dst_port↓ : port C&C fixe (80, 443 ou autre)
    #   entropy_src_ip↓↓ : UN SEUL hôte compromis — signal SAFE fort
    #   syn↑, fin modéré : TCP beacons ouverts et fermés proprement
    #   tcp↑, icmp=0, udp=0
    #   fin/syn ≈ 1 : chaque beacon se ferme proprement (≠ Slowloris)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "BOTNET_CC_BEACONING",
        "type": "Attaque persistante — Botnet C&C beaconing",
        "start": "2025-11-19 08:00:00",
        "duration_h": 4.0,
        "intensity": "low",
        "ramp_frac": 0.20,
        "signature": {
            # ── Métriques existantes ──────────────────────────────────────────
            # bytes↓↓ : heartbeat = petits messages — Safe fort (≠ DDoS)
            # Ref : Binsalleeh 2010 — Bytes per Flow (BPF) "flat and low" = C&C signature
            "prophet_bytes": (1.0, 2.0, 8.0),  # bytes↓↓ (heartbeat minimal)
            "prophet_packets": (1.5, 3.0, 7.0),  # packets↓ (petits beacons)
            # flows↑ modéré : connexions répétées périodiquement
            # Garcia 2014 CTU-13 : nb_flows/minute élevé = feature primaire botnet
            "prophet_flows": (10.0, 5.0, 3.0),  # flows↑ modéré (beaconing)
            # syn↑ modéré : chaque beacon ouvre une connexion TCP
            "prophet_syn": (8.0, 5.0, 3.0),  # SYN↑ modéré (ouvertures)
            # Source unique (hôte compromis) → entropy src_ip très basse
            "prophet_entropy_src_ip": (0.5, 1.5, 9.0),  # source UNIQUE → NOR fort
            # Ports source variés par beacon (os port ephemeral allocation)
            "prophet_entropy_src_port": (5.0, 5.0, 4.0),  # ports source modérément variés
            # Port destination C&C FIXE → concentration anormale (port 80/443/autre)
            "prophet_entropy_dst_port": (8.0, 5.0, 2.0),  # port C&C fixe → ATK
            # avg_pkt_size↓ : petits paquets heartbeat ~100-500B
            # Hofstede 2014 : BPF "flat and low" s'applique aussi aux C&C
            "prophet_avg_pkt_size": (8.0, 5.0, 2.0),  # paquets heartbeat ↓
            # bytes/packets ratio légèrement perturbé (petits paquets uniformes)
            "reconst_bytes_from_packets": (2.0, 4.0, 6.0),  # ratio modérément normal
            "reconst_bytes_from_entropy_src_port": (4.0, 5.0, 3.0),  # signal modéré
            # ── Nouvelles métriques protocole ────────────────────────────────
            # tcp↑ : C&C HTTP/HTTPS = TCP (MITRE T1071.001)
            "prophet_tcp": (10.0, 5.0, 3.0),  # TCP↑ modéré (beaconing)
            "prophet_icmp": (0.0, 1.0, 9.0),  # ICMP normal
            "prophet_udp": (0.5, 2.0, 8.0),  # UDP faible (si DNS C&C)
            # FIN modéré : beacons se ferment proprement (TCP teardown normal)
            # ≠ Slowloris (fin≈0) ≠ SYN Flood (fin≈0)
            "prophet_fin": (6.0, 5.0, 3.0),  # FIN modéré (fermeture normale)
            # fin/syn ≈ 1 : chaque beacon TCP s'ouvre et se ferme → ratio ≈ 1
            "reconst_fin_from_syn": (3.0, 4.0, 5.0),  # fin/syn ≈ neutre
            # tcp/packets modéré (tout est TCP mais bytes faibles)
            "reconst_tcp_from_packets": (5.0, 4.0, 3.0),  # tcp/pkt modéré
            # udp/flows : flows↑, udp normal → ratio udp/flows ↓
            "reconst_udp_from_flows": (0.0, 1.0, 9.0),  # normal (TCP beaconing)
        }
    },

    # ──────────────────────────────────────────────────────────────────────────
    # 3. PORT SCAN AGRESSIF — reconnaissance TCP SYN
    # Ref : Mirsky et al. 2018 Kitsune ; MITRE ATT&CK T1046
    # Caractéristiques :
    #   flows↑↑↑ (nombreuses connexions vers ports variés)
    #   entropy_dst_port↑↑↑ (diversité ports destination)
    #   bytes↓ (SYN seuls, sans payload)
    #   syn↑↑ (probes SYN), fin modéré (RST des ports fermés, pas FIN)
    #   fin/syn < 1 (beaucoup d'envois SYN sans réponse FIN)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "AGGRESSIVE_PORT_SCAN",
        "type": "Reconnaissance — Port Scan agressif",
        "start": "2025-11-28 10:15:00",
        "duration_h": 2.5,
        "intensity": "medium",
        "ramp_frac": 0.05,
        "signature": {
            # ── Métriques existantes ──────────────────────────────────────────
            "prophet_bytes":                       (1.0,  4.0,  6.0),  # bytes bas (SYN sans payload)
            "prophet_packets":                     (4.0,  6.0,  3.0),
            "prophet_flows":                       (20.0, 5.0,  0.5),  # flows ↑↑↑ — signal primaire
            "prophet_syn":                         (15.0, 4.0,  0.5),
            "prophet_entropy_src_ip":              (1.0,  2.0,  9.0),  # source unique (attaquant fixe)
            "prophet_entropy_src_port":            (8.0,  4.0,  2.0),
            "prophet_entropy_dst_port":            (25.0, 3.0,  0.0),  # dst_port ↑↑↑ — signal primaire
            "prophet_avg_pkt_size":                (15.0, 4.0,  0.5),  # SYN ~60B → anormal
            "reconst_bytes_from_packets":          (2.0,  5.0,  4.0),
            "reconst_bytes_from_entropy_src_port": (2.0,  4.0,  4.0),
            # ── Nouvelles métriques protocole ────────────────────────────────
            # TCP↑ (probes SYN), pas UDP ni ICMP
            "prophet_tcp":                         (15.0, 5.0,  1.0),  # TCP ↑ (probes)
            "prophet_udp":                         (0.0,  1.0,  9.0),  # UDP normal
            "prophet_icmp":                        (0.0,  1.0,  9.0),  # ICMP normal
            # FIN modéré : les ports ouverts répondent SYN-ACK, puis FIN ; les fermés RST
            # Ref Mirsky 2018 : ratio complétion partielle caractérise port scan
            "prophet_fin":                         (3.0,  4.0,  5.0),  # FIN partiel (ports ouverts)
            # fin/syn < 1 (beaucoup de SYN sans FIN : ports fermés → RST)
            "reconst_fin_from_syn":                (8.0,  5.0,  2.0),  # fin << syn — signal modéré
            # tcp/packets↑ (tous les paquets sont TCP)
            "reconst_tcp_from_packets":            (12.0, 5.0,  1.0),  # tcp/pkt ↑ (tous TCP)
            # udp/flows normal
            "reconst_udp_from_flows":              (0.0,  1.0,  9.0),  # normal
        }
    },

    # ──────────────────────────────────────────────────────────────────────────
    # 4. EXFILTRATION LENTE — furtive (Data Exfiltration)
    # Ref : MITRE ATT&CK T1048 ; Jiang et al. 2018 DNS exfiltration
    # Caractéristiques :
    #   bytes↑ (uploads de gros blocs de données)
    #   avg_pkt_size↑↑ (données par blocs > MTU)
    #   flows↓ (peu de connexions, durées longues)
    #   syn normal (connexions légitimes, peu fréquentes)
    #   TCP uniquement (exfiltration TCP directe), icmp=0, udp=0
    #   fin/syn ≈ 1 (connexions TCP se ferment normalement)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "DATA_EXFILTRATION_SLOW",
        "type": "Exfiltration — Transfert lent et furtif",
        "start": "2025-12-02 23:00:00",
        "duration_h": 6.0,
        "intensity": "low",
        "ramp_frac": 0.20,
        "signature": {
            # ── Métriques existantes ──────────────────────────────────────────
            "prophet_bytes":                       (10.0, 5.0,  1.0),  # bytes↑ (uploads)
            "prophet_packets":                     (3.0,  5.0,  4.0),
            "prophet_flows":                       (0.5,  3.0,  8.0),  # flows↓ (furtif)
            "prophet_syn":                         (1.0,  2.0,  8.0),  # SYN normal
            "prophet_entropy_src_ip":              (0.5,  2.0,  9.0),  # source fixe (furtif)
            "prophet_entropy_src_port":            (0.5,  2.0,  9.0),
            "prophet_entropy_dst_port":            (0.5,  2.0,  9.0),  # destination fixe
            "prophet_avg_pkt_size":                (10.0, 5.0,  1.5),  # gros paquets
            "reconst_bytes_from_packets":          (4.0,  5.0,  2.0),
            "reconst_bytes_from_entropy_src_port": (18.0, 5.0,  0.5),
            # ── Nouvelles métriques protocole ────────────────────────────────
            # TCP uniquement (exfiltration TCP), icmp et udp normaux
            "prophet_tcp":                         (5.0,  4.0,  3.0),  # TCP modéré (connexion ouverte)
            "prophet_icmp":                        (0.0,  1.0,  9.0),  # ICMP normal
            "prophet_udp":                         (0.0,  1.0,  9.0),  # UDP normal
            # FIN normal : connexions exfiltration se ferment normalement
            "prophet_fin":                         (3.0,  4.0,  4.0),  # FIN modéré (connexions complètes)
            # fin/syn ≈ 1 : chaque connexion TCP ouverte se ferme proprement
            "reconst_fin_from_syn": (1.0, 2.0, 8.0),  # fin≈syn → résidu ≈ 0 → SAFE fort
            # FIX : DATA_EXFIL utilise des connexions TCP normales (fin≈syn → complétion normale).
            # Le résidu RANSAC doit être clairement Safe.
            # Ref : MITRE ATT&CK T1048 — exfiltration lente sur connexions TCP légitimes.
            # L'ambiguïté (3/4/4) donnait un signal faiblement atk → SLOWLORIS captait via fin_ratio.            # tcp/packets normal
            "reconst_tcp_from_packets":            (4.0,  4.0,  3.0),  # tcp/pkt normal
            # udp/flows normal
            "reconst_udp_from_flows":              (0.0,  1.0,  9.0),  # normal
        }
    },

    # ──────────────────────────────────────────────────────────────────────────
    # 5. HTTP FLOOD L7 — attaque applicative
    # Ref : CIC-IDS2017 DoS Hulk/GoldenEye Sharafaldin 2018
    # Caractéristiques :
    #   flows↑↑↑ (connexions HTTP complètes : GET/POST légitimes en masse)
    #   bytes↑↑, packets↑↑ (charge applicative complète)
    #   syn↑↑ ET fin↑↑ (connexions TCP qui s'ouvrent ET se ferment)
    #   fin/syn ≈ 1 : différence clé vs SYN Flood (connections complètes)
    #   tcp↑↑ (toutes les connexions sont TCP)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "HTTP_FLOOD_L7_DDOS",
        "type": "DDoS Applicatif L7 — HTTP Flood",
        "start": "2025-12-07 16:30:00",
        "duration_h": 1.5,
        "intensity": "high",
        "ramp_frac": 0.15,
        "signature": {
            # ── Métriques existantes ──────────────────────────────────────────
            "prophet_bytes":                       (15.0, 6.0,  0.5),
            "prophet_packets":                     (12.0, 5.0,  0.5),
            "prophet_flows":                       (18.0, 5.0,  0.5),  # flows ↑↑↑
            "prophet_syn":                         (12.0, 4.0,  1.0),  # SYN↑ (connexions)
            "prophet_entropy_src_ip":              (10.0, 4.0,  1.0),  # botnet distribué
            "prophet_entropy_src_port":            (8.0,  4.0,  2.0),
            "prophet_entropy_dst_port":            (1.0,  2.0,  8.0),  # port 80/443 fixe
            "prophet_avg_pkt_size":                (4.0,  6.0,  3.0),
            "reconst_bytes_from_packets":          (10.0, 5.0,  1.0),
            "reconst_bytes_from_entropy_src_port": (12.0, 5.0,  1.0),
            # ── Nouvelles métriques protocole ────────────────────────────────
            "prophet_tcp":                         (18.0, 5.0,  0.5),  # TCP ↑↑ (HTTP = TCP)
            "prophet_icmp":                        (0.0,  1.0,  9.0),  # ICMP normal
            "prophet_udp":                         (0.0,  1.0,  9.0),  # UDP normal
            # FIN↑↑ : les connexions HTTP se ferment après chaque requête (HTTP/1.0) ou
            # à l'issue du flood (HTTP/1.1 keepalive limité)
            # Ref CIC-IDS2017 DoS Hulk : FIN count élevé (connexions complètes)
            "prophet_fin":                         (12.0, 5.0,  1.0),  # FIN↑↑ (connexions HTTP complètes)
            # fin/syn ≈ 1 — distingue HTTP Flood de SYN Flood et Slowloris (fin≈0)
            "reconst_fin_from_syn": (1.0, 3.0, 6.0),  # fin≈syn → résidu ≈ 0 → SAFE
            # FIX : HTTP_FLOOD a des connexions TCP complètes (fin≈syn). Le résidu RANSAC
            # fin_from_syn doit être ≈0 (Normal), pas anomalous.
            # Ref : Sharafaldin 2018 CIC-IDS2017 — HTTP flood : connexions complètes, fin≈syn.
            # L'erreur (6/4/2) donnait ev_attack dominant → SLOWLORIS captait via fin_ratio.            # tcp/packets ↑↑ (tout est TCP)
            "reconst_tcp_from_packets":            (15.0, 5.0,  0.5),  # tcp/pkt ↑↑
            # udp/flows normal
            "reconst_udp_from_flows":              (0.0,  1.0,  9.0),  # normal
        }
    },

    # ──────────────────────────────────────────────────────────────────────────
    # 6. DNS AMPLIFICATION — réflexion ×50-70
    # Ref : IETF RFC 7534 (DNS Response Rate Limiting) ; Cloudflare 2024 Q3
    # Caractéristiques :
    #   bytes↑↑↑ (réponses DNS amplifiées jusqu'à 4 KB par requête ~40B → ×100)
    #   avg_pkt_size↑↑↑ (réponses massives)
    #   entropy_src_ip↑↑ (réflecteurs variés)
    #   udp↑ (DNS utilise UDP) — signal modéré
    #   flows normaux (réponses arrivent par les mêmes flux que les requêtes légit)
    #   syn=0, fin=0 (DNS sur UDP → pas de TCP)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "DNS_AMPLIFICATION",
        "type": "DDoS Réflexion/Amplification — DNS",
        "start": "2025-12-11 08:00:00",
        "duration_h": 3.0,
        "intensity": "extreme",
        "ramp_frac": 0.08,
        "signature": {
            # ── Métriques existantes ──────────────────────────────────────────
            "prophet_bytes":                       (35.0, 8.0,  0.0),  # bytes ↑↑↑ (amplification)
            "prophet_packets":                     (15.0, 6.0,  0.5),
            "prophet_flows":                       (4.0,  6.0,  3.0),  # flows normaux
            "prophet_syn":                         (0.0,  1.0,  9.0),  # pas de SYN (UDP)
            "prophet_entropy_src_ip":              (18.0, 4.0,  0.5),  # réflecteurs variés
            "prophet_entropy_src_port":            (3.0,  4.0,  5.0),  # port 53 DNS (fixe→faible entropie)
            "prophet_entropy_dst_port":            (1.0,  2.0,  8.0),  # port destination fixe
            "prophet_avg_pkt_size":                (22.0, 5.0,  0.0),  # paquets 3000+ bytes ↑↑↑
            "reconst_bytes_from_packets":          (25.0, 6.0,  0.5),  # bytes/packets ratio cassé
            "reconst_bytes_from_entropy_src_port": (15.0, 5.0,  1.0),
            # ── Nouvelles métriques protocole ────────────────────────────────
            # DNS sur UDP → udp↑ (mais plus modéré car réflecteurs déjà dans le trafic normal)
            "prophet_udp":                         (18.0, 6.0,  0.5),  # UDP↑ (réponses DNS)
            "prophet_tcp":                         (0.0,  1.0,  9.0),  # TCP normal (DNS=UDP)
            "prophet_icmp":                        (0.0,  1.0,  9.0),  # ICMP normal
            # FIN absent (DNS sur UDP, pas de TCP)
            "prophet_fin":                         (0.0,  1.0,  9.0),  # FIN ≈ 0 (UDP)
            # fin/syn : aucun TCP → résidu RANSAC ≈ 0 → neutre
            "reconst_fin_from_syn":                (0.5,  1.0,  9.0),  # neutre
            # tcp/packets : tcp normal, packets↑ (DNS) → ratio tcp/packets ↓
            "reconst_tcp_from_packets":            (10.0, 4.0,  1.0),  # tcp/pkt ↓ (UDP domine)
            # udp/flows : UDP↑ mais flows normaux → ratio monte
            "reconst_udp_from_flows":              (15.0, 5.0,  1.0),  # udp/flow ↑
        }
    },

    # ──────────────────────────────────────────────────────────────────────────
    # 7. SLOWLORIS — épuisement de connexions TCP
    # Ref : Hansen (2009) Slowloris ; CIC-IDS2017 DoS Slowloris
    # Caractéristiques :
    #   flows↑↑ (connexions TCP maintenues ouvertes indéfiniment)
    #   bytes↓↓ (seuls les headers HTTP partiels sont envoyés)
    #   avg_pkt_size minimal (headers ~200 bytes seulement)
    #   syn↑ (ouvertures), fin≈0 (pas de fermeture — c'est le but de l'attaque)
    #   fin/syn → 0 : signature analogue à SYN Flood mais différente en volume
    #   Signal clé différenciateur vs SYN Flood : bytes↓ (Slowloris) vs bytes↑ (SYN)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "SLOWLORIS_DOS",
        "type": "DoS Lent — Slowloris",
        "start": "2025-12-15 22:00:00",
        "duration_h": 8.0,
        "intensity": "low",
        "ramp_frac": 0.25,
        "signature": {
            # ── Métriques existantes ──────────────────────────────────────────
            "prophet_bytes":                       (0.3,  2.0,  8.0),  # bytes↓↓ (seuls headers)
            "prophet_packets":                     (0.3,  2.0,  8.0),  # packets↓
            "prophet_flows":                       (12.0, 5.0,  0.5),  # flows↑↑ (connexions ouvertes)
            "prophet_syn":                         (8.0,  4.0,  1.5),  # SYN↑ (ouvertures)
            # Entropie source : Slowloris opère depuis une source UNIQUE vers un serveur unique
            # → entropy_src_ip = Safe (source fixe, un seul attaquant)
            # → entropy_src_port = Safe (ports sources séquentiels et prévisibles)
            # Ref : Hansen (2009) Slowloris — attaque monosource par conception
            "prophet_entropy_src_ip":              (0.0,  1.0,  9.0),  # source unique → NOR
            "prophet_entropy_src_port":            (0.0,  1.0,  9.0),  # ports séquentiels → NOR
            "prophet_entropy_dst_port":            (0.5,  2.0,  8.0),  # port 80 fixe
            "prophet_avg_pkt_size":                (8.0,  5.0,  1.0),  # paquets ~200B
            "reconst_bytes_from_packets":          (0.5,  3.0,  7.0),
            "reconst_bytes_from_entropy_src_port": (1.0,  3.0,  6.0),
            # ── Nouvelles métriques protocole ────────────────────────────────
            # TCP↑ (connexions persistantes ouvertes)
            "prophet_tcp":                         (12.0, 5.0,  1.0),  # TCP↑ (connexions ouvertes)
            "prophet_icmp":                        (0.0,  1.0,  9.0),  # ICMP normal
            "prophet_udp":                         (0.0,  1.0,  9.0),  # UDP normal
            # FIN ≈ 0 : Slowloris NE FERME JAMAIS ses connexions (c'est l'objectif)
            # Signal identique à SYN Flood sur fin/syn, mais volume différent
            "prophet_fin":                         (0.0,  0.5,  9.5),  # FIN ≈ 0 (connexions jamais fermées)
            # fin/syn → 0 : signature Slowloris (Roesch 1999 ; Hansen 2009)
            # syn↑ mais fin≈0 → résidu RANSAC fort négatif
            "reconst_fin_from_syn":                (25.0, 5.0,  0.0),  # fin << syn — signal fort
            # tcp/packets : tcp↑, packets↓ → ratio tcp/packets monte (tous paquets = TCP)
            "reconst_tcp_from_packets":            (10.0, 4.0,  1.0),  # tcp/pkt ↑
            # udp/flows : flows↑↑, udp normal → ratio udp/flows ↓
            "reconst_udp_from_flows":              (0.0,  1.0,  9.0),  # normal
        }
    },

    # ──────────────────────────────────────────────────────────────────────────
    # 8. ICMP FLOOD BURST — burst ultra-court et brutal
    # Ref : Moustafa & Slay (2015) UNSW-NB15 ; RFC 792 ICMP
    # Caractéristiques :
    #   icmp↑↑↑ — SIGNAL PRIMAIRE discriminateur vs UDP_FLOOD
    #   packets↑↑↑, bytes↑↑ (paquets ICMP echo à ~1400 bytes = MTU padding)
    #   avg_pkt_size↑ (MTU-sized ICMP requests)
    #   flows = 0 (ICMP est sans état, pas de notion de flux TCP/UDP)
    #   syn = 0, fin = 0 (ICMP ne passe pas par TCP)
    #   tcp = 0 (aucun trafic TCP pendant ICMP flood)
    #   udp ≈ normal — SIGNAL ACTIF Safe (différenciateur vs UDP_FLOOD)
    #   reconst_tcp_from_packets : tcp≈0 mais packets↑↑ → ratio tcp/packets ↓↓ → signal fort
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "ICMP_FLOOD_BURST",
        "type": "DoS Volumétrique — ICMP Ping Flood",
        "start": "2025-12-18 11:30:00",
        "duration_h": 0.5,
        "intensity": "extreme",
        "ramp_frac": 0.05,
        "signature": {
            # ── Métriques existantes ──────────────────────────────────────────
            "prophet_bytes":                       (22.0, 5.0,  0.0),
            "prophet_packets":                     (40.0, 5.0,  0.0),  # packets ↑↑↑
            "prophet_flows":                       (0.0,  1.0,  9.0),  # flows = 0 (ICMP stateless)
            "prophet_syn":                         (0.0,  0.5,  9.5),  # SYN = 0 (pas TCP)
            "prophet_entropy_src_ip":              (4.0,  5.0,  3.0),
            "prophet_entropy_src_port":            (1.0,  2.0,  8.0),  # ICMP pas de port source
            "prophet_entropy_dst_port":            (1.0,  2.0,  8.0),
            "prophet_avg_pkt_size":                (15.0, 4.0,  0.5),  # paquets MTU-size (~1400B)
            "reconst_bytes_from_packets":          (18.0, 5.0,  0.5),
            "reconst_bytes_from_entropy_src_port": (10.0, 5.0,  1.0),
            # ── Nouvelles métriques protocole ────────────────────────────────
            # ICMP ↑↑↑ — DISCRIMINATEUR PRINCIPAL vs UDP_FLOOD
            # Ref Moustafa & Slay 2015 UNSW-NB15 : icmp_count est feature top-5 pour DoS
            # RedeRio normal : icmp mean=652, p99=1651 ; ICMP flood extrême : ×20-50 normal
            "prophet_icmp":                        (40.0, 5.0,  0.0),  # ICMP ↑↑↑ — DISCRIMINATEUR CLÉ
            # UDP ≈ normal — signal Safe ACTIF : enseigne explicitement "pas UDP flood"
            "prophet_udp":                         (0.0,  1.0,  9.0),  # UDP normal — anti-UDP_FLOOD
            # TCP ≈ 0 : ICMP flood ne génère aucun trafic TCP
            "prophet_tcp":                         (0.0,  1.0,  9.0),  # TCP absent
            # FIN = 0 : pas de connexions TCP
            "prophet_fin":                         (0.0,  0.5,  9.5),  # FIN = 0 (pas TCP)
            # fin/syn : syn=0 et fin=0 → résidu RANSAC ≈ 0 (modèle prédit ~0, observé ~0)
            "reconst_fin_from_syn":                (1.0,  2.0,  8.0),  # résidu neutre (0≈0)
            # tcp/packets : tcp=0, packets↑↑↑ → résidu tcp_from_packets massif négatif
            # Tous les paquets sont ICMP donc tcp/packets → 0 → résidu fort
            "reconst_tcp_from_packets":            (18.0, 4.0,  0.5),  # tcp/pkt ↓↓ (paquets ICMP)
            # udp/flows : udp normal, flows=0 → résidu udp_from_flows légèrement anormal
            "reconst_udp_from_flows":              (5.0,  3.0,  3.0),  # udp/flow légèrement anormal
        }
    },
    # ──────────────────────────────────────────────────────────────────────────
    # 10. NTP AMPLIFICATION — réflexion/amplification BAF ×556
    # Ref : van Rijswijk-Deij et al. (2014) IMC — BAF NTP = 556.9 vs DNS 28-54
    # Ref : Czyz et al. (2014) IMC — "Taming the 800 Pound Gorilla : NTP DDoS"
    # Ref : Cisco CVE-2013-5211 — monlist MON_GETLIST : réponse 5 500× la requête
    #
    # Mécanisme : requête MONLIST spoofée (8 bytes) → réponse 600 IPs×72B = ~43 KB
    # en plusieurs paquets UDP. Attaquant spoofie IP victime vers serveurs NTP ouverts.
    #
    # Signature NetFlow (différences clés vs DNS_AMP) :
    #   bytes↑↑↑↑ — encore plus fort que DNS_AMP (BAF 556 vs 54 → ×10)
    #   avg_pkt_size↑↑↑↑ — paquets monlist très lourds (600 entrées × 72B/paquet)
    #   udp↑↑↑ — pur UDP/123 (plus fort que DNS car réponses plus volumineuses)
    #   entropy_src_ip↑↑ — de nombreux serveurs NTP vulnérables comme réflecteurs
    #   entropy_src_port↓↓ — port source 123 fixe chez les réflecteurs (≡ DNS port 53)
    #   entropy_dst_port↓ — victime reçoit tout sur un port fixe
    #   syn=0, fin=0, tcp=0 — pur UDP, aucun handshake TCP
    #   flows modéré — chaque réflecteur ouvre sa propre session UDP
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "NTP_AMPLIFICATION",
        "type": "DDoS Réflexion/Amplification — NTP (BAF×556)",
        "start": "2025-12-04 08:00:00",
        "duration_h": 3.0,
        "intensity": "extreme",
        "ramp_frac": 0.08,
        "signature": {
            # ── Métriques existantes ──────────────────────────────────────────
            # BAF 556 >> DNS BAF 54 → bytes encore plus amplifiés
            "prophet_bytes": (50.0, 7.0, 0.0),  # bytes ↑↑↑↑ (BAF×556)
            "prophet_packets": (20.0, 6.0, 0.5),  # paquets ↑↑ (chaque monlist = N paquets)
            "prophet_flows": (5.0, 7.0, 3.0),  # flows modérés (réflecteurs UDP)
            "prophet_syn": (0.0, 1.0, 9.0),  # SYN = 0 (UDP pur)
            # Nombreux serveurs NTP vulnérables = beaucoup de sources distinctes
            "prophet_entropy_src_ip": (20.0, 4.0, 0.5),  # réflecteurs variés ↑↑
            # Port source fixe 123 chez tous les réflecteurs NTP → entropie basse
            # (analogue à DNS port 53 mais encore plus concentré)
            "prophet_entropy_src_port": (1.0, 2.0, 8.0),  # port 123 fixe → NOR
            "prophet_entropy_dst_port": (1.0, 2.0, 8.0),  # destination fixe
            # avg_pkt_size ↑↑↑↑ : monlist = 600 adresses IP en réponse
            # Réponse NTP monlist : jusqu'à ~48 paquets de ~500 bytes chacun (> DNS ~4KB)
            # Ref Cisco CVE-2013-5211 : réponse 5 500× la requête de 8 bytes
            "prophet_avg_pkt_size": (30.0, 5.0, 0.0),  # paquets NTP ↑↑↑↑ (monlist lourd)
            # bytes/packets ratio encore plus cassé que DNS_AMP (BAF supérieur)
            "reconst_bytes_from_packets": (35.0, 6.0, 0.5),  # ratio bytes/pkt ↑↑↑↑
            "reconst_bytes_from_entropy_src_port": (20.0, 5.0, 0.5),  # reconstruction cassée
            # ── Nouvelles métriques protocole ────────────────────────────────
            # UDP ↑↑↑ : encore plus fort que DNS_AMP car réponses NTP plus volumineuses
            "prophet_udp": (25.0, 6.0, 0.5),  # UDP ↑↑↑ (NTP/UDP pur)
            # TCP, ICMP, SYN, FIN = 0 (aucune connexion TCP)
            "prophet_tcp": (0.0, 1.0, 9.0),  # TCP normal (UDP pur)
            "prophet_icmp": (0.0, 1.0, 9.0),  # ICMP normal
            "prophet_fin": (0.0, 1.0, 9.0),  # FIN = 0 (UDP)
            "reconst_fin_from_syn": (0.5, 1.0, 9.0),  # neutre (pas de TCP)
            # tcp/packets : tcp≈0, packets ↑↑ → ratio tcp/pkt ↓ (UDP domine)
            "reconst_tcp_from_packets": (12.0, 4.0, 1.0),  # tcp/pkt ↓↓ (UDP noie TCP)
            # udp/flows : UDP ↑↑↑, flows modérés → ratio explose encore plus que DNS_AMP
            "reconst_udp_from_flows": (20.0, 5.0, 0.5),  # udp/flow ↑↑↑
        }
    },

    # ──────────────────────────────────────────────────────────────────────────
    # 11. BRUTE_FORCE_SSH — attaque par dictionnaire sur SSH (port 22)
    # Ref : Najafabadi et al. (2015) ICMLA — SSH BF detection via NetFlow agrégé ;
    #        features-clés : nb flows, bytes, packets sur fenêtres de 5 min
    # Ref : Hynek et al. (2020) IFIP SEC 35 — ML sur IP flows étendus, campus réseau
    # Ref : Hofstede et al. (2014) ACM SIGCOMM CCR — SSHCure : trafic "plat"
    #        PPF/BPF/duration constants pendant la phase BF (boucle de tentatives)
    #
    # Caractéristiques NetFlow (réseau académique, attaque distribuée type botnet) :
    #   flows↑↑↑ — SIGNAL PRIMAIRE : de nombreuses connexions TCP courtes vers port 22
    #   syn↑↑ — chaque tentative ouvre une connexion TCP (SYN)
    #   fin modéré — tentatives échouées → RST (pas FIN) ; succès → FIN
    #                → fin/syn < 1 (beaucoup de SYN sans FIN correspondant)
    #   tcp↑↑ — 100% TCP
    #   bytes modéré-bas — petites connexions (quelques centaines de bytes/tentative)
    #   avg_pkt_size↓ — handshake SSH ~100–300 bytes (Hofstede 2014 : BPF faible)
    #   entropy_dst_port↓ — port 22 fixe (signal discriminant)
    #   entropy_src_ip modéré — attaque botnet distribuée (≠ source unique)
    #   entropy_src_port élevé — ports source variés par l'attaquant
    #   icmp=0, udp=0 — pur TCP
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "BRUTE_FORCE_SSH",
        "type": "Credential Attack — SSH Brute Force (distributed)",
        "start": "2025-11-25 14:00:00",
        "duration_h": 3.0,
        "intensity": "medium",
        "ramp_frac": 0.10,
        "signature": {
            # ── Métriques existantes ──────────────────────────────────────────
            # bytes modéré : nombreuses petites connexions SSH (~300 bytes/tentative)
            # N tentatives × 300 bytes = volume total modéré sur backbone
            "prophet_bytes": (4.0, 6.0, 3.0),  # bytes modéré (petits flux × nombreux)
            # packets modéré : chaque tentative = SYN + SYN-ACK + AUTH + RST/FIN ~6-8 paquets
            "prophet_packets": (6.0, 6.0, 2.0),  # packets modéré
            # flows↑↑↑ : SIGNAL PRIMAIRE — centaines de connexions TCP/22 par fenêtre
            # Ref Najafabadi 2015 : nb de flows = feature discriminante n°1 pour SSH BF
            "prophet_flows": (20.0, 5.0, 0.5),  # flows ↑↑↑ — signal primaire
            # syn↑↑ : chaque tentative ouvre un nouveau TCP SYN (pas de réutilisation de connexion)
            "prophet_syn": (15.0, 4.0, 1.0),  # SYN ↑↑ (ouvertures multiples)
            # Attaque distribuée botnet → plusieurs sources → entropie src_ip modérée
            "prophet_entropy_src_ip": (8.0, 5.0, 2.0),  # botnet distribué (modéré)
            # Ports source variés par les bots → entropie src_port ↑
            "prophet_entropy_src_port": (10.0, 4.0, 1.0),  # ports source variés ↑
            # Port destination 22 fixe → entropie dst_port ↓ — signal discriminant clé
            "prophet_entropy_dst_port": (12.0, 4.0, 0.5),  # port 22 fixe ↓ — signal fort
            # avg_pkt_size↓ : Hofstede 2014 — BPF faible, paquets SSH handshake ~200B
            "prophet_avg_pkt_size": (10.0, 5.0, 1.0),  # paquets SSH ~200B → ↓
            # bytes/packets ratio perturbé (petits paquets, nombreux)
            "reconst_bytes_from_packets": (3.0, 5.0, 4.0),  # ratio modérément perturbé
            "reconst_bytes_from_entropy_src_port": (5.0, 5.0, 3.0),  # signal modéré
            # ── Nouvelles métriques protocole ────────────────────────────────
            # tcp↑↑ : 100% TCP — signal secondaire fort
            "prophet_tcp": (18.0, 5.0, 0.5),  # TCP ↑↑ (100% TCP)
            # icmp=0, udp=0 : pur TCP/22, pas d'ICMP ni UDP
            "prophet_icmp": (0.0, 1.0, 9.0),  # ICMP normal
            "prophet_udp": (0.0, 1.0, 9.0),  # UDP normal
            # FIN modéré : les tentatives échouées se terminent par RST (sans FIN)
            # Les rares connexions réussies ont un FIN propre
            # → prophet_fin légèrement ↑ mais inférieur à syn
            "prophet_fin": (5.0, 5.0, 3.0),  # FIN modéré (RST > FIN sur échecs)
            # fin/syn < 1 : beaucoup de SYN sans FIN correspondant (RST sur échecs)
            # Ref Hofstede 2014 : ratio complétion TCP < 1 caractérise la phase BF
            # Signal modéré (moins fort que SYN Flood ou Slowloris)
            "reconst_fin_from_syn": (8.0, 5.0, 2.0),  # fin < syn — signal modéré
            # tcp/packets↑ : tous les paquets sont TCP → ratio normal/élevé
            "reconst_tcp_from_packets": (10.0, 4.0, 2.0),  # tcp/pkt ↑ (tout TCP)
            # udp/flows : UDP=0, flows↑↑ → ratio udp/flows ↓
            "reconst_udp_from_flows": (0.0, 1.0, 9.0),  # udp/flow ↓ (flows↑, udp=0)
        }
    },

    # ──────────────────────────────────────────────────────────────────────────
    # 12. DNS TUNNELING — exfiltration covert via DNS (MITRE ATT&CK T1048.001)
    # Ref : Sharma et al. (2018) Procedia CS 132 — labeled DNS flow dataset,
    #        features : flows, avg_pkt_size, entropy src_port
    # Ref : Habibi et al. (2019) IEEE IM — entropie hostname + longueur query
    #        détection sur 10 Gbps en temps réel depuis backbone académique
    # Ref : MDPI Electronics (2023) 12(6):1467 — taille normale DNS 50–550 bytes ;
    #        DNS tunneling génère des paquets hors de cette plage (noms encodés)
    # Ref : GIAC (2015) — labels jusqu'à 63 chars, noms totaux jusqu'à 255 chars
    #
    # Mécanisme : hôte compromis encapsule des données dans des noms de sous-domaines
    # encodés (base32/base64) envoyés vers un serveur DNS contrôlé par l'attaquant.
    # Exemples d'outils : iodine, dns2tcp, dnscat2, DNSExfiltrator.
    #
    # Signature NetFlow (différences clés vs DNS_AMP et DATA_EXFILTRATION) :
    #   bytes↑ modéré — exfiltration (pas d'amplification) ; plus faible que DNS_AMP
    #   udp↑ modéré — DNS/UDP/53 pour les petites requêtes
    #   tcp↑ modéré — DNS/TCP/53 pour les grandes requêtes (>512 bytes)
    #                  → SYN + FIN présents (connexions TCP propres)
    #   avg_pkt_size↑ — noms encodés plus longs que DNS normal (50–550B normal)
    #   entropy_src_port↑↑ — ports source variés par client (≠ DNS_AMP port 123 fixe)
    #   entropy_src_ip↓↓ — UN SEUL hôte compromis (≠ DNS_AMP : nombreux réflecteurs)
    #                       SIGNAL DISCRIMINANT CLÉ vs DNS_AMP
    #   entropy_dst_port↓ — port 53 fixe (comme DNS_AMP, mais même logique)
    #   flows modéré-↑ — sessions DNS persistantes (tunneling continu)
    #   syn modéré, fin modéré, fin/syn ≈ 1 — TCP/53 se ferme proprement
    #                   (≠ Slowloris : fin≈0 ; ≠ SYN Flood : fin<<syn)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "DNS_TUNNELING",
        "type": "Exfiltration Covert — DNS Tunneling (T1048.001)",
        "start": "2025-12-09 10:00:00",
        "duration_h": 6.0,
        "intensity": "low",
        "ramp_frac": 0.15,
        "signature": {
            # ── Métriques existantes ──────────────────────────────────────────
            # bytes↑ modéré : exfiltration de données encodées, pas d'amplification
            # (beaucoup moins que DNS_AMP ; similaire à DATA_EXFIL mais via DNS)
            "prophet_bytes": (8.0, 6.0, 2.0),  # bytes↑ modéré (exfiltration)
            # packets modéré : nombreuses requêtes DNS encodées
            "prophet_packets": (5.0, 5.0, 3.0),  # packets modéré
            # flows↑ modéré : sessions DNS persistantes (hôte compromis → serveur attaquant)
            # Ref Sharma 2018 : avg flows/min anormalement élevé en DNS tunneling
            "prophet_flows": (8.0, 5.0, 2.0),  # flows↑ modéré (sessions persistantes)
            # SYN modéré : TCP/53 utilisé pour les grandes requêtes DNS (>512B)
            "prophet_syn": (4.0, 4.0, 4.0),  # SYN modéré (TCP/53 parfois)
            # entropy_src_ip↓↓ : UN SEUL hôte compromis — SIGNAL DISCRIMINANT vs DNS_AMP
            # C'est la principale différence avec DNS_AMP (multi-réflecteurs)
            "prophet_entropy_src_ip": (0.5, 2.0, 9.0),  # source UNIQUE — NOR (hôte compromis)
            # entropy_src_port↑↑ : le client utilise des ports source variés
            # Ref Sharma 2018 Fig.3b : entropie port source anormale en DNS tunneling
            "prophet_entropy_src_port": (12.0, 4.0, 1.0),  # ports source variés ↑↑
            # entropy_dst_port↓ : port 53 fixe (comme DNS_AMP)
            "prophet_entropy_dst_port": (0.5, 1.5, 8.0),  # port 53 fixe → NOR
            # avg_pkt_size↑ : noms de domaine encodés plus longs que DNS normal
            # Ref MDPI 2023 : DNS normal 50–550B ; tunneling dépasse souvent 550B
            # Ref GIAC 2015 : labels jusqu'à 63 chars → paquets plus lourds
            "prophet_avg_pkt_size": (8.0, 6.0, 2.0),  # requêtes encodées ↑ (> DNS normal)
            # bytes/packets : ratio légèrement perturbé (paquets lourds vs DNS normal)
            "reconst_bytes_from_packets": (6.0, 5.0, 2.0),  # ratio modérément perturbé
            # bytes vs entropy_src_port : bytes↑ + entropy_src_port↑↑ → signal fort
            "reconst_bytes_from_entropy_src_port": (10.0, 5.0, 1.0),  # reconstr. perturbée ↑↑
            # ── Nouvelles métriques protocole ────────────────────────────────
            # UDP↑ modéré : DNS tunneling principalement UDP/53 pour petites requêtes
            "prophet_udp": (10.0, 5.0, 2.0),  # UDP↑ modéré (DNS/UDP/53)
            # TCP↑ modéré : DNS/TCP/53 pour les grandes requêtes (>512B encodées)
            # Ref MDPI 2023 : "DNS queries > 512 bytes use TCP"
            "prophet_tcp": (8.0, 5.0, 2.0),  # TCP↑ modéré (DNS/TCP/53)
            "prophet_icmp": (0.0, 1.0, 9.0),  # ICMP normal
            # FIN modéré : TCP/53 se ferme proprement (connexions complètes)
            # Contrairement à Slowloris (fin≈0) ou SYN Flood (fin<<syn)
            "prophet_fin": (6.0, 5.0, 3.0),  # FIN modéré (TCP/53 propre)
            # fin/syn ≈ 1 : chaque TCP/53 ouvert est proprement fermé
            # Signal différenciateur vs Slowloris et SYN Flood
            "reconst_fin_from_syn": (3.0, 4.0, 5.0),  # fin/syn ≈ 1 (neutre-modéré)
            # tcp/packets↑ : tcp modéré, packets↑ (UDP + TCP) → ratio stable
            "reconst_tcp_from_packets": (4.0, 4.0, 4.0),  # tcp/pkt stable (UDP + TCP)
            # udp/flows : UDP↑, flows↑ → ratio modéré
            "reconst_udp_from_flows": (6.0, 5.0, 3.0),  # udp/flow↑ modéré
        }
    },
]

# PATCH-C1 fix (2026-04-19) — Guard post-definition, vérification de bijection
# par NOM (pas par ordre de déclaration — l'ordre n'est pas sémantiquement
# significatif). Les timestamps doivent coïncider pour chaque nom.
#
# Ancienne version (buggy) :
#   - comparait les listes par ordre (assert ==) → faux-positif si les deux
#     catalogues contiennent les mêmes attaques dans un ordre différent.
#   - utilisait la clé 'start_time' alors que l'injecteur déclare 'start'.
_local_names    = {a['name'] for a in ATTACK_CATALOG}
_canonical_set  = set(_CANONICAL_NAMES)
_missing_in_inj = _canonical_set - _local_names
_missing_in_cfg = _local_names - _canonical_set
assert not _missing_in_inj and not _missing_in_cfg, (
    f"Divergence catalogue (bijection de noms) :\n"
    f"  manquant dans injecteur : {_missing_in_inj}\n"
    f"  manquant dans config    : {_missing_in_cfg}"
)
_cfg_by_name = {a['name']: a for a in _CANONICAL_ORDER}
for a_inj in ATTACK_CATALOG:
    a_cfg = _cfg_by_name[a_inj['name']]
    assert a_inj['start'] == a_cfg['start'], (
        f"Timestamp divergent pour {a_inj['name']} : "
        f"injecteur={a_inj['start']} vs config={a_cfg['start']}"
    )

# ==============================================================================
# RÉSOLUTION DU CATALOGUE DEPUIS CONFIG (cross-dataset)
# ==============================================================================
_catalog_override = CONFIG.get('ATTACK_CATALOG', None)
if _catalog_override is not None:
    # Autre dataset : soit catalogue custom, soit injection désactivée
    ATTACK_CATALOG = _catalog_override
# Si None : on garde le catalogue brésilien ci-dessus (comportement par défaut)


# ==============================================================================
# HELPERS SCIENTIFIQUES
# ==============================================================================

def _normalize_signature(ev_attack: float, ev_suspect: float, ev_normal: float,
                          window_size: int) -> tuple:
    """
    Normalise les poids bruts d'une signature pour que ev_attack + ev_suspect + ev_normal
    = window_size, maintenant l'invariant de la bijection SL (Jøsang 2016, Def. 3.9).

    Les valeurs du catalogue sont des poids relatifs guidés par la littérature.
    La normalisation préserve les proportions et garantit que les opinions injectées
    ont la même plage de certitude (u = W/(W + window_size)) qu'une fenêtre naturelle.

    Cas dégénéré : si la somme est nulle, retourne la fenêtre entièrement Safe
    (aucune évidence d'attaque → comportement conservatif).
    """
    total = ev_attack + ev_suspect + ev_normal
    if total <= 0:
        return (0.0, 0.0, float(window_size))
    scale = window_size / total
    return (ev_attack * scale, ev_suspect * scale, ev_normal * scale)


def _check_no_overlap(attacks: list) -> None:
    """
    Vérifie qu'aucune paire d'attaques ne se chevauche temporellement.
    Un recouvrement produirait une signature hybride non documentée (la seconde
    injection écrasant silencieusement la première).
    Lève ValueError si un recouvrement est détecté.
    """
    intervals = []
    for a in attacks:
        s = pd.Timestamp(a['start'])
        e = s + pd.Timedelta(hours=a['duration_h'])
        for (s2, e2, name2) in intervals:
            if s < e2 and e > s2:
                raise ValueError(
                    f"Recouvrement temporel entre '{a['name']}' ({s}→{e}) "
                    f"et '{name2}' ({s2}→{e2}). Corriger le catalogue."
                )
        intervals.append((s, e, a['name']))


_REQUIRED_CATALOG_KEYS = {'name', 'type', 'start', 'duration_h', 'ramp_frac', 'signature'}

def _validate_catalog(attacks: list) -> None:
    """Valide le schéma et les valeurs de chaque entrée du catalogue."""
    names_seen = set()
    split_date = pd.Timestamp(CONFIG['split_date'])
    for atk in attacks:
        missing = _REQUIRED_CATALOG_KEYS - set(atk.keys())
        if missing:
            raise ValueError(f"Entrée catalogue incomplète : {missing} "
                             f"manquants dans '{atk.get('name', '?')}'")
        if atk['name'] in names_seen:
            raise ValueError(f"Nom d'attaque dupliqué : '{atk['name']}'")
        names_seen.add(atk['name'])
        # A3.5 enforcement (PATCH 2026-05-06) — every catalog event must
        # start strictly after split_date so the calibration of δ on
        # df_train_calib cannot see the injected windows. Without this
        # check, a misplaced catalog entry silently contaminates the
        # decision threshold.
        t_start = pd.Timestamp(atk['start'])
        if t_start <= split_date:
            raise ValueError(
                f"[A3.5] Catalog event '{atk['name']}' starts at {t_start} "
                f"which is on or before split_date={split_date}. The threshold "
                f"calibrator ({CONFIG.get('VERSION_NAME', 'train_models.py')}) "
                f"would see this attack as 'normal' and bias the decision "
                f"threshold. Move the event into the test span or update "
                f"CONFIG['split_date']."
            )
        for metric_key, (ev_a, ev_s, ev_n) in atk['signature'].items():
            if ev_a < 0 or ev_s < 0 or ev_n < 0:
                raise ValueError(
                    f"[{atk['name']}][{metric_key}] valeur de signature négative : "
                    f"({ev_a}, {ev_s}, {ev_n}). Les preuves doivent être ≥ 0."
                )


# ==============================================================================
# RAMP PROFILE
# ==============================================================================

def make_ramp(n_windows, ramp_frac):
    """
    Profil d'intensité : montée linéaire → plateau → descente symétrique.
    Modélise le comportement réaliste d'une attaque (démarrage progressif).
    Ref : profil trapézoïdal utilisé dans CIC-IDS2017 pour les injections synthétiques.
    """
    ramp_n = max(1, int(n_windows * ramp_frac))
    profile = np.ones(n_windows, dtype=float)
    for i in range(ramp_n):
        profile[i] = (i + 1) / ramp_n
    for i in range(ramp_n):
        idx = n_windows - 1 - i
        profile[idx] = min(profile[idx], (i + 1) / ramp_n)
    return profile


# ==============================================================================
# INJECTION
# ==============================================================================

def inject_attack_into_evidence(df, attack, available_cols, verbose=True):
    """
    Injecte une attaque dans le DataFrame d'evidence.
    Écrase les colonnes _P, _S, _N de chaque métrique concernée.
    compute_opinions recalculera tout le reste (ageing, bijection, WBF, CBF).

    CONVENTION d'inversion P↔N :
        Dans le pipeline compute_evidence, la convention est :
            col_P (suffix _P) = evidence Safe  (positive pour l'état normal)
            col_N (suffix _N) = evidence Attack (négative pour l'état normal)
        Dans les signatures du catalogue (ev_attack, ev_suspect, ev_normal) :
            ev_attack → stocké dans col_N
            ev_normal → stocké dans col_P
        L'inversion est donc intentionnelle et correcte.

    NORMALISATION (SC1) :
        Les poids bruts du catalogue sont normalisés de sorte que
        ev_attack + ev_suspect + ev_normal = WINDOW_SIZE avant injection.
        Cela maintient l'invariant P+S+N = WINDOW_SIZE (Jøsang 2016, Def. 3.9) :
        les opinions injectées ont la même plage de certitude que les fenêtres
        naturelles produites par compute_evidence_v2.

    PROFIL DE RAMP — interpolation linéaire (SC2) :
        Pour chaque fenêtre d'attaque, le triplet injecté est :
            P = (1 − alpha) × WINDOW_SIZE + alpha × ev_normal_norm
            S =  alpha × ev_suspect_norm
            N =  alpha × ev_attack_norm
        où alpha ∈ [0, 1] est donné par make_ramp().
        Propriété : P + S + N = WINDOW_SIZE pour tout alpha (invariant maintenu).
        Sémantique : alpha=0 → fenêtre entièrement normale (certitude Safe) ;
                     alpha=1 → signature d'attaque à pleine intensité.

    INJECTION_SKIP_N_DOMINANT (config.py = False) :
        Quand False : toutes les métriques sont injectées, y compris les métriques
        N-dominantes (ev_normal > ev_attack). Cela permet d'injecter un signal Safe
        actif (ex: icmp=Safe pendant UDP_FLOOD) pour aider la qualification downstream.
    """
    start_dt = pd.Timestamp(attack['start'])
    end_dt   = start_dt + pd.Timedelta(hours=attack['duration_h'])

    mask = (df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)
    idxs = df.index[mask].tolist()

    if not idxs:
        print(f"  ⚠️  [{attack['name']}] Aucune fenêtre entre {start_dt} et {end_dt}")
        return 0

    profile = make_ramp(len(idxs), attack['ramp_frac'])
    injected_metrics = 0
    skipped_n_dominant = 0

    for metric_key, (ev_attack_raw, ev_suspect_raw, ev_normal_raw) in attack['signature'].items():
        # Utiliser clean_key pour compatibilité défensive avec compute_evidence_v2
        # (qui écrit f"{clean_key}_P" avec clean_key = key.replace("->", "_to_"))
        clean_metric_key = metric_key.replace("->", "_to_")
        col_P = f"{clean_metric_key}_P"
        col_S = f"{clean_metric_key}_S"
        col_N = f"{clean_metric_key}_N"

        if col_P not in available_cols:
            continue  # métrique absente du CSV — normale si non dans ACTIVE_METRICS

        # Optionnellement, ignorer les métriques où le signal normal domine
        # CONFIG['INJECTION_SKIP_N_DOMINANT'] = False → toujours injecter (comportement actuel)
        if CONFIG.get("INJECTION_SKIP_N_DOMINANT", False):
            if ev_normal_raw >= ev_attack_raw and ev_normal_raw >= ev_suspect_raw:
                skipped_n_dominant += 1
                continue

        # Normalisation : maintien de l'invariant P+S+N = WINDOW_SIZE (Def. 3.9)
        ev_attack_norm, ev_suspect_norm, ev_normal_norm = _normalize_signature(
            ev_attack_raw, ev_suspect_raw, ev_normal_raw, WINDOW_SIZE
        )

        for df_idx, alpha in zip(idxs, profile):
            # Interpolation linéaire : état normal (alpha=0) → attaque (alpha=1).
            # P+S+N = (1−α)·W + α·(ev_n+ev_s+ev_a) = (1−α)·W + α·W = W  ✓
            df.at[df_idx, col_P] = (1.0 - alpha) * WINDOW_SIZE + alpha * ev_normal_norm
            df.at[df_idx, col_S] = alpha * ev_suspect_norm
            df.at[df_idx, col_N] = alpha * ev_attack_norm

        injected_metrics += 1

    if verbose:
        missing = len(attack['signature']) - injected_metrics - skipped_n_dominant
        status = ""
        if skipped_n_dominant > 0:
            status += f" | {skipped_n_dominant} N-dom. skipped"
        if missing > 0:
            status += f" ⚠️  {missing} COLONNES MANQUANTES"
        print(f"  ✅ [{attack['name']:<28}] {len(idxs)} fenêtres | "
              f"{injected_metrics}/{len(attack['signature'])} métriques injectées{status}")
        # Vérification de l'invariant P+S+N sur la fenêtre centrale
        if idxs and injected_metrics > 0:
            sample_key = next(
                (k.replace("->", "_to_") for k in attack['signature']
                 if f"{k.replace('->', '_to_')}_N" in available_cols), None
            )
            if sample_key:
                mid_idx = idxs[len(idxs) // 2]
                p_v = df.at[mid_idx, f"{sample_key}_P"]
                s_v = df.at[mid_idx, f"{sample_key}_S"]
                n_v = df.at[mid_idx, f"{sample_key}_N"]
                print(f"     → [{sample_key}] fenêtre centrale : "
                      f"P={p_v:.2f}, S={s_v:.2f}, N={n_v:.2f}, "
                      f"Σ={p_v+s_v+n_v:.2f} (attendu={WINDOW_SIZE})")

    return len(idxs)


# ==============================================================================
# GÉNÉRATION DU FICHIER TEXTE DE CALENDRIER
# ==============================================================================

def generate_attack_schedule_txt(attacks):
    from datetime import datetime
    _n_prophet = len(CONFIG.get('ACTIVE_METRICS', []))
    _n_reconst = len(CONFIG.get('RECONST_RULES', []))
    _dataset   = CONFIG.get('ACTIVE_DATASET', 'RedeRio')
    lines = []
    lines.append("=" * 80)
    lines.append("ATTACK_SCHEDULE.txt — Calendrier des attaques synthétiques (niveau evidence)")
    lines.append("=" * 80)
    lines.append(f"Généré le  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Dataset    : {_dataset}")
    lines.append(f"Pipeline   : inject_at_evidence_level.py v3 → compute_opinions_v3.py")
    lines.append(f"Nb attaques: {len(attacks)}")
    lines.append(f"Métriques  : {_n_prophet + _n_reconst} ({_n_prophet} Prophet + {_n_reconst} RANSAC)")
    lines.append("")
    lines.append("IMPORTANT : injection AU NIVEAU DES PREUVES BRUTES (evidence CSV).")
    lines.append("compute_opinions applique ensuite ageing + bijection + WBF + CBF.")
    lines.append("")

    intensity_bars = {
        "low": "⬛⬜⬜⬜", "medium": "⬛⬛⬜⬜",
        "high": "⬛⬛⬛⬜", "extreme": "⬛⬛⬛⬛"
    }

    lines.append("-" * 80)
    lines.append(f"{'#':<4} {'Nom':<30} {'Début':>19}  {'Durée':>6}  {'Intensité'}")
    lines.append("-" * 80)
    for i, atk in enumerate(attacks, 1):
        s = pd.Timestamp(atk['start'])
        lines.append(f"{i:<4} {atk['name']:<30} {str(s)[:19]}  "
                     f"{atk['duration_h']:>5.1f}h  "
                     f"{intensity_bars.get(atk['intensity'], '')} {atk['intensity']}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("DÉTAIL PAR ATTAQUE")
    lines.append("=" * 80)

    for i, atk in enumerate(attacks, 1):
        s = pd.Timestamp(atk['start'])
        e = s + pd.Timedelta(hours=atk['duration_h'])
        lines.append("")
        lines.append(f"┌─ [{i:02d}] {atk['name']}")
        lines.append(f"│  Type    : {atk['type']}")
        lines.append(f"│  Période : {str(s)[:19]}  →  {str(e)[:19]}")
        lines.append(f"│  Durée   : {atk['duration_h']}h  |  Intensité : {atk['intensity']}  "
                     f"|  Montée : {int(atk['ramp_frac']*100)}%")
        lines.append(f"│  Métriques injectées (ev_attack / ev_suspect / ev_normal) :")
        for m, (Pa, Ps, Pn) in atk['signature'].items():
            dom = "→ ATK" if Pa > Ps and Pa > Pn else "→ SUS" if Ps > Pa and Ps > Pn else "→ NOR"
            lines.append(f"│    {m:<50}  att={Pa:5.1f}  sus={Ps:4.1f}  nor={Pn:4.1f}  {dom}")
        lines.append(f"└{'─'*78}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("RÉFÉRENCES")
    lines.append("=" * 80)
    lines.append("  Sharafaldin et al. (2018) — CIC-IDS2017, ICISSP")
    lines.append("  Mirsky et al. (2018) — Kitsune, NDSS")
    lines.append("  Moustafa & Slay (2015) — UNSW-NB15, MilCIS")
    lines.append("  Hansen R. (2009) — Slowloris HTTP DoS tool")
    lines.append("  Roesch M. (1999) — Snort, USENIX LISA")
    lines.append("  Lippmann et al. (2000) — MIT Lincoln Lab DARPA evaluation")
    lines.append("  MITRE ATT&CK T1046 (Scan), T1048 (Exfil), T1498/T1499 (DDoS)")
    lines.append("  Cloudflare DDoS Threat Report Q3 2024")
    lines.append("  RFC 792 — ICMP")
    lines.append("  van Rijswijk-Deij et al. (2014) APNIC/IMC — BAF NTP=556.9 vs DNS=28-54")
    lines.append("  Czyz et al. (2014) IMC — Taming the 800 Pound Gorilla: NTP DDoS")
    lines.append("  Cisco CVE-2013-5211 — NTP monlist : réponse 5500x la requête initiale")
    lines.append("  Najafabadi et al. (2015) ICMLA — SSH BF via NetFlow agrégé 5min")
    lines.append("  Hynek et al. (2020) IFIP SEC 35 — ML sur IP flows étendus, campus")
    lines.append("  Hofstede et al. (2014) ACM SIGCOMM CCR — SSHCure : trafic BF plat")
    lines.append("  Sharma et al. (2018) Procedia CS 132 — labeled DNS flow dataset")
    lines.append("  Habibi et al. (2019) IEEE IM — DNS tunneling detection backbone 10Gbps")
    lines.append("  MDPI Electronics (2023) 12(6):1467 — DNS normal 50-550B ; tunnel hors-norme")
    return "\n".join(lines)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    _n_active = len(CONFIG.get('ACTIVE_METRICS', [])) + len(CONFIG.get('RECONST_RULES', []))
    print(f"\n{'='*65}")
    print(f"  INJECTION EVIDENCE-LEVEL — v3 ({_n_active} métriques)")
    print(f"{'='*65}\n")

    # Vérification de cohérence du nommage (CP1) :
    # compute_opinions_v3 cherche exactement f"evidence_{VERSION_NAME}_attacks.csv".
    _expected_by_opinions = f"{VERSION_NAME}_attacks"
    if VERSION_NAME_MODIF != _expected_by_opinions:
        print(f"⚠️  VERSION_NAME_MODIF='{VERSION_NAME_MODIF}' ≠ '{_expected_by_opinions}' "
              f"attendu par compute_opinions_v3. Vérifier config.py.\n")

    # ── Gestion injection désactivée (ATTACK_CATALOG = [] dans config) ────────
    if ATTACK_CATALOG == []:
        print(f"  -> ATTACK_CATALOG vide pour le dataset '{CONFIG.get('ACTIVE_DATASET', '')}' "
              f"— injection désactivée.")
        print(f"  -> Copie de {INPUT_EVIDENCE_CSV} → {OUTPUT_EVIDENCE_CSV}")
        if not os.path.exists(INPUT_EVIDENCE_CSV):
            print(f"❌ Fichier introuvable : {INPUT_EVIDENCE_CSV}"); return
        os.makedirs(os.path.dirname(OUTPUT_EVIDENCE_CSV), exist_ok=True)
        shutil.copy2(INPUT_EVIDENCE_CSV, OUTPUT_EVIDENCE_CSV)
        print("✅ Copie terminée (pas d'injection).")
        return

    if not os.path.exists(INPUT_EVIDENCE_CSV):
        print(f"❌ Fichier introuvable : {INPUT_EVIDENCE_CSV}")
        print(f"   Vérifier que compute_evidence_v2.py a bien tourné.")
        return

    os.makedirs(os.path.dirname(OUTPUT_EVIDENCE_CSV), exist_ok=True)

    print(f"→ Chargement : {INPUT_EVIDENCE_CSV}")
    df = pd.read_csv(INPUT_EVIDENCE_CSV, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"   {len(df)} fenêtres | {df['timestamp'].min()} → {df['timestamp'].max()}")

    available_cols = set(df.columns)
    evidence_cols  = [c for c in df.columns if c.endswith(('_P', '_S', '_N'))]
    _n_metrics_csv = len(evidence_cols) // 3
    print(f"   Colonnes evidence : {len(evidence_cols)} ({_n_metrics_csv} métriques × 3)")

    # Validation du catalogue avant toute modification du DataFrame
    _validate_catalog(ATTACK_CATALOG)

    # Vérification des nouvelles métriques
    new_metrics = ['prophet_icmp', 'prophet_udp', 'prophet_tcp', 'prophet_fin',
                   'reconst_udp_from_flows']
    missing_new = [m for m in new_metrics if f"{m}_N" not in available_cols]
    if missing_new:
        print(f"\n   ⚠️  MÉTRIQUES MANQUANTES : {missing_new}")
        print(f"   → Vérifier que compute_evidence_v2.py a bien traité le CSV evidence")
    else:
        print(f"   ✅ Toutes les métriques actives présentes dans le CSV")

    # Vérification plages temporelles
    csv_start = df['timestamp'].min()
    csv_end   = df['timestamp'].max()
    print(f"\n→ Vérification des plages temporelles :")
    valid_attacks = []
    for atk in ATTACK_CATALOG:
        s = pd.Timestamp(atk['start'])
        e = s + pd.Timedelta(hours=atk['duration_h'])
        ok = (s >= csv_start) and (e <= csv_end)
        print(f"   {'✅' if ok else '⚠️  HORS PLAGE'} "
              f"{atk['name']:<32} {str(s)[:19]} ({atk['duration_h']}h)")
        if ok:
            valid_attacks.append(atk)

    if not valid_attacks:
        print("\n❌ Aucune attaque dans la plage du CSV.")
        return

    # Vérification d'absence de recouvrement temporel (SC3)
    _check_no_overlap(valid_attacks)

    # Initialisation de la colonne de vérité terrain (SC4)
    df['injection_label'] = 'normal'
    df['injection_ramp_alpha'] = 0.0

    # Injection
    print(f"\n→ Injection de {len(valid_attacks)} attaques "
          f"(normalisation P+S+N={WINDOW_SIZE}, interpolation normal↔attaque)...")
    total_windows = 0
    for atk in valid_attacks:
        n_win = inject_attack_into_evidence(df, atk, available_cols)
        total_windows += n_win

        # Vérité terrain : marquer les fenêtres et stocker le profil alpha (SC4)
        s = pd.Timestamp(atk['start'])
        e = s + pd.Timedelta(hours=atk['duration_h'])
        mask = (df['timestamp'] >= s) & (df['timestamp'] < e)
        idxs_gt = df.index[mask].tolist()
        profile_gt = make_ramp(len(idxs_gt), atk['ramp_frac'])
        df.loc[mask, 'injection_label'] = atk['name']
        for df_idx, alpha in zip(idxs_gt, profile_gt):
            df.at[df_idx, 'injection_ramp_alpha'] = alpha

    # Sauvegarde
    df.to_csv(OUTPUT_EVIDENCE_CSV, index=False)
    print(f"\n✅ Evidence injectée : {OUTPUT_EVIDENCE_CSV}")
    _n_attack_win = (df['injection_label'] != 'normal').sum()
    print(f"   {_n_attack_win} fenêtres d'attaque sur {len(df)} "
          f"({100*_n_attack_win/len(df):.1f}%)")
    print(f"   Colonnes ajoutées : injection_label, injection_ramp_alpha")

    # Génération ATTACK_SCHEDULE.txt
    txt = generate_attack_schedule_txt(valid_attacks)
    txt_path = os.path.join(os.path.dirname(OUTPUT_EVIDENCE_CSV), "ATTACK_SCHEDULE.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(txt)
    print(f"✅ Calendrier         : {txt_path}")

    print(f"\n→ Prochaine étape :")
    print(f"   Dans compute_opinions_v3.py, vérifier VERSION_NAME = '{VERSION_NAME_MODIF}'")
    print(f"   puis lancer compute_opinions_v3.py normalement.\n")


if __name__ == "__main__":
    main()