"""
qualify_anomaly_sbn.py - SL template qualifier (legacy SBN filename)
====================================================================
Backward-compatible replacement for the legacy qualifier.

Scientific terminology note
---------------------------
This module is not a canonical Subjective Bayesian Network in Joesang's
strict sense. The historical module name and output columns keep ``sbn``
for compatibility, but the paper should call it an expert-template-driven
Subjective Logic qualifier, or SL-template qualifier (SL-TQ).

A strict SBN would require an explicit DAG, conditional subjective
opinions with uncertainty for each edge/state, and conditional SL
deduction/abduction propagation. This code does not implement that.

Architecture effective du qualificateur SL-TQ :
────────────────────────────────────────────────────
  L1 : Métriques brutes  → opinions ternaires {Safe, Susp, Anom} dans le CSV
  L2 : Groupes sémantiques → P^{groupe}_s = geomean_{m∈g}(P^m_s)      [depuis CSV]
  L3 : Table experte P(G=s|type_k) = SBN_COND_OPINIONS                [config.py]
       Score(k,g) = Σ_s P^{obs}_{g,s} · c^{k|g}_s                     [dot product]
  L4 : Fusion inter-groupes  → geomean des log-scores + bijection SL   [Jøsang Def. 3.9]
  L5 : Prior temporel        → T (kill chain) → WBF avec opinion t-1 escomptée
                               (prior Markovien interprétable, non SBN strict)
  L6 : Uncertainty Maximisation → signal de nouveauté amplifié          [Jøsang Eq. 3.27]

Différences fondamentales vs qualify_anomaly.py :
──────────────────────────────────────────────────
  1. Opinions conditionnelles MOLLES (distributions P) au lieu d'états binaires
     → confiance variable par groupe (u_cond encodé dans la distribution)
  2. Score = dot-product projecté (non LR brut) → plus stable numériquement
  3. Prior temporel Markovien (WBF + décroissance λ^Δt) pour persistance d'attaque
  4. Matrice de transition entre types (kill chain recon→exfil, persistance)
  5. NETWORK_OUTAGE classifié nativement par SBN_COND_OPINIONS (sans gate bypass)

Fondements théoriques et limites :
───────────────────────
  Jøsang (2016) §3.5.2      : bijection évidence → opinion
  Jøsang (2016) §12         : WBF / CBF
  Jøsang (2016) §3.6 Eq 3.27 : Uncertainty Maximisation
  Neyman & Pearson (1933)    : LR comme statistique suffisante
  Chow (1970) IEEE TIT       : reject option / classe résiduelle
  Sharafaldin et al. (2018) CIC-IDS2017 : signatures UDP/SYN/HTTP
  Mirsky et al. (2018) Kitsune           : Slowloris, Port Scan
  Moustafa & Slay (2015) UNSW-NB15      : ICMP discriminateur
  Rossow (2014) NDSS                     : DNS amplification
  MITRE ATT&CK T1046/T1048/T1498/T1499  : vocabulaire d'attaque

Usage :
───────
  python qualify_anomaly_sbn.py
  python qualify_anomaly_sbn.py --input detection_results_INJECTED.csv
  python qualify_anomaly_sbn.py --temporal             # active le prior temporel
  python qualify_anomaly_sbn.py --no_um                # désactive UM post-fusion
  python qualify_anomaly_sbn.py --compare              # compare avec qualify_anomaly.py
"""

from __future__ import annotations   # type hints Python 3.9-compatible (PEP 563)

import argparse
import math
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Imports projet
# ──────────────────────────────────────────────────────────────────────────────
try:
    from sl_ads.config import CONFIG, QUALIFY_GROUP_SOURCES as _GS_MODULE  # Phase H
    from sl_ads.paths import get_version_names, get_results_dir, get_decision_threshold, get_detection_col  # Phase H
    _THRESHOLD   = get_decision_threshold(CONFIG, up_levels=1)
    _DET_COL     = get_detection_col(CONFIG, up_levels=1)
    _OUTPUT_DIR  = get_results_dir(CONFIG, up_levels=1)
    GROUP_SOURCES = CONFIG.get('QUALIFY_GROUP_SOURCES', _GS_MODULE)
    SBN_PARAMS    = CONFIG.get('SBN_COND_OPINIONS', {})
except ImportError as e:
    print(f"[WARN] config.py/paths.py non trouvés — fallback. ({e})")
    _THRESHOLD  = 0.15
    _DET_COL    = 'FINAL_SYSTEM_CBF_proj_atk'
    _OUTPUT_DIR = '.'
    GROUP_SOURCES = {}
    SBN_PARAMS    = {}

# ──────────────────────────────────────────────────────────────────────────────
# SBN_COND_OPINIONS : opinions conditionnelles P(G=s | type_k)
# ──────────────────────────────────────────────────────────────────────────────
# Chaque valeur = distribution de probabilité sur {Safe, Susp, Anom} (somme=1).
# Encode la certitude sur la relation : plus piqué → plus discriminant.
#
# Convention de signaux (par analogie avec l'encodage ATTACK_PRIORS) :
#   STRONG Anom  : {'S':0.03, 'M':0.07, 'A':0.90}   ud≈0.10
#   MOD    Anom  : {'S':0.08, 'M':0.22, 'A':0.70}   ud≈0.20
#   WEAK   Anom  : {'S':0.15, 'M':0.35, 'A':0.50}   ud≈0.35
#   STRONG Susp  : {'S':0.10, 'M':0.80, 'A':0.10}
#   MOD    Susp  : {'S':0.20, 'M':0.55, 'A':0.25}
#   STRONG Safe  : {'S':0.85, 'M':0.12, 'A':0.03}
#   MOD    Safe  : {'S':0.70, 'M':0.20, 'A':0.10}
#   NON-DISCRIM  : {'S':0.33, 'M':0.34, 'A':0.33}   → score neutre ≈ 1/3
#
# Sources littérature par type d'attaque :
#   Sharafaldin 2018 (CIC-IDS2017), Mirsky 2018 (Kitsune), Moustafa 2015 (UNSW-NB15),
#   Rossow 2014 (NDSS DNS-AMP), MITRE ATT&CK, Cloudflare DDoS Report Q3 2024
#   van Rijswijk-Deij (2014) APNIC/IMC          : NTP amplification BAF=556.9
#   Najafabadi et al. (2015) ICMLA              : SSH BF via NetFlow agrégé
#   Hynek et al. (2020) IFIP SEC 35             : SSH BF ML campus réseau
#   Hofstede et al. (2014) ACM SIGCOMM CCR      : SSHCure flat traffic
#   Sharma et al. (2018) Procedia CS 132        : DNS tunneling flow features
#   Habibi et al. (2019) IEEE IM                : DNS tunnel detection 10Gbps
#   MDPI Electronics (2023) 12(6):1467         : DNS normal 50-550B
# ──────────────────────────────────────────────────────────────────────────────

_S = 'Safe'
_M = 'Susp'
_A = 'Anom'

# Conventions abrégées pour lisibilité
def _strong_anom():   return {_S: 0.03, _M: 0.07, _A: 0.90}
def _mod_anom():      return {_S: 0.08, _M: 0.22, _A: 0.70}
def _weak_anom():     return {_S: 0.15, _M: 0.35, _A: 0.50}
def _strong_susp():   return {_S: 0.10, _M: 0.80, _A: 0.10}
def _mod_susp():      return {_S: 0.20, _M: 0.55, _A: 0.25}
def _weak_susp():     return {_S: 0.30, _M: 0.45, _A: 0.25}
def _strong_safe():   return {_S: 0.85, _M: 0.12, _A: 0.03}
def _mod_safe():      return {_S: 0.70, _M: 0.20, _A: 0.10}
def _neutral():       return {_S: 0.33, _M: 0.34, _A: 0.33}

# Fallback si SBN_COND_OPINIONS absent de config.py
# Architecture 10 groupes : volume, protocol_tcp, protocol_udp, protocol_icmp,
#   tcp_flags (prophet_syn + prophet_fin uniquement),
#   fin_ratio (reconst_fin_from_syn — groupe dédié, sans dilution geomean),
#   connections, entropy, packet_size, reconstruction.
# IMPORTANT : tcp_flags ne contient plus reconst_fin_from_syn → le signal
# R_fin_from_syn est porté EXCLUSIVEMENT par 'fin_ratio'.
# Ceci élimine simultanément le double-comptage (BUG-01) et la dilution geomean.
_DEFAULT_SBN_COND = {

    # ── UDP FLOOD ──────────────────────────────────────────────────────────────
    # Signature : explosion UDP (volume+udp), TCP absent, paquets petits (~64-128B)
    # Ref : Sharafaldin et al. 2018 CIC-IDS2017 Table III ; Mirsky 2018 Kitsune
    'UDP_FLOOD': {
        'volume':        _strong_anom(),    # bytes+packets ↑↑↑ — signal primaire
        'protocol_udp':  _strong_anom(),    # UDP ↑↑↑ — discriminateur vs ICMP
        'protocol_tcp':  _strong_safe(),    # TCP absent
        'protocol_icmp': _strong_safe(),    # ICMP normal — discriminateur vs ICMP_FLOOD
        'tcp_flags':     _strong_safe(),    # syn≈0, fin≈0 (pas de TCP)
        'fin_ratio':     _strong_safe(),    # pas de TCP → ratio fin/syn indéfini → SAFE
        'connections':   _mod_susp(),       # flows légèrement anormaux
        'entropy':       _mod_susp(),       # src_port ↑ (spoofing)
        'packet_size':   _mod_safe(),       # paquets UDP ~64-128B (petits)
        'reconstruction': _mod_anom(),      # relation bytes~packets cassée
        'volume_surplus': _strong_anom(),   # bytes/packets/flows ↑↑↑ = surplus positif
        'volume_deficit': _strong_safe(),   # pas de chute
    },

    # ── SYN FLOOD ─────────────────────────────────────────────────────────────
    # syn ↑↑↑, fin≈0 → R_fin_from_syn très élevé (demi-connexions jamais fermées)
    # Ref : MITRE ATT&CK T1498.001 ; Roesch 1999 Snort
    'SYN_FLOOD': {
        'volume':        _weak_susp(),      # volume modéré (SYN sans payload)
        'protocol_tcp':  _strong_anom(),    # TCP ↑↑ (tous SYN)
        'protocol_udp':  _strong_safe(),
        'protocol_icmp': _strong_safe(),
        'tcp_flags':     _strong_anom(),    # syn ↑↑↑ (signal brut fort — prophet_syn)
        'fin_ratio':     _strong_anom(),    # R_fin_from_syn → 0 (demi-conn. non fermées) ATK
        'connections':   _mod_susp(),       # flows élevés (demi-connexions)
        'entropy':       _mod_susp(),       # src_ip aléatoire (spoofing)
        'packet_size':   _strong_anom(),    # paquets SYN petits (~60B uniformes)
        'reconstruction': _weak_susp(),
        'volume_surplus': _weak_susp(),     # volume modéré (SYN sans payload)
        'volume_deficit': _strong_safe(),
    },

    # ── ICMP FLOOD ────────────────────────────────────────────────────────────
    # icmp ↑↑↑, gros paquets (padding ~1400B), syn=0, fin=0
    # Discriminateur primaire vs UDP_FLOOD : protocol_icmp + packet_size
    # Ref : Moustafa & Slay 2015 UNSW-NB15 ; Mirsky 2018 Kitsune
    'ICMP_FLOOD': {
        'volume':        _strong_anom(),    # bytes+packets élevés
        'protocol_udp':  _strong_safe(),    # UDP absent — discriminateur vs UDP_FLOOD
        'protocol_tcp':  _strong_safe(),    # TCP absent
        'protocol_icmp': _strong_anom(),    # ICMP ↑↑↑ — discriminateur PRIMAIRE
        'tcp_flags':     _strong_safe(),    # syn=0, fin=0 (pas de TCP)
        'fin_ratio':     _strong_safe(),    # pas de TCP → ratio SAFE
        'connections':   _weak_anom(),      # flows≈0 (ICMP sans état)
        'entropy':       _strong_safe(),    # paquets identiques → entropie faible
        'packet_size':   _strong_anom(),    # paquets MTU ~1400B — DISCRIMINATEUR
        'reconstruction': _mod_anom(),
        'volume_surplus': _strong_anom(),
        'volume_deficit': _strong_safe(),
    },

    # ── DNS AMPLIFICATION ─────────────────────────────────────────────────────
    # Réponses UDP gonflées (×28-556), gros paquets, peu de sources (réflecteurs)
    # Ref : Rossow 2014 NDSS Table 2 ; Cloudflare DDoS Report Q3 2024
    'DNS_AMP': {
        'volume':        _strong_anom(),    # bytes ↑↑↑ (réponses jusqu'à 4KB)
        'protocol_udp':  _mod_susp(),       # udp ↑ (réponses DNS port 53)
        'protocol_tcp':  _strong_safe(),    # DNS principalement UDP
        'protocol_icmp': _strong_safe(),
        'tcp_flags':     _strong_safe(),    # pas de SYN/FIN (UDP)
        'fin_ratio':     _strong_safe(),    # UDP → pas de TCP → ratio SAFE
        'connections':   _mod_safe(),       # flows normaux (peu de requêtes)
        'entropy':       _mod_susp(),       # src_ip ↑ (réflecteurs variés) / src_port fixe (53)
        'packet_size':   _strong_anom(),    # très grands paquets (3000+ bytes)
        'reconstruction': _mod_anom(),
        'volume_surplus': _strong_anom(),
        'volume_deficit': _strong_safe(),
    },

    # ── HTTP FLOOD (L7) ───────────────────────────────────────────────────────
    # Connexions TCP COMPLÈTES (fin≈syn), volume+flows↑↑↑
    # Ref : Sharafaldin 2018 CIC-IDS2017 ; Loránd Lékó 2020 USENIX
    'HTTP_FLOOD': {
        'volume':        _strong_anom(),
        'protocol_tcp':  _strong_anom(),    # TCP ↑↑ (conn. complètes)
        'protocol_udp':  _strong_safe(),
        'protocol_icmp': _strong_safe(),
        'tcp_flags':     _mod_susp(),       # syn↑ ET fin↑ (conn. complètes ouvertes ET fermées)
        'fin_ratio':     _mod_safe(),       # fin≈syn → ratio ≈ 1 → fermeture normale → SAFE
        'connections':   _strong_anom(),    # flows ↑↑↑
        'entropy':       _mod_susp(),       # src_ip variable
        'packet_size':   _weak_susp(),      # paquets HTTP variables
        'reconstruction': _mod_anom(),
        'volume_surplus': _strong_anom(),
        'volume_deficit': _strong_safe(),
    },

    # ── SLOWLORIS ─────────────────────────────────────────────────────────────
    # Connexions TCP ouvertes, jamais fermées → fin≈0, R_fin_from_syn très élevé.
    # Volume et paquets FAIBLES (headers partiels seulement).
    # Mécanisme de détection : tcp_flags est NEUTRE (syn↑ mais fin↓ → signal mixte
    # sur prophet_syn / prophet_fin). Le discriminant est porté par fin_ratio seul
    # (reconst_fin_from_syn → 0 = ATK FORT) dans son groupe dédié.
    # Ref : Mirsky 2018 Kitsune ; Hansen 2009 (Slowloris original)
    'SLOWLORIS': {
        'volume':        _strong_safe(),    # bytes≈0 (headers partiels)
        'protocol_tcp':  _mod_susp(),       # tcp ↑ (connexions maintenues)
        'protocol_udp':  _strong_safe(),
        'protocol_icmp': _strong_safe(),
        'tcp_flags':     _neutral(),        # syn↑ MAIS fin≈0 → signal MIXTE (prophet_syn vs prophet_fin)
                                            # ← NEUTRE intentionnel : discriminant délégué à fin_ratio
        'fin_ratio':     _strong_anom(),    # R_fin_from_syn → 0 = ATK FORT — DISCRIMINATEUR CLÉ
                                            # ← vs DATA_EXFIL : fin_ratio=SAFE (fin≈syn)
        'connections':   _strong_anom(),    # flows ↑↑ (connexions ouvertes longtemps)
        'entropy':       _strong_safe(),    # source unique → entropie faible
        'packet_size':   _neutral(),        # variable selon impl. — non-discriminant
        'reconstruction': _mod_anom(),
        'volume_surplus': _strong_safe(),   # bytes surplus ≈ 0
        'volume_deficit': _strong_safe(),   # pas de chute structurelle
    },

    # ── PORT SCAN ─────────────────────────────────────────────────────────────
    # Scan massif de ports → flows ↑↑↑, entropy_dst_port ↑↑↑, bytes faibles
    # Ref : MITRE ATT&CK T1046 ; Nmap signatures (Gordon 2006)
    'PORT_SCAN': {
        'volume':        _strong_safe(),    # bytes faibles (SYN seuls)
        'protocol_tcp':  _mod_susp(),       # tcp ↑ (SYN + RST des ports fermés)
        'protocol_udp':  _strong_safe(),
        'protocol_icmp': _strong_safe(),
        'tcp_flags':     _weak_susp(),      # syn ↑, fin modéré (RST fermés, prophet_fin bas)
        'fin_ratio':     _mod_susp(),       # RST ≠ FIN → ratio perturbé → SUSP
        'connections':   _strong_anom(),    # flows ↑↑↑ (scan massif)
        'entropy':       _strong_anom(),    # entropy_dst_port ↑↑↑ — DISCRIMINATEUR
        'packet_size':   _strong_anom(),    # paquets très petits (SYN seuls ~60B)
        'reconstruction': _weak_susp(),
        'volume_surplus': _strong_safe(),   # bytes surplus ≈ 0
        'volume_deficit': _strong_safe(),   # pas de chute structurelle
    },

    # ── DATA EXFILTRATION LENTE ───────────────────────────────────────────────
    # Connexions TCP complètes fermées normalement (fin≈syn → R_fin_from_syn=NOR)
    # Gros paquets (données), canal inhabituel.
    # Ref : MITRE ATT&CK T1048 ; Srivastava 2016 IEEE Big Data
    #
    # Discriminateurs CLÉS vs Autre_Anomalie (justification renforcement) :
    #   fin_ratio=strong_safe  : fermetures TCP normales → signature spécifique DATA_EXFIL
    #   entropy=strong_safe    : source unique ciblée → entropie DST faible (≠ scan)
    #   connections=strong_anom: nombreuses connexions TCP successives (transfert long)
    #   packet_size=strong_anom: grands paquets (payload de données, ≠ SYN seuls)
    # Ces quatre groupes forment un vecteur de croyances suffisamment distinct pour que
    # le score DATA_EXFIL surpasse Autre_Anomalie même en injection basse intensité.
    # Validation attendue : QP DATA_EXFIL > 0 après ce renforcement (précédemment 0 %).
    'DATA_EXFIL': {
        'volume':        _mod_susp(),       # bytes ↑ mais modéré (lent)
        'protocol_tcp':  _mod_anom(),       # tcp ↑↑ (transfert TCP soutenu) — RENFORCÉ
        'protocol_udp':  _strong_safe(),
        'protocol_icmp': _strong_safe(),
        'tcp_flags':     _strong_safe(),    # syn=NOR, fin=NOR (fermetures normales)
        'fin_ratio':     _strong_safe(),    # fin≈syn → ratio ≈ 1 = NOR FORT — DISCRIMINATEUR CLÉ
                                            # ← vs SLOWLORIS : fin_ratio=ATK (fin≈0)
        'connections':   _strong_anom(),    # flows ↑↑ (nombreuses conn. successives) — RENFORCÉ
        'entropy':       _strong_safe(),    # entropie DST faible (exfil ciblée ≠ scan) — RENFORCÉ
        'packet_size':   _strong_anom(),    # gros paquets (payload données, ≠ SYN ~60B) — RENFORCÉ
        'reconstruction': _mod_susp(),      # relation bytes~packets perturbée — RENFORCÉ
        'volume_surplus': _mod_susp(),      # bytes ↑ modéré
        'volume_deficit': _strong_safe(),
    },

    # ── NETWORK OUTAGE / PANNE ────────────────────────────────────────────────
    # Chute de TOUT le trafic → résidus élevés sur bytes, packets, flows.
    # NOTE DIRECTION : la ternaire {Safe,Susp,Anom} ne distingue pas surplus vs déficit.
    # volume=strong_anom est PARTAGÉ avec les floods — la discrimination repose sur le
    # fait que TOUS les protocoles (tcp,udp,icmp) sont simultaneously safe (chute globale)
    # alors que pour un flood, au moins un protocole est anormal.
    # Ref : observation RedeRio Dec 2025 ; Matta et al. (2012) ISP network events
    'NETWORK_OUTAGE': {
        'volume':        _strong_anom(),    # bytes+packets : résidu absolu élevé (chute globale)
        'protocol_tcp':  _strong_safe(),    # pas d'explosion TCP
        'protocol_udp':  _strong_safe(),    # pas d'explosion UDP
        'protocol_icmp': _strong_safe(),    # pas de flood ICMP
        'tcp_flags':     _strong_safe(),    # pas de SYN/FIN surge
        'fin_ratio':     _neutral(),        # trafic minimal → ratio indéterminé
        'connections':   _strong_anom(),    # flows chutent aussi
        'entropy':       _neutral(),        # indéterminé (peu de paquets)
        'packet_size':   _neutral(),        # indéterminé
        'reconstruction': _strong_anom(),   # relations structurelles cassées
        'volume_surplus': _strong_safe(),   # pas de flood positif
        'volume_deficit': _strong_anom(),   # bytes/packets/flows CHUTENT ← seul type à avoir Anom ici
    },

    # ── BOTNET C&C ────────────────────────────────────────────────────────────
    # Connexions périodiques régulières vers C&C → beaconing
    # Faible volume, pattern temporel régulier
    # Ref : MITRE ATT&CK T1071 ; Binsalleeh 2010 IEEE
    'BOTNET_CC': {
        'volume':        _mod_safe(),       # trafic faible (heartbeat)
        'protocol_tcp':  _weak_susp(),      # TCP (HTTP/HTTPS C&C)
        'protocol_udp':  _neutral(),
        'protocol_icmp': _strong_safe(),
        'tcp_flags':     _mod_susp(),       # connexions courtes et répétées
        'fin_ratio':     _mod_susp(),       # beaconing → ratio perturbé → SUSP
        'connections':   _mod_susp(),       # flows réguliers (beaconing)
        'entropy':       _mod_safe(),       # port destination fixe (C&C port)
        'packet_size':   _weak_susp(),      # petits paquets (heartbeat)
        'reconstruction': _weak_susp(),
        'volume_surplus': _mod_safe(),
        'volume_deficit': _strong_safe(),
    },

    # ── NTP AMPLIFICATION ─────────────────────────────────────────────────────
    # Attaque de réflexion/amplification via serveurs NTP vulnérables.
    # BAF = 556.9 >> DNS BAF = 28-54 (van Rijswijk-Deij 2014 IMC).
    # Commande monlist (CVE-2013-5211) : 8 bytes → jusqu'à 48 paquets × ~480B chacun.
    #
    # PROFIL SBN vs DNS_AMP :
    #   Similaire à DNS_AMP (même mécanisme amplification UDP) MAIS :
    #   - packet_size↑↑↑↑ encore plus fort (monlist >> réponse DNS 4KB)
    #   - reconstruction encore plus cassée (bytes/packets ratio NTP)
    #   - entropy src_port : port 123 fixe ≡ DNS port 53 → même comportement SAFE
    #
    # LIMITE CONNUE : NTP_AMP et DNS_AMP partagent un profil SBN très proche.
    # La discrimination exacte nécessiterait les ports (123 vs 53) non disponibles
    # dans les groupes actuels. Le système peut confondre les deux — documenter
    # comme limitation de la couverture de métriques (cf. §9.3 Future Work).
    # Ref : Rossow (2014) NDSS Table 2 — DNS/NTP : même vecteur, BAF différent.
    'NTP_AMP': {
        # bytes↑↑↑↑ (BAF×556), packets↑↑ → volume encore plus fort que DNS_AMP
        'volume': _strong_anom(),
        # UDP ↑↑↑ (port 123) — pur UDP, encore plus fort que DNS (BAF supérieur)
        'protocol_udp': _strong_anom(),
        # Aucun TCP (UDP pur, aucun handshake)
        'protocol_tcp': _strong_safe(),
        # Aucun ICMP
        'protocol_icmp': _strong_safe(),
        # syn=0, fin=0 (UDP → pas de TCP → prophet_syn et prophet_fin en SAFE)
        'tcp_flags': _strong_safe(),
        # UDP → reconst_fin_from_syn indéterminé → SAFE (pas de TCP)
        'fin_ratio': _strong_safe(),
        # Flows modérés (chaque réflecteur ouvre sa session UDP) → légèrement suspect
        'connections': _mod_susp(),
        # src_ip ↑↑ (nombreux réflecteurs NTP vulnérables) = Anom
        # src_port fixe = port 123 (tous les réflecteurs) = Safe  → net: Susp
        'entropy': _mod_susp(),
        # SIGNAL CLÉ vs DNS_AMP : paquets monlist encore plus lourds (600 IPs × 72B)
        # Cisco CVE-2013-5211 : réponse jusqu'à 5500× la requête initiale de 8 bytes
        # → avg_pkt_size plus extrême que DNS_AMP (3KB-4KB) → custom {S:0.01, M:0.04, A:0.95}
        'packet_size': {_S: 0.01, _M: 0.04, _A: 0.95},
        # bytes/packets ratio encore plus cassé que DNS_AMP (BAF ×10) → signal fort
        'reconstruction': _strong_anom(),
        # Surplus positif fort (bytes amplifiés en direction de la victime)
        'volume_surplus': _strong_anom(),
        'volume_deficit': _strong_safe(),
    },

    # ── BRUTE FORCE SSH ───────────────────────────────────────────────────────
    # Attaque par dictionnaire sur SSH (port 22). Très fréquente sur réseaux académiques.
    # Ref : Najafabadi et al. (2015) ICMLA — signal primaire : nb flows agrégés 5min
    # Ref : Hynek et al. (2020) IFIP SEC — campus réseau, ML IP flows étendus
    # Ref : Hofstede et al. (2014) ACM SIGCOMM CCR — trafic "plat" (PPF/BPF constants)
    #
    # Profil NetFlow (attaque distribuée botnet) :
    #   flows ↑↑↑  : SIGNAL PRIMAIRE — centaines de connexions TCP courtes vers port 22
    #   syn ↑↑     : chaque tentative = TCP SYN (nouvelle connexion)
    #   fin modéré : échecs → RST (pas FIN) ; succès → FIN → fin/syn < 1
    #   tcp ↑↑     : 100% TCP (SSH)
    #   avg_pkt_size ↓ : SSH handshake ~100-300B (Hofstede 2014 : BPF faible)
    #   entropy_dst_port ↑ (ANOM) : port 22 FIXE → concentration anormale
    #   entropy_src_ip : modéré (botnet distribué = plusieurs sources)
    #   bytes : modéré (nombreuses petites connexions, pas de volume DDoS)
    #
    # DISCRIMINATEURS clés vs PORT_SCAN :
    #   volume=weak_susp (bytes non-nuls, ≠ scan bytes≈0)
    #   tcp_flags=mod_anom (syn↑↑ persistant, ≠ scan syn modéré)
    #   protocol_tcp=strong_anom (100% TCP, ≠ scan mod_susp)
    #   fin_ratio=mod_susp (RST > FIN sur échecs, ≠ scan plus aléatoire)
    'BRUTE_FORCE_SSH': {
        # bytes modéré (centaines d'octets × nombreuses tentatives — pas de flood volumétrique)
        # ANTI-SIGNAL DDoS : distingue BF des floods (key discriminator)
        'volume': _weak_susp(),
        # 100% TCP (SSH utilise TCP exclusivement)
        # Ref : Hynek 2020 — protocol_tcp est feature discriminante #2 (après flows)
        'protocol_tcp': _strong_anom(),
        # UDP absent (SSH = TCP uniquement)
        'protocol_udp': _strong_safe(),
        # ICMP absent
        'protocol_icmp': _strong_safe(),
        # syn ↑↑ (chaque tentative ouvre un SYN) + fin modéré (RST sur échecs)
        # → prophet_syn fort ATK, prophet_fin faiblement ATK → moyenne : mod_anom
        'tcp_flags': _mod_anom(),
        # fin/syn < 1 : beaucoup de SYN sans FIN correspondant (RST sur échecs)
        # Ref : Hofstede 2014 SSHCure — ratio complétion TCP < 1 caractérise phase BF
        # Moins extrême que Slowloris (fin≈0) et SYN Flood (fin=0) → mod_susp
        'fin_ratio': _mod_susp(),
        # flows ↑↑↑ : SIGNAL PRIMAIRE
        # Najafabadi 2015 : nb_flows = feature discriminante n°1 pour SSH BF
        'connections': _strong_anom(),
        # entropy_dst_port ↑ (ATK) : port 22 fixe = concentration anormale du trafic
        # entropy_src_port ↑ (ATK) : ports source variés par les bots (aléatoires)
        # entropy_src_ip modéré (ATK) : botnet distribué = plusieurs adresses sources
        # → net : forte anomalie entropique → strong_anom
        'entropy': _strong_anom(),
        # avg_pkt_size ↓ (ANOM) : SSH handshake ~200B << trafic normal backbone
        # Hofstede 2014 : BPF "flat and low" = signature de la phase BF
        'packet_size': _mod_anom(),
        # bytes~packets légèrement perturbé (petits paquets uniformes) → faible signal
        'reconstruction': _weak_susp(),
        # Pas de surplus volumétrique (BF ≠ DDoS — bytes modérés)
        'volume_surplus': _weak_susp(),
        'volume_deficit': _strong_safe(),
    },

    # ── DNS TUNNELING ─────────────────────────────────────────────────────────
    # Exfiltration covert via DNS (MITRE ATT&CK T1048.001).
    # Données encodées dans noms de sous-domaines (base32/base64).
    # Outils : iodine, dns2tcp, dnscat2, DNSExfiltrator.
    # Ref : Sharma et al. (2018) Procedia CS 132 — labeled DNS flow dataset
    # Ref : Habibi et al. (2019) IEEE IM — single host detection, 10Gbps backbone
    # Ref : MDPI Electronics (2023) 12(6):1467 — DNS normal 50-550B, tunnel > 550B
    #
    # Profil NetFlow (hôte compromis unique) :
    #   bytes ↑ modéré : exfiltration encodée (pas d'amplification — ≠ DNS_AMP)
    #   udp ↑ modéré   : DNS/UDP/53 pour petites requêtes
    #   tcp ↑ modéré   : DNS/TCP/53 pour requêtes > 512B (MDPI 2023)
    #   avg_pkt_size ↑ : noms encodés > DNS normal (Habibi 2019 : labels > 52 chars)
    #   entropy_src_port ↑↑ : ports source VARIÉS (client tunneling) = ATK
    #   entropy_src_ip ↓↓   : UNIQUE hôte compromis = STRONG SAFE ← DISCRIMINATEUR vs DNS_AMP
    #   entropy_dst_port ↓  : port 53 fixe = Safe (comme DNS_AMP)
    #   fin/syn ≈ 1 : TCP/53 se ferme proprement (≠ Slowloris : fin≈0)
    #
    # DISCRIMINATEURS clés vs DNS_AMP :
    #   entropy_src_ip = strong_safe (source unique ≠ nombreux réflecteurs DNS_AMP)
    #   volume = mod_susp (exfiltration modérée ≠ fort amplification)
    #   fin_ratio = neutral (TCP/53 normal ≠ DNS_AMP : fort safe)
    #
    # DISCRIMINATEURS clés vs DATA_EXFIL :
    #   protocol_udp = mod_susp (UDP présent via DNS ≠ DATA_EXFIL : strong_safe)
    #   entropy = mod_susp (entropy_src_port variés ≠ DATA_EXFIL : strong_safe)
    #
    # LIMITE CONNUE : profil "soft" (aucun signal dominant) → incertitude u_sbn élevée
    # → risque de classification Autre_Anomalie ou confusion DATA_EXFIL.
    # Documenter comme cas difficile basse intensité dans l'analyse de qualification.
    'DNS_TUNNELING': {
        # bytes ↑ modéré (exfiltration) — beaucoup moins fort que DNS_AMP (pas d'amplification)
        'volume': _mod_susp(),
        # UDP ↑ modéré (DNS/UDP/53 pour petites requêtes encodées)
        'protocol_udp': _mod_susp(),
        # TCP ↑ modéré (DNS/TCP/53 pour grandes requêtes > 512B)
        # Ref MDPI 2023 : "DNS queries > 512 bytes use TCP"
        'protocol_tcp': _mod_susp(),
        # ICMP absent (DNS sur UDP/TCP uniquement)
        'protocol_icmp': _strong_safe(),
        # syn modéré (TCP/53 ouvre des connexions) + fin modéré (TCP/53 fermé proprement)
        # → ni fort ATK, ni fort Safe → faible signal tcp_flags
        'tcp_flags': _weak_susp(),
        # fin/syn ≈ 1 : chaque TCP/53 ouvert se ferme proprement (≠ Slowloris, ≠ SYN Flood)
        # Neutral pour ne pas confondre avec les attaques à fin≈0 ou fin<<syn
        'fin_ratio': _neutral(),
        # flows ↑ modéré (sessions DNS persistantes — hôte compromis → serveur attaquant)
        # Ref Sharma 2018 : avg_flows_per_minute anormalement élevé
        'connections': _mod_susp(),
        # entropy_src_ip ↓↓ (SAFE fort) : UN SEUL hôte compromis → source unique
        #   SIGNAL CLÉ vs DNS_AMP (multi-réflecteurs → entropy↑↑)
        # entropy_src_port ↑↑ (ATK fort) : client DNS tunneling = ports variés (Sharma 2018)
        # entropy_dst_port ↓ (SAFE) : port 53 fixe
        # Geomean des 3 : SAFE × ATK × SAFE → net modérément suspect (ATK atténué par les SAFE)
        'entropy': _mod_susp(),
        # avg_pkt_size ↑ (légèrement ANOM) : noms encodés plus longs que DNS normal
        # Ref Habibi 2019 : labels jusqu'à 63 chars → paquets > 550B (plage normale DNS)
        'packet_size': _mod_susp(),
        # reconst_bytes_from_entropy_src_port perturbé (bytes↑ + entropy_src_port↑↑)
        # → signal modéré dans le groupe reconstruction
        'reconstruction': _mod_susp(),
        # Surplus modéré (exfiltration → bytes ↑ modéré)
        'volume_surplus': _weak_susp(),
        'volume_deficit': _strong_safe(),
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# MATRICE DE TRANSITION MARKOVIENNE ENTRE TYPES
# T[k_prev][k_next] = P(type_actuel = k_next | type_précédent = k_prev)
# Justification : les attaques persistent (haute diagonale) et certaines
# transitions sont naturelles dans la kill chain (PORT_SCAN → DATA_EXFIL).
# Ref : Hutchins et al. 2011 Lockheed Martin Kill Chain
# ──────────────────────────────────────────────────────────────────────────────

def _build_transition_matrix(type_names: list) -> np.ndarray:
    """
    Construit la matrice de transition Markovienne entre types d'attaque.

    Règles :
      - Diagonale : haute (0.80) par défaut — les attaques persistent
      - PORT_SCAN → DATA_EXFIL : 0.20 (recon → exfil, kill chain Hutchins 2011)
      - NETWORK_OUTAGE → NETWORK_OUTAGE : 0.92 (pannes longues — override spécifique)
      - Autre_Anomalie → tout : uniforme (inconnu → inconnu ou transition)
      - Lignes normalisées à 1.0

    Ref : Hutchins et al. (2011) Intelligence-Driven CND with Kill Chain.
    """
    K = len(type_names)
    idx = {t: i for i, t in enumerate(type_names)}
    T = np.zeros((K, K))  # initialisation à zéro (les entrées sont toutes écrasées ci-dessous)

    SELF_PROB = 0.80          # persistance nominale par défaut
    for i in range(K):
        T[i, i] = SELF_PROB   # haute auto-transition
        # distribue le reste uniformément
        off = (1.0 - SELF_PROB) / max(K - 1, 1)
        for j in range(K):
            if j != i:
                T[i, j] = off

    # Overrides spécifiques (kill chain Hutchins 2011)
    # NOTE : après normalisation ligne, la persistance effective SELF_PROB n'est plus 0.80
    # pour les types avec des overrides non-diagonaux élevés.
    # Exemple pour K=11 : off = 0.02, row_sum_PORT_SCAN = 0.80+0.20+0.05+9*0.02 = 1.23
    # → T_eff[PORT_SCAN,PORT_SCAN] ≈ 0.80/1.23 ≈ 0.65 (persistance effective)
    # Voir tableau complet des valeurs effectives dans la documentation.
    overrides = [
        # (from, to, prob_nominale)
        ('PORT_SCAN',         'DATA_EXFIL',      0.20),  # recon → exfil (kill chain Hutchins 2011)
        ('PORT_SCAN',         'HTTP_FLOOD',       0.05),  # scan → flood (opportuniste)
        ('BOTNET_CC',         'UDP_FLOOD',        0.15),  # C&C → flood (DDoS commandé)
        ('BOTNET_CC',         'SYN_FLOOD',        0.10),
        ('NETWORK_OUTAGE',    'NETWORK_OUTAGE',   0.92),  # pannes persistantes
        # ── Nouvelles transitions (kill chain étendue) ──────────────────────
        # Brute Force → escalade vers exfiltration ou botnet (Hutchins 2011 §3)
        # Ref : MITRE ATT&CK T1110 (BruteForce) → T1048 (Exfil) / T1071 (C&C)
        ('BRUTE_FORCE_SSH',   'DATA_EXFIL',       0.10),  # credential → exfiltration
        ('BRUTE_FORCE_SSH',   'DNS_TUNNELING',    0.08),  # compromis → tunnel DNS exfil
        ('BRUTE_FORCE_SSH',   'BOTNET_CC',        0.05),  # compromis → bot C&C
        # DNS Tunneling ↔ DNS_AMP : même infrastructure DNS
        ('DNS_TUNNELING',     'DNS_AMP',          0.05),
        # NTP_AMP persist au même titre que DNS_AMP (mêmes serveurs réflecteurs)
        ('NTP_AMP',           'DNS_AMP',          0.05),
]

    for (src, dst, p) in overrides:
        if src in idx and dst in idx:
            i, j = idx[src], idx[dst]
            T[i, j] = p

    # Normalisation ligne par ligne → matrice stochastique (simplex)
    row_sums = T.sum(axis=1, keepdims=True)
    T = T / np.maximum(row_sums, 1e-12)

    # Vérifications de cohérence stochastique
    assert np.allclose(T.sum(axis=1), 1.0, atol=1e-9), \
        f"Matrice T non stochastique : row_sums = {T.sum(axis=1)}"
    assert (T >= 0).all(), \
        "Matrice T contient des entrées négatives — vérifier les overrides"

    return T


# ──────────────────────────────────────────────────────────────────────────────
# FONCTIONS NOYAU SBN
# ──────────────────────────────────────────────────────────────────────────────

STATES = ('Safe', 'Susp', 'Anom')
STATE_COLS = {
    'Safe': 'proj_safe',
    'Susp': 'proj_susp',
    'Anom': 'proj_atk',
}

EPS = 1e-12


def _compute_group_projected(row: pd.Series, group_name: str) -> dict | None:
    """
    Calcule les probabilités projetées P^{obs}_{g,s} pour un groupe sémantique.

    Pour chaque état s ∈ {Safe, Susp, Anom} :
        P^g_s = geomean_{m ∈ groupe} P^m_s
    où P^m_s est la prob. projetée de la métrique m pour l'état s.

    La moyenne géométrique est invariante à l'ordre de magnitude et évite
    qu'une seule métrique dominante ne biaise le groupe.
    Fondement : logarithmic opinion pooling (Genest & Zidek 1986, Statist. Sci. §4) —
    la moyenne géométrique normalisée est l'agrégateur optimal minimisant la divergence KL
    à la pooled opinion sous contrainte de calibration.
    (Aczél & Daróczy 1975 — caractérisation axiomatique des moyennes)

    Retourne None si aucune colonne disponible.
    """
    sources = GROUP_SOURCES.get(group_name, [])
    result = {}

    for state, col_suffix in STATE_COLS.items():
        vals = []
        for src in sources:
            col = f"{src}_{col_suffix}"
            if col in row.index and pd.notna(row[col]):
                vals.append(float(row[col]))
        if not vals:
            result[state] = None
        else:
            log_mean = np.mean([np.log(max(v, EPS)) for v in vals])
            result[state] = float(np.exp(log_mean))

    if all(v is None for v in result.values()):
        return None

    # Normalisation sur les états disponibles
    total = sum(v for v in result.values() if v is not None)
    if total < EPS:
        return None
    for state in STATES:
        if result[state] is None:
            result[state] = 0.0
    total = sum(result.values())
    if total > EPS:
        result = {s: v / total for s, v in result.items()}

    return result


def _sbn_group_score(group_pp: dict, cond_opinion: dict) -> float:
    """
    Score SBN pour un groupe g et un type k.

    Score(k, g) = Σ_s P^{obs}_{g,s} · c^{k|g}_s

    C'est le produit scalaire entre :
      - les probabilités projetées observées P^{obs}_{g,s}
      - les probabilités conditionnelles P(G=s | type_k) depuis SBN_COND_OPINIONS

    Propriétés :
      - Neutre (= 1/3) si group_pp est uniforme OU si cond_opinion est uniforme
      - Maximum si group_pp = delta sur l'état attendu par cond_opinion
      - Continu et différentiable — meilleure stabilité numérique que LR brut

    Interprétation : Score(k,g) = E_{s ~ P^obs_g}[c^{k|g}_s] — espérance de la
    compatibilité de l'état s avec le type k, sous la distribution observée du groupe g.
    C'est une mesure de compatibilité entre l'observation et la signature experte.

    Note : ce n'est PAS P(obs|k) au sens d'un modèle génératif bayésien strict.
    C'est une approximation par proxy discrétisé (Duda & Hart 1973, Pattern Classification,
    §2.6 — nearest-mean classifier). La connexion bayésienne est approchée via la
    marginalisation discrète sur les états {Safe, Susp, Anom}.

    Returns : float in (0, 1]
    """
    score = 0.0
    for state in STATES:
        p_obs  = group_pp.get(state, 0.0)
        p_cond = cond_opinion.get(state, 1.0 / 3.0)
        score += p_obs * p_cond
    return max(score, EPS)


def _normalize_cond_opinion(cond: dict) -> dict:
    """Normalise une opinion conditionnelle (somme = 1.0)."""
    total = sum(cond.get(s, 0.0) for s in STATES)
    if total < EPS:
        return {s: 1.0 / 3.0 for s in STATES}
    return {s: cond.get(s, 0.0) / total for s in STATES}


def _evidence_sum_scores(scores: list, evidence_scale: float = 3.0) -> float:
    """
    Convertit des dot-product scores en masse d'évidence (Jøsang 2016 Déf. 3.9).

    e(k) = Σ_g max(0, score(k,g) - 1/3) * evidence_scale

    Propriétés :
      - score = 1/3 (groupe neutre)    → contribution = 0  (pas d'évidence)
      - score = 1.0 (match parfait)    → contribution = 2/3 * evidence_scale
      - Tous neutres (nouveauté)       → e(k) = 0 pour tout k → u → 1.0  ✓
      - Un type domine (connu)         → e(top) >> 0, e(autres) ≈ 0 → u << 1  ✓

    Interprétation Bayes Factor : l'écart à la neutralité 1/3 est équivalent
    au log-odd d'un test d'hypothèse contre H0 (neutralité uniforme).
    Ref : Good 1952 Ann. Math. Stat. §4 (scoring rules).

    IMPORTANT : remplace l'ancienne _geomean_log_scores qui compressait tous les
    scores vers 0.03-0.04, rendant la bijection SL non-discriminante.
    """
    if not scores:
        return 0.0
    return float(sum(max(0.0, s - 1.0 / 3.0) * evidence_scale for s in scores))


def _sl_bijection(likelihoods: dict) -> tuple[dict, float]:
    """
    Bijection SL sur evidence counts (Jøsang 2016, Déf. 3.9, cas multinomial).

    Formulation evidence-based :
        b(k)  = e(k) / D
        u     = W / D
        D     = Σ_k e(k) + W

    où e(k) sont des masses d'évidence positives (sortie de _evidence_sum_scores).

    W = K = len(likelihoods) : valeur canonique pour un cadre à K hypothèses
    (Jøsang §3.5.2 : W=K garantit u=1.0 à évidence nulle et prior uniforme).

    Différence vs ancienne version :
      - Ancienne : W=3 fixe (domaine ternaire) → W≠K pour K=11 types → u trop bas
      - Nouvelle  : W=K dynamique
          → si Σ e(k) ≈ 0 (nouveauté) : u = K/K = 1.0              ✓
          → si e(top) ≈ 18 (9 groupes parfaits, K=11) : u ≈ 11/29 ≈ 0.38  ✓

    Classe résiduelle Autre_Anomalie :
        b(autre) = 0.0  [la masse résiduelle est portée par u * a_autre]
        Conforme à Jøsang §14.4 : la classe résiduelle n'a pas d'évidence directe.

    Contrainte SL :  Σ b(k) + b(autre) + u = 1  ← vérifiée
    """
    # W = K : valeur canonique pour un cadre à K hypothèses (Jøsang §3.5.2).
    # max(K, 2) garantit W ≥ 2 (domaine binaire minimal — évite D=0 pour K=1).
    # Base rate implicite : a(k) = 1/K (prior uniforme — ignorance totale sur les
    # fréquences d'attaque). Justification : absence de distribution empirique des
    # types d'attaque sur le réseau cible → prior non-informatif de Jøsang §3.5.2.
    K = len(likelihoods)
    _w_cfg = CONFIG.get("SBN_PRIOR_WEIGHT", "K") if CONFIG else "K"
    if _w_cfg == "K":
        W = float(max(K, 2))
    else:
        # W fixe (ex. 2.0) : Jøsang 2013 §3.2.2 + R-EDL OpenReview 2023
        W = float(max(float(_w_cfg), 2.0))

    D = sum(likelihoods.values()) + W
    if D < EPS:
        b_out = {k: 0.0 for k in likelihoods}
        b_out['Autre_Anomalie'] = 0.0
        return b_out, 1.0

    b_out = {k: v / D for k, v in likelihoods.items()}
    b_out['Autre_Anomalie'] = 0.0   # masse résiduelle via u * a_autre
    u_out = W / D
    return b_out, u_out


def _apply_um(b_dict: dict, u: float) -> tuple[dict, float]:
    """
    Uncertainty Maximisation post-qualification (Jøsang 2016, §3.6, Eq. 3.27).

    Maximise u tout en préservant les probabilités projetées P(xi) = b(xi) + a(xi)*u.
    Prior uniforme a(xi) = 1/K sur les K hypothèses ACTIVES (hors classe résiduelle).
    Hypothèse : base rate uniforme — ignorance a priori sur la fréquence des types
    d'attaque (prior non-informatif, cohérent avec Jøsang §3.5.2 et _sl_bijection).

    La classe résiduelle 'Autre_Anomalie' (toujours b=0) est exclue du calcul de K
    et de u_max : l'inclure bloquerait u_max = u (Autre_Anomalie contraindrait
    systématiquement min(P) = u/K_total → u_max = u, UM sans effet).
    Après calcul, b[Autre_Anomalie] est remis à 0.0 (invariant).

    Effet différentiel (avec K=11 types actifs) :
      - Attaque connue bien classifiée (b_top dominant) : u_max modéré (~0.3-0.6)
      - Anomalie inconnue (b distribuées uniformément) : u_max → 1.0

    Invariant : P(xi) préservées (Eq. 3.23) — neutre sur la classification.
    """
    _RESIDUAL = 'Autre_Anomalie'

    # Séparer hypothèses actives (K) de la classe résiduelle
    active_keys   = [k for k in b_dict if k != _RESIDUAL]
    K1            = len(active_keys)
    if K1 == 0:
        return dict(b_dict), float(u)

    a_i = 1.0 / K1

    # Probabilités projetées P(xi) = b(xi) + a(xi)*u  (sur les K hypothèses actives)
    P = {k: b_dict[k] + a_i * u for k in active_keys}

    # u_max = min_i(P(xi)/a(xi)) = K1 * min_i(P(xi))
    u_max = float(np.clip(K1 * min(P.values()), 0.0, 1.0))

    # Recalcul des belief masses actives
    b_new = {k: max(P[k] - a_i * u_max, 0.0) for k in active_keys}
    b_new[_RESIDUAL] = 0.0  # classe résiduelle invariante

    # Renormalisation numérique
    total = sum(b_new.values()) + u_max
    if total > EPS and not np.isclose(total, 1.0):
        scale = 1.0 / total
        b_new = {k: v * scale for k, v in b_new.items()}
        u_max *= scale

    return b_new, float(u_max)


def _wbf_two(b1: dict, u1: float, w1: float,
             b2: dict, u2: float, w2: float) -> tuple[dict, float]:
    """
    WBF (Weighted Belief Fusion) entre deux opinions sur le même domaine.

    Jøsang 2016, Eq. 12.22 :
        b^{fused}(k) = (w1·b1(k) + w2·b2(k)) / (w1 + w2)
        u^{fused}    = (w1·u1 + w2·u2) / (w1 + w2)

    Propriété : WBF est une moyenne pondérée des belief masses.
    Utilisé ici pour la fusion avec le prior temporel (w2 = temporal_weight).

    Note : CBF (addition d'évidence, Eq. 12.14) serait plus appropriée si
    les deux sources sont indépendantes; WBF est préféré ici car le prior
    temporel et l'évidence courante ne sont PAS indépendants (même phénomène).
    """
    wt = w1 + w2
    if wt < EPS:
        return dict(b1), u1

    keys = set(b1.keys()) | set(b2.keys())
    b_fused = {}
    for k in keys:
        b_fused[k] = (w1 * b1.get(k, 0.0) + w2 * b2.get(k, 0.0)) / wt
    u_fused = (w1 * u1 + w2 * u2) / wt

    return b_fused, float(u_fused)


def _discount_opinion(b: dict, u: float, decay: float) -> tuple[dict, float]:
    """
    Escompte d'une opinion avec facteur de décroissance temporelle.

    b_d(k) = decay * b(k)
    u_d    = 1 - decay * (1 - u) = 1 - decay * Σ b(k)

    Interprétation : avec decay=0.8, la confiance (1-u) réduit de 20%.
    decay = λ^(Δt fenêtres)  avec λ ∈ CONFIG.SBN_TEMPORAL_DECAY.

    Ref : Jøsang 2016, Définition 14.6 (trust discounting)
    """
    decay = float(np.clip(decay, 0.0, 1.0))
    b_d = {k: decay * v for k, v in b.items()}
    sum_b = sum(b.values())
    u_d = 1.0 - decay * sum_b
    u_d = float(np.clip(u_d, 0.0, 1.0))

    # Renormalisation
    total = sum(b_d.values()) + u_d
    if total > EPS and not np.isclose(total, 1.0):
        scale = 1.0 / total
        b_d = {k: v * scale for k, v in b_d.items()}
        u_d *= scale

    return b_d, float(u_d)


# ──────────────────────────────────────────────────────────────────────────────
# NOYAU PRINCIPAL : QUALIFICATION SBN D'UNE FENÊTRE
# ──────────────────────────────────────────────────────────────────────────────

def _lr_novelty(likelihoods: dict) -> float:
    """
    Métrique de nouveauté basée sur la dominance relative du meilleur type.

    LR_novelty = max(L) / mean(L)
    novelty_lr = 1 / LR_novelty  ∈ (0, 1]

    Propriétés :
      - Attaque connue (un type domine clairement) : LR >> 1 → novelty_lr modéré (0.4–0.6)
      - Anomalie inconnue (tous types similaires)  : LR ≈ 1 → novelty_lr ≈ 1

    Cette métrique est bien plus sensible que u_sbn (qui a un gap de seulement 0.007)
    parce qu'elle mesure la CONCENTRATION du signal sur un type unique,
    indépendamment du niveau absolu d'incertitude.

    Seuil recommandé : novelty_lr > 0.85 pour signal de nouveauté.
    Validation sur signatures théoriques (group_pp = SBN_COND[k]) :
      - Attaques connues  : 0.47–0.62 (toutes < 0.85)
      - Anomalie inconnue : 0.996      (>> 0.85)
    ATTENTION : cette validation est effectuée sur des signatures parfaites, pas sur
    des données réelles. Sur données bruitées, les attaques connues peuvent avoir
    novelty_lr plus proche de 1.0. Le seuil 0.85 doit être recalibré empiriquement
    sur le dataset d'injection (procédure recommandée : courbe ROC sur novelty_lr).

    Ref : rapport de vraisemblance maximum/moyen (Good 1950, *Probability and the
    Weighing of Evidence* §6 — "weight of evidence" maximum sur K hypothèses).
    Ce n'est PAS un concept de la théorie Dempster-Shafer (Shafer 1976) —
    c'est un indice de concentration analogue à l'inverse de l'indice de Herfindahl.
    """
    if not likelihoods:
        return 1.0
    L_vals = list(likelihoods.values())
    max_L  = max(L_vals)
    mean_L = sum(L_vals) / len(L_vals)
    lr = max_L / mean_L if mean_L > EPS else 1.0
    return float(1.0 / lr)

def _compute_jsd_weights(sbn_cond: dict, active_groups: set) -> dict:
    """
    Poids discriminant de chaque groupe = JSD entre conditionnelles de types.

    JSD_g = H(mean_k P(G=s|k)) - mean_k H(P(G=s|k))
          = 0   si toutes les conditionnelles sont identiques (groupe non-discriminant)
          = log(3) si elles sont maximalement distinctes

    Normalisation : mean(weights) = 1.0 → préserve l'échelle d'évidence moyenne.

    Ref : Yang & Pedersen (1997) ICML §3 — information gain comme critère de sélection.
          Formellement : JSD = 2 × I(G ; Type) (information mutuelle discrete).
    """
    type_keys = [k for k in sbn_cond if isinstance(sbn_cond[k], dict)]
    if len(type_keys) < 2:
        return {g: 1.0 for g in active_groups}

    jsd_scores = {}
    for grp in active_groups:
        dists = []
        for tk in type_keys:
            cond_raw = sbn_cond[tk].get(grp, _neutral())
            cond = _normalize_cond_opinion(cond_raw)
            p = np.array([cond.get(s, 1.0/3) for s in STATES], dtype=float)
            p = np.maximum(p, EPS)
            p /= p.sum()
            dists.append(p)
        dists_arr = np.array(dists)                           # (K_types, 3)
        mixture   = dists_arr.mean(axis=0)                   # (3,)
        H_mix  = float(-np.sum(mixture * np.log(mixture + EPS)))
        H_mean = float(np.mean(
            [-np.sum(p * np.log(p + EPS)) for p in dists_arr]
        ))
        jsd_scores[grp] = max(H_mix - H_mean, 0.0)

    vals    = list(jsd_scores.values())
    mean_v  = sum(vals) / len(vals) if vals else 1.0
    if mean_v < EPS:
        return {g: 1.0 for g in active_groups}
    return {g: v / mean_v for g, v in jsd_scores.items()}


def sbn_qualify_row(
    row:              pd.Series,
    sbn_cond:         dict,
    threshold:        float,
    prev_opinion:     dict | None = None,   # {'b': {k: float}, 'u': float}
    delta_windows:    int = 1,
    lambda_temporal:  float = 0.80,
    temporal_weight:  float = 0.30,
    apply_temporal:   bool = True,
    apply_um:         bool = True,
    transition_matrix: np.ndarray | None = None,  # matrice T[k_prev][k_next]
    type_names_tm:    list | None = None,          # ordre des types dans T
    evidence_scale:   float = 3.0,                 # facteur d'échelle évidence (SBN_EVIDENCE_SCALE)
    autre_anomalie_prior: float | None = None,
    # ↑ Prior de la classe résiduelle Autre_Anomalie pour la décision top1.
    #   None = 2/K (calibré automatiquement sur le nombre de types actifs).
    #   Principe (Jøsang §14.4) : top1 = argmax_k P_proj(k) où k inclut Autre_Anomalie.
    #   P_proj(Autre) = u × a_autre  avec  a_autre = autre_anomalie_prior.
    #   Condition pour que Autre gagne :  u × (a_autre − 1/K) > b_top.
    #   Avec a_autre = 2/K  →  Autre gagne ⟺  b_top < u/K
    #   (= "aucun type ne dépasse son prior d'une quantité supérieure à u/K").
    #   Mettre 0.0 pour désactiver (comportement pre-redesign).
) -> dict:
    """
    Qualification SBN pour une fenêtre temporelle.

    Paramètres :
        row                  : ligne du CSV opinions (pd.Series)
        sbn_cond             : SBN_COND_OPINIONS depuis config.py
        threshold            : seuil gate P(Anom) = DECISION_THRESHOLD
                               W = K dynamique dans _sl_bijection (Jøsang §3.5.2)
        prev_opinion         : opinion SBN de la fenêtre précédente (gate_open=True)
        delta_windows        : Δt depuis dernière fenêtre gate_open
        lambda_temporal      : facteur de décroissance temporelle λ (Jøsang Déf. 14.6)
                               NOTE : utiliser la même valeur que LAMBDA_DECAY dans
                               compute_opinions_v3.py pour cohérence du système.
        temporal_weight      : poids fixe du prior temporel dans WBF (séparé de l'escompte)
        apply_temporal       : active/désactive le prior temporel (L5)
        apply_um             : active/désactive l'Uncertainty Maximisation (L6)
        transition_matrix    : matrice de transition Markovienne T[k_prev][k_next]
                               (Hutchins 2011 Kill Chain) — None = désactivée
        type_names_tm        : liste des types correspondant aux lignes/colonnes de T
        evidence_scale       : facteur d'échelle évidence (SBN_EVIDENCE_SCALE)
        autre_anomalie_prior : prior de la classe résiduelle (voir ci-dessus)

    Retourne :
        dict avec :
            gate_open, top1_type, top1_b, top1_proj,
            qual_status ('normal'|'qualified'|'autre_anomalie'|'no_groups'),
            b_sbn_{type}, u_sbn, novelty_lr,
            b_sbn_raw_{type}, u_sbn_raw  (avant temporal + UM)
    """
    # Clés de résultat
    type_names = [k for k in sbn_cond.keys()]

    def _empty_result():
        r = {f'b_sbn_{k}': 0.0 for k in type_names}
        r['b_sbn_Autre_Anomalie'] = 0.0
        # u_sbn = 1.0 et novelty_lr = 1.0 par convention pour les fenêtres normales
        # (gate_open=False). Ces valeurs NE DOIVENT PAS être interprétées comme un signal
        # de nouveauté — elles signifient simplement qu'aucune qualification n'a été
        # effectuée. Filtrer sur gate_open=True avant toute analyse de u_sbn ou novelty_lr.
        r['u_sbn'] = 1.0
        r['novelty_lr'] = 1.0
        r['top1_type'] = ''
        r['top1_b'] = 0.0
        r['top1_proj'] = 0.0
        r['qual_status'] = 'normal'   # P_atk < threshold : fenêtre non-anormale
        r['gate_open'] = False
        r['u_sbn_raw'] = 1.0
        for k in type_names:
            r[f'b_sbn_raw_{k}'] = 0.0
        r['b_sbn_raw_Autre_Anomalie'] = 0.0
        return r

    # ── L1 : Gate ────────────────────────────────────────────────────────────
    p_atk = row.get(_DET_COL, row.get('FINAL_SYSTEM_CBF_proj_atk', 0.0))
    if pd.isna(p_atk) or p_atk < threshold:
        return _empty_result()

    result = _empty_result()
    result['gate_open'] = True

    # ── L2 : Opinions de groupes projetées ───────────────────────────────────
    group_pp = {}
    for grp in GROUP_SOURCES:
        pp = _compute_group_projected(row, grp)
        if pp is not None:
            group_pp[grp] = pp

    if not group_pp:
        # Aucune métrique disponible : qualification impossible pour cette fenêtre.
        # u_sbn = 1.0 par manque de données (distinct du cas "anomalie inconnue").
        # qual_status = 'no_groups' permet de filtrer ce cas dans les analyses aval.
        result['u_sbn'] = 1.0
        result['novelty_lr'] = 1.0
        result['qual_status'] = 'no_groups'
        return result

    # ── NETWORK_OUTAGE classifié nativement par le SBN (L3-L4) ───────────────
    # Pas de gate séparée. Mécanisme de discrimination vs FLOOD :
    #   - volume=strong_anom est PARTAGÉ par OUTAGE et les floods (résidus absolus élevés
    #     dans les deux cas — la ternaire {Safe,Susp,Anom} ne distingue pas la direction).
    #   - La discrimination repose sur les groupes PROTOCOLES : pour un OUTAGE, tous les
    #     protocoles (tcp, udp, icmp) chutent simultanément → proj_safe élevé pour chaque
    #     groupe protocole. Pour un FLOOD, au moins un protocole explose → proj_anom élevé.
    # ── NETWORK_OUTAGE classifié nativement par le SBN (L3-L4) ───────────────
    # Discrimination OUTAGE vs FLOOD via les groupes directionnels 5-états :
    #   volume_surplus : résidus positifs (bytes/packets/flows ↑) → FLOOD
    #   volume_deficit : résidus négatifs (bytes/packets/flows ↓) → OUTAGE
    # Pour un OUTAGE : volume_surplus=SAFE, volume_deficit=STRONG_ANOM (seul type)
    # Pour un FLOOD  : volume_surplus=STRONG_ANOM, volume_deficit=SAFE
    # Colonne {metric}_dir_pos/neg_proj_* produite par compute_opinions_v3.
    # Ref : Jøsang §3.5.4 coarsening ; Matta et al. (2012) ISP network events.

    # ── L3-L4 : Score SBN par type et bijection SL ───────────────────────────
    # e(k) = Σ_g max(0, Score(k,g) - 1/3) * evidence_scale
    # Score(k,g) = E_{s~P^obs_g}[c^{k|g}_s] (espérance du signal de compatibilité
    #              avec le type k sous la distribution observée du groupe g).
    # Hypothèse (Naive Bayes) : indépendance conditionnelle des groupes étant donné
    # le type (Rish 2001 IJCAI ; Mitchell 1997 Ch.6).
    # Corrélations résiduelles attendues et leur impact :
    #   1. volume ↔ protocol_udp/tcp : corrélés pour floods volumétriques.
    #      Impact : légère surestimation du score du bon type → biais PRO-discrimination.
    #   2. tcp_flags ↔ reconstruction (reconst_fin_from_syn) : double-comptage corrigé
    #      en retirant reconst_fin_from_syn du groupe 'reconstruction' (voir config.py).
    # Validation : Domingos & Pazzani (1997) Machine Learning 29(2-3):103-130 — Naive Bayes reste robuste sous
    # violations modérées d'indépendance, surtout si les corrélations favorisent le bon type.

    _ev_mode = CONFIG.get("SBN_EVIDENCE_MODE", "absolute")
    _use_jsd = CONFIG.get("SBN_USE_JSD_WEIGHTS", False)

    # Matrice de scores (K_types × G_groups)
    score_matrix: dict[str, dict[str, float]] = {}
    active_grps: set = set()
    for type_k, cond_by_group in sbn_cond.items():
        if not isinstance(cond_by_group, dict):
            continue
        score_matrix[type_k] = {}
        for grp, cond_raw in cond_by_group.items():
            if grp not in group_pp:
                continue
            cond_norm = _normalize_cond_opinion(cond_raw)
            sc = _sbn_group_score(group_pp[grp], cond_norm)
            score_matrix[type_k][grp] = sc
            active_grps.add(grp)

    # Levier 3 : poids JSD par groupe
    jsd_w = _compute_jsd_weights(sbn_cond, active_grps) if _use_jsd else {
        g: 1.0 for g in active_grps
    }

    # Levier 2 : scoring contrastif ou absolu
    if _ev_mode == "contrastive":
        # Pour chaque groupe, moyenne inter-types → base de référence locale
        grp_mean: dict[str, float] = {
            grp: (
                    sum(score_matrix[t].get(grp, 1.0 / 3) for t in score_matrix) /
                    max(len(score_matrix), 1)
            )
            for grp in active_grps
        }
        likelihoods = {
            t: sum(
                max(0.0, sc - grp_mean.get(g, 1.0 / 3))
                * evidence_scale * jsd_w.get(g, 1.0)
                for g, sc in grp_sc.items()
            )
            for t, grp_sc in score_matrix.items()
        }
    else:
        # Mode absolu original
        likelihoods = {
            t: sum(
                max(0.0, sc - 1.0 / 3) * evidence_scale * jsd_w.get(g, 1.0)
                for g, sc in grp_sc.items()
            )
            for t, grp_sc in score_matrix.items()
        }

    # W = K = len(likelihoods) calculé dynamiquement dans _sl_bijection (Jøsang §3.5.2)
    b_raw, u_raw = _sl_bijection(likelihoods)

    # Métrique de nouveauté LR (calculée AVANT bijection et UM pour préserver le signal)
    # novelty_lr ≈ 1.0 → inconnu (tous types similaires)
    # novelty_lr ≈ 0.5 → connu (un type domine)
    nov_lr = _lr_novelty(likelihoods)

    # Copie raw (avant temporal + UM) pour diagnostics
    for k, v in b_raw.items():
        result[f'b_sbn_raw_{k}'] = v
    result['u_sbn_raw'] = u_raw

    b_final = dict(b_raw)
    u_final = float(u_raw)

    # ── L5 : Prior temporel Markovien (WBF avec opinion t-1 escomptée) ───────
    # Architecture : matrice de transition T (kill chain) → escompte Jøsang
    # Déf. 14.6 → WBF avec poids fixe temporal_weight (séparé de l'escompte).
    # NOTE : un seul mécanisme de décroissance est utilisé (_discount_opinion).
    # temporal_weight est fixe et ne duplique PAS l'escompte.
    if apply_temporal and prev_opinion is not None:
        b_prev = dict(prev_opinion.get('b', {}))
        u_prev = prev_opinion.get('u', 1.0)

        # Application de la matrice de transition Markovienne (kill chain)
        # b_trans[j] = Σ_i T[i][j] * b_prev[i]  (Chapman-Kolmogorov, ordre 1)
        # Ref : Hutchins et al. (2011) Lockheed Martin Kill Chain
        #
        # Propriété clé : T stochastique → sum(T^T @ b_vec) = sum(b_vec) = 1 - u_prev.
        # On conserve b_trans sans renormalisation pour préserver l'incertitude u_prev :
        # l'opinion {b_trans, u_prev} est déjà une opinion SL valide (sum(b)+u=1).
        # Renormaliser à sum=1 forcerait u=0 (opinion dogmatique) avant discounting,
        # ce qui amplifierait artificiellement la confiance du prior temporel (BUG corrigé).
        if transition_matrix is not None and type_names_tm is not None:
            b_vec = np.array([b_prev.get(t, 0.0) for t in type_names_tm])
            b_trans_vec = transition_matrix.T @ b_vec
            b_prev = {t: float(b_trans_vec[i]) for i, t in enumerate(type_names_tm)}
            b_prev['Autre_Anomalie'] = 0.0
            # Pas de renormalisation : sum(b_prev) = 1 - u_prev ← conservé intentionnellement

        # Décroissance exponentielle : decay = λ^(Δt) — Jøsang Déf. 14.6
        decay = float(lambda_temporal ** delta_windows)
        b_prev_d, u_prev_d = _discount_opinion(b_prev, u_prev, decay)

        # WBF avec poids fixe (temporal_weight ≠ decay — mécanismes séparés)
        w_curr = 1.0 - temporal_weight
        b_final, u_final = _wbf_two(
            b_final, u_final, w_curr,
            b_prev_d, u_prev_d, temporal_weight
        )

    # ── L6 : Uncertainty Maximisation ────────────────────────────────────────
    if apply_um:
        b_final, u_final = _apply_um(b_final, u_final)

    # Remplissage résultat
    for k, v in b_final.items():
        result[f'b_sbn_{k}'] = v
    result['u_sbn'] = u_final
    # novelty_lr = 1/LR_dominance : modéré (0.47-0.62) pour attaque connue, ~1.0 pour inconnu
    # Calibration sur signatures théoriques uniquement — recalibrer sur données réelles.
    result['novelty_lr'] = nov_lr

    # ── Décision top-1 : argmax(b_final) parmi les types nommés ─────────────
    # Propriété clé : argmax_k b(k) = argmax_k P_proj(k) = argmax_k [b(k) + u/K]
    # car le terme u/K est une constante additive identique pour tous les types k.
    # → La décision par argmax des belief masses est ÉQUIVALENTE à argmax des
    #   probabilités projetées pour des priors uniformes (Jøsang §14.4, cas uniforme).
    # La classe résiduelle 'Autre_Anomalie' est EXCLUE de ce argmax (b_Autre = 0 toujours) :
    # son signal est porté par 'qual_status' séparément (voir ci-dessous).
    K_types   = len(type_names)
    a_std     = 1.0 / K_types   # prior uniforme sur les K types actifs

    # Probabilités projetées P_proj(k) — pour diagnostics et rapport
    p_proj    = {k: b_final.get(k, 0.0) + u_final * a_std for k in type_names}
    top1      = max(type_names, key=lambda k: b_final.get(k, 0.0))
    top1_b    = b_final[top1]
    top1_proj = p_proj[top1]

    # ── Détection de nouveauté : u_raw vs SBN_NOVELTY_U_RAW_THRESHOLD ────────
    # PRINCIPE : utiliser u_raw (avant UM et avant fusion temporelle) pour mesurer
    # l'incertitude épistémique PURE — la force de l'évidence brute des groupes.
    #
    # Pourquoi u_raw et PAS u_sbn (post-UM) ?
    #   - L'UM (L6) augmente artificiellement u pour préserver P_proj. Elle est utile
    #     pour la COMMUNICATION de l'incertitude mais biaise u vers le haut pour tous.
    #   - u_raw = K / (e_total + K) mesure directement si les groupes ont fourni de
    #     l'évidence discriminante. C'est la bonne métrique de "nouveauté".
    #
    # Calibration analytique (K=11, scale=3.0) :
    #   u_raw = K / (Σ_g max(0, score-1/3)×scale + K)
    #   Extrêmes connus (_strong_anom, 9 groupes) : u_raw ≈ 0.458
    #   Faibles connus  (_weak_susp,   4 groupes) : u_raw ≈ 0.62-0.72
    #   Vraiment inconnu (scores ≈ 1/3 partout)   : u_raw → 1.0
    #
    # Seuil recommandé : 0.82
    #   → e_total < K × (1/0.82 - 1) ≈ 11 × 0.22 ≈ 2.4 unités d'évidence
    #   → Signifie : moins de ~2.4 "pas" d'évidence au-dessus de la neutralité
    #   → Typiquement déclenché pour vraiment inconnu, pas pour faible mais connu
    #
    # Ce seuil est CALIBRABLE via CONFIG['SBN_NOVELTY_U_RAW_THRESHOLD'].
    # Sa validation empirique sur RedeRio doit être faite et reportée.
    u_nov_thr = float(
        (autre_anomalie_prior if autre_anomalie_prior is not None else 0.82)
        # NB : on réutilise autre_anomalie_prior comme seuil numérique u_raw
        # si != None, sinon défaut 0.82. Ce paramètre est résolu dans run().
    )
    is_autre_anomalie = (u_raw > u_nov_thr)
    qual_status = 'autre_anomalie' if is_autre_anomalie else 'qualified'

    result['top1_type']  = top1
    result['top1_b']     = top1_b
    result['top1_proj']  = top1_proj   # P_proj(top1) = b_top + u/K — utile pour rapport
    result['qual_status'] = qual_status

    return result


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSE DE SENSIBILITÉ DES OPINIONS CONDITIONNELLES EXPERTES
# ──────────────────────────────────────────────────────────────────────────────

def _sensitivity_analysis(sbn_cond: dict, perturb: float = 0.05,
                          evidence_scale: float = 3.0) -> None:
    """
    Analyse de sensibilité des opinions conditionnelles expertes SBN_COND.

    Pour chaque type d'attaque k :
      1. Construit la signature théorique parfaite (group_pp = SBN_COND[k]).
      2. Calcule le top-1 de base (doit être k si la signature est discriminante).
      3. Perturbe chaque masse SBN_COND[k][groupe][état] de ±perturb (normalisé).
      4. Recompute le top-1 et signale si k n'est plus dominant.

    Objectif : vérifier que les opinions expertes sont robustes à une imprécision
    d'élicitation de ±perturb (typiquement 0.05).

    IMPORTANT : utiliser le même evidence_scale que dans run() pour que les résultats
    STABLE/INSTABLE correspondent au comportement réel du système.

    Ref : Cooke (1991) Experts in Uncertainty — méthode d'élicitation Bayésienne.


    Analyse de sensibilité des opinions SBN à ±0.05.

    ⚠️  LIMITE MÉTHODOLOGIQUE (documentée PATCH-m5, 2026-04-18) :
    Cette analyse perturbe SBN_COND_OPINIONS autour de sa valeur de référence,
    puis mesure la variation de _compute_group_projected par rapport à
    cette même référence. Elle mesure donc la **consistance interne** du
    mapping, pas la **robustesse au bruit réel** d'observation.

    Pour une vraie étude de robustesse, perturber les ÉVIDENCES d'entrée
    (ev_safe, ev_susp, ev_atk) par un bruit gaussien calibré sur la variance
    empirique des résidus — non fait actuellement (cf. SCIENTIFIC_AUDIT §1.5.J).
    """
    type_names = list(sbn_cond.keys())

    def _top1_with_cond(group_pp_ref: dict, test_cond: dict) -> str:
        scores = {}
        for type_k, cond_by_group in test_cond.items():
            if not isinstance(cond_by_group, dict):
                continue
            scores_k = [
                _sbn_group_score(group_pp_ref[grp], _normalize_cond_opinion(cond_raw))
                for grp, cond_raw in cond_by_group.items()
                if grp in group_pp_ref
            ]
            scores[type_k] = scores_k
        likes = {t: _evidence_sum_scores(sc, evidence_scale=evidence_scale)
                 for t, sc in scores.items()}
        b, _ = _sl_bijection(likes)
        return max(b, key=b.get)

    print(f"\n{'='*72}")
    print(f"  SENSIBILITÉ SBN_COND_OPINIONS (perturbation ±{perturb:.2f} sur chaque masse)")
    _n_groups = len(sbn_cond[type_names[0]]) if type_names else 0
    print(f"  {len(type_names)} types × {_n_groups} groupes (typ.)")
    print(f"  [OK] STABLE = top-1 inchange pour toutes perturbations de +/-{perturb:.2f}")
    print(f"  [!!] INSTABLE = au moins une perturbation change le top-1")
    print(f"{'='*72}")
    print(f"  {'Type':<28} {'Baseline top-1':>16} {'Instable/Total':>16} {'Statut':>10}")
    print(f"  {'-'*72}")

    all_stable = True
    for ref_type in type_names:
        # Signature parfaite : group_pp = SBN_COND[ref_type] normalisé
        group_pp_ref = {
            grp: _normalize_cond_opinion(sbn_cond[ref_type][grp])
            for grp in sbn_cond[ref_type]
            if isinstance(sbn_cond[ref_type][grp], dict)
        }

        top1_base = _top1_with_cond(group_pp_ref, sbn_cond)
        n_instable = 0
        n_total = 0
        examples = []

        for p_grp in list(sbn_cond[ref_type].keys()):
            if not isinstance(sbn_cond[ref_type][p_grp], dict):
                continue
            for p_state in STATES:
                for delta in [+perturb, -perturb]:
                    # Deep copy partielle : seule la distribution perturbée est copiée
                    sbn_p = {
                        t: {g: dict(m) for g, m in grps.items()}
                        for t, grps in sbn_cond.items()
                    }
                    old_val = sbn_p[ref_type][p_grp].get(p_state, 0.0)
                    sbn_p[ref_type][p_grp][p_state] = max(0.0, old_val + delta)
                    # Renormalisation du simplex perturbé
                    tot = sum(sbn_p[ref_type][p_grp].values())
                    if tot > EPS:
                        sbn_p[ref_type][p_grp] = {
                            s: v / tot for s, v in sbn_p[ref_type][p_grp].items()
                        }
                    top1_p = _top1_with_cond(group_pp_ref, sbn_p)
                    n_total += 1
                    if top1_p != top1_base:
                        n_instable += 1
                        if len(examples) < 3:
                            examples.append(
                                f"[{p_grp}/{p_state}{delta:+.2f}] -> {top1_p}"
                            )

        status = "✅ STABLE" if n_instable == 0 else "⚠️  INSTABLE"
        if n_instable > 0:
            all_stable = False
        print(f"  {ref_type:<28} {top1_base:>16} {n_instable:>6}/{n_total:<8}  {status}")
        for ex in examples:
            print(f"      -> {ex}")

    print(f"  {'-'*72}")
    if all_stable:
        print(f"  [OK] Toutes les signatures sont robustes a une perturbation de +/-{perturb:.2f}")
    else:
        print(f"  [!!] Certains types sont sensibles -- revoir les masses concernees")
    print(f"{'='*72}\n")


# ──────────────────────────────────────────────────────────────────────────────
# ÉVALUATION QUANTITATIVE : QUALIFICATION DE TYPE VS GROUND-TRUTH
# ──────────────────────────────────────────────────────────────────────────────

def _eval_type_performance(df_out: pd.DataFrame, df_input: pd.DataFrame) -> None:
    """
    Évalue la précision de la qualification de type par rapport au ground-truth.

    Cherche automatiquement une colonne de type d'attaque dans df_input parmi :
      ['injected_type', 'attack_type', 'true_type', 'label_type', 'true_label']

    Calcule pour chaque type : Précision, Rappel, F1, N.
    Affiche la matrice de confusion sur les types les plus fréquents.

    Ne prend en compte que les fenêtres avec gate_open=True ET une étiquette
    d'attaque non-vide (exclut 'Normal', 'NORMAL', '').
    """
    GT_CANDIDATES = ['injected_type', 'attack_type', 'true_type', 'label_type', 'true_label']
    gt_col = next((c for c in GT_CANDIDATES if c in df_input.columns), None)

    if gt_col is None:
        return  # Pas de ground-truth disponible — silencieux

    # Alignement par position (même CSV, même ordre de lignes)
    eval_df = pd.DataFrame({
        'gate_open': df_out['gate_open'].values,
        'top1_type': df_out['top1_type'].values,
        'gt':        df_input[gt_col].values,
    })

    NORMAL_LABELS = {'', 'normal', 'Normal', 'NORMAL', 'nan', 'None'}
    eval_df = eval_df[
        eval_df['gate_open'] &
        eval_df['gt'].notna() &
        (~eval_df['gt'].astype(str).isin(NORMAL_LABELS))
    ].copy()

    if len(eval_df) == 0:
        print(f"\n  [Éval type] Colonne '{gt_col}' trouvée mais aucune fenêtre "
              f"gate_open avec étiquette d'attaque.")
        return

    all_types = sorted(set(eval_df['gt'].unique()) | set(eval_df['top1_type'].unique()))

    print(f"\n{'='*72}")
    print(f"  ÉVALUATION QUALIFICATION DE TYPE  (N={len(eval_df)} fenêtres étiquetées)")
    print(f"  Colonne ground-truth : '{gt_col}'")
    print(f"{'='*72}")
    print(f"  {'Type GT':<28} {'Précision':>10} {'Rappel':>10} {'F1':>10} {'N GT':>6}")
    print(f"  {'-'*68}")

    for t in all_types:
        tp = int(((eval_df['gt'] == t) & (eval_df['top1_type'] == t)).sum())
        fp = int(((eval_df['gt'] != t) & (eval_df['top1_type'] == t)).sum())
        fn = int(((eval_df['gt'] == t) & (eval_df['top1_type'] != t)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        n_gt = int((eval_df['gt'] == t).sum())
        print(f"  {t:<28} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {n_gt:>6}")

    acc = float((eval_df['gt'] == eval_df['top1_type']).mean())
    print(f"  {'-'*68}")
    print(f"  {'Accuracy globale top-1':<28} {acc:>10.3f}")
    print(f"{'='*72}")

    # Matrice de confusion (5 types les plus fréquents dans GT)
    top_gt = eval_df['gt'].value_counts().head(5).index.tolist()
    top_pred = eval_df['top1_type'].value_counts().head(5).index.tolist()
    conf_types = sorted(set(top_gt) | set(top_pred))
    if len(conf_types) > 1:
        print(f"\n  Matrice de confusion (top types) — lignes=GT, colonnes=Prédit :")
        col_w = max(len(t) for t in conf_types) + 2
        header = f"  {'GT \\ Prédit':<{col_w}}" + "".join(f"{t:<{col_w}}" for t in conf_types)
        print(header)
        for gt_t in conf_types:
            row_s = f"  {gt_t:<{col_w}}"
            for pred_t in conf_types:
                n = int(((eval_df['gt'] == gt_t) & (eval_df['top1_type'] == pred_t)).sum())
                row_s += f"{n:<{col_w}}"
            print(row_s)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — TRAITEMENT DU CSV
# ──────────────────────────────────────────────────────────────────────────────

def run(
    input_csv:        str,
    output_csv:       str,
    threshold:        float,
    apply_temporal:   bool = True,
    apply_um:         bool = True,
    lambda_temporal:  float = 0.80,
    temporal_weight:  float = 0.30,
    novelty_threshold: float = 0.85,
    compare_csv:      str | None = None,
    run_sensitivity:  bool = False,
) -> pd.DataFrame:
    """
    Lance la qualification SBN sur tout le CSV et écrit le résultat.

    W est calculé dynamiquement W=K (nombre de types) dans _sl_bijection
    (Jøsang §3.5.2 — garantit u=1.0 à évidence nulle et prior uniforme).

    Paramètres :
        apply_temporal    : active le prior temporel Markovien (L5, WBF + transition)
        apply_um          : active l'Uncertainty Maximisation (L6)
        lambda_temporal   : facteur de décroissance temporelle λ (Jøsang Déf. 14.6).
                            COHÉRENCE SYSTÈME : aligner sur LAMBDA_DECAY de config.py
                            (utilisé dans compute_opinions_v3.py pour l'ageing temporel
                            des métriques Prophet). Les deux paramètres contrôlent la
                            "mémoire temporelle" du système à des niveaux différents :
                            - LAMBDA_DECAY (compute_opinions_v3.py) : ageing intra-détection
                              (lissage des opinions métriques Prophet au sein d'une série)
                            - lambda_temporal (qualify_anomaly_sbn.py) : persistance inter-fenêtres
                              de la classification d'attaque (prior temporel Markovien).
                            Ces deux paramètres peuvent être égaux (mémoire système uniforme)
                            ou différents (détection plus réactive que qualification).
        temporal_weight   : poids fixe du prior temporel dans WBF (indépendant de λ)
        novelty_threshold : seuil novelty_lr pour stats diagnostiques (défaut 0.85)
        compare_csv       : CSV qualify_anomaly.py (heuristique) pour comparaison
        run_sensitivity   : si True, exécute l'analyse de sensibilité SBN_COND
    """
    # ── Validation des paramètres numériques ──────────────────────────────────
    if not (0 < lambda_temporal <= 1.0):
        raise ValueError(f"lambda_temporal doit être dans (0, 1] : {lambda_temporal}")
    if not (0 <= temporal_weight <= 1.0):
        raise ValueError(f"temporal_weight doit être dans [0, 1] : {temporal_weight}")

    _g = globals()
    _cfg = _g.get('CONFIG', {})

    # SBN_TEMPORAL_ENABLED : la config peut forcer le prior temporel indépendamment du CLI
    apply_temporal = apply_temporal or bool(_cfg.get('SBN_TEMPORAL_ENABLED', False))

    # SBN_EVIDENCE_SCALE : passé explicitement à sbn_qualify_row (évite tout accès global)
    evidence_scale = float(_cfg.get('SBN_EVIDENCE_SCALE', 3.0))

    # Résolution des SBN_COND_OPINIONS (config > fallback)
    sbn_cond = _cfg.get('SBN_COND_OPINIONS', _DEFAULT_SBN_COND) if _cfg else _DEFAULT_SBN_COND
    if not sbn_cond:
        sbn_cond = _DEFAULT_SBN_COND

    # ── SBN_NOVELTY_U_RAW_THRESHOLD : seuil u_raw pour la détection de nouveauté ──
    # u_raw est calculé AVANT l'Uncertainty Maximisation (UM, L6) et avant le prior
    # temporel (L5). Il mesure l'incertitude épistémique pure des groupes de métriques.
    #
    # Interprétation pour K=11, scale=3.0 :
    #   u_raw ≈ 0.46 : 9 groupes _strong_anom → attaque parfaitement discriminée
    #   u_raw ≈ 0.60-0.70 : signal modéré (DATA_EXFIL low, SLOWLORIS low)
    #   u_raw > 0.82 : < 2.4 unités d'évidence totale → signal vraiment indiscriminé
    #   u_raw → 1.0 : aucune évidence → anomalie totalement inconnue
    #
    # Valeur par défaut = 0.82 (à recalibrer empiriquement sur RedeRio avec injections).
    # Pour désactiver la détection de nouveauté : mettre à 1.0.
    _u_nov_raw = _cfg.get('SBN_NOVELTY_U_RAW_THRESHOLD', 0.82)
    autre_anomalie_prior = float(_u_nov_raw)   # passé à sbn_qualify_row comme u_raw threshold

    # Vérification GROUP_SOURCES non vide — si vide, toutes les fenêtres retournent
    # u_sbn=1.0 sans classification (échec silencieux sinon).
    if not GROUP_SOURCES:
        raise RuntimeError(
            "[SBN] GROUP_SOURCES est vide — vérifier QUALIFY_GROUP_SOURCES dans config.py. "
            "Sans groupes définis, aucune qualification de type n'est possible."
        )

    type_names = list(sbn_cond.keys())
    K = len(type_names)

    # ── Validation config partielle de SBN_COND_OPINIONS ─────────────────────
    # Si SBN_COND_OPINIONS est défini dans config.py mais ne contient qu'un sous-ensemble
    # des types de _DEFAULT_SBN_COND, K est réduit et u_min change significativement.
    if sbn_cond is not _DEFAULT_SBN_COND:
        _expected = set(_DEFAULT_SBN_COND.keys())
        _provided = set(sbn_cond.keys())
        _missing  = _expected - _provided
        if _missing:
            # Impact sur u_min : u_min = K / (e_max + K)
            # avec e_max ≈ 9 × (2/3) × evidence_scale (9 groupes parfaits)
            _e_max = 9 * (2.0 / 3.0) * evidence_scale
            _u_min_expected = len(_expected) / (_e_max + len(_expected))
            _u_min_actual   = K            / (_e_max + K)
            print(f"\n  [WARN] SBN_COND_OPINIONS PARTIEL : {len(_missing)} types manquants "
                  f"vs _DEFAULT_SBN_COND.")
            print(f"  [WARN] Types manquants : {sorted(_missing)}")
            print(f"  [WARN] K effectif = {K} au lieu de {len(_expected)}")
            print(f"  [WARN] Impact sur u_min (9 groupes parfaits) : "
                  f"{_u_min_expected:.3f} → {_u_min_actual:.3f}")
            print(f"  [WARN] La matrice de transition est reconstruite sur {K} types "
                  f"(les {len(_missing)} types manquants sont ignorés).")

    # Vérification cohérence GROUP_SOURCES ↔ SBN_COND_OPINIONS
    for tk in sbn_cond:
        missing_g = set(sbn_cond[tk].keys()) - set(GROUP_SOURCES.keys())
        if missing_g:
            print(f"  [WARN] Type '{tk}' : groupes dans SBN_COND non définis "
                  f"dans GROUP_SOURCES : {missing_g}")

    # Construction matrice de transition Markovienne (L5)
    transition_matrix = _build_transition_matrix(type_names) if apply_temporal else None

    print(f"[SBN] {K} types (W={K} dynamique) : {type_names}")
    print(f"  Groupes : {list(GROUP_SOURCES.keys())}")
    print(f"  Gate seuil={threshold:.3f}  UM={'on' if apply_um else 'off'}  "
          f"Temporal={'on (lambda={lambda_temporal}, w={temporal_weight})' if apply_temporal else 'off'}")
    print(f"  Nouveauté : qual_status=autre_anomalie si u_raw > {autre_anomalie_prior:.3f}  "
          f"(SBN_NOVELTY_U_RAW_THRESHOLD — recalibrer sur données réelles)")
    print(f"  NETWORK_OUTAGE : qualification SBN native (sans gate separee)")

    print(f"\n[SBN] Lecture : {input_csv}")
    df = pd.read_csv(input_csv, parse_dates=['timestamp'])
    print(f"  {len(df)} fenêtres, {len(df.columns)} colonnes")

    # Vérification colonnes obligatoires
    required_cols = list({_DET_COL, 'FINAL_SYSTEM_CBF_proj_atk', 'FINAL_SYSTEM_CBF_u'})
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    # Warnings colonnes manquantes par groupe
    for grp, srcs in GROUP_SOURCES.items():
        for src in srcs:
            for suf in STATE_COLS.values():
                col = f"{src}_{suf}"
                if col not in df.columns:
                    print(f"  [WARN] Colonne manquante : {col} (groupe {grp})")

    # ── Qualification ligne par ligne avec prior temporel ──────────────────
    print(f"\n[SBN] Qualification en cours ...")
    results_list = []
    prev_opinion  = None   # opinion t-1
    prev_gate     = False
    delta_windows = 1

    for i, (_, row) in enumerate(df.iterrows()):
        r = sbn_qualify_row(
            row                  = row,
            sbn_cond             = sbn_cond,
            threshold            = threshold,
            prev_opinion         = prev_opinion if apply_temporal else None,
            delta_windows        = delta_windows,
            lambda_temporal      = lambda_temporal,
            temporal_weight      = temporal_weight,
            apply_temporal       = apply_temporal,
            apply_um             = apply_um,
            transition_matrix    = transition_matrix,
            type_names_tm        = type_names if apply_temporal else None,
            evidence_scale       = evidence_scale,
            autre_anomalie_prior = autre_anomalie_prior,
        )
        results_list.append(r)

        # Mise à jour prior temporel
        if r['gate_open']:
            # Temporairement, extraire l'opinion depuis r
            b_dict = {k: r.get(f'b_sbn_{k}', 0.0) for k in type_names}
            b_dict['Autre_Anomalie'] = r.get('b_sbn_Autre_Anomalie', 0.0)
            prev_opinion  = {'b': b_dict, 'u': r['u_sbn']}
            delta_windows = 1
            prev_gate     = True
        elif prev_gate:
            delta_windows += 1
            # Pas de reset de prev_opinion — on continue d'escompter
        else:
            prev_opinion  = None
            delta_windows = 1
            prev_gate     = False

    df_qualif = pd.DataFrame(results_list)

    # ── Concaténation avec timestamp ──────────────────────────────────────
    df_out = pd.concat([df[['timestamp']], df_qualif], axis=1)

    # ── Stats rapides ─────────────────────────────────────────────────────
    n_triggered = df_qualif['gate_open'].sum()
    print(f"  Gate ouverte : {n_triggered}/{len(df)} fenêtres "
          f"({100 * n_triggered / len(df):.1f}%)")

    # Breakdown qual_status (toutes fenêtres)
    if 'qual_status' in df_qualif.columns:
        qs_counts = df_qualif['qual_status'].value_counts()
        print(f"  Statut qualification :")
        for qs, cnt in qs_counts.items():
            print(f"    {qs:<20} : {cnt:>6} ({100*cnt/len(df_qualif):>5.1f}%)")

    if n_triggered > 0:
        df_anom = df_qualif[df_qualif['gate_open']]
        # Exclure fenêtres 'no_groups' des stats de type (u_sbn=1.0 par manque de données)
        df_qual_only = df_anom[df_anom.get('qual_status', 'qualified') != 'no_groups'] \
            if 'qual_status' in df_anom.columns else df_anom

        top1_counts = df_qual_only['top1_type'].value_counts()
        u_mean = df_qual_only['u_sbn'].mean() if len(df_qual_only) > 0 else float('nan')
        _lr_thr = _cfg.get('SBN_LR_NOVELTY_THRESHOLD', novelty_threshold) if _cfg else novelty_threshold
        n_novelty = (df_qual_only['novelty_lr'] > _lr_thr).sum() if len(df_qual_only) > 0 else 0

        b_cols = sorted(
            [c for c in df_qual_only.columns if c.startswith('b_sbn_') and '_raw_' not in c],
            key=lambda c: -df_qual_only[c].mean()
        )

        n_qual = len(df_qual_only)
        print(f"\n  Opinion SBN (fenêtres qualifiées, N={n_qual}) :")
        print(f"  {'Type':<28} {'b_sbn (moy)':>12} {'top1 (nb)':>10} {'top1 (%)':>9}")
        print(f"  {'-'*63}")
        for col in b_cols:
            tname = col.replace('b_sbn_', '')
            b_mean = df_qual_only[col].mean() if n_qual > 0 else 0.0
            n_top1 = top1_counts.get(tname, 0)
            denom  = max(n_qual, 1)
            marker = " [AUTRE]" if tname == 'Autre_Anomalie' else ""
            print(f"  {tname:<28} {b_mean:>12.4f} {n_top1:>10} "
                  f"{100*n_top1/denom:>8.1f}%{marker}")
        print(f"  {'-'*63}")
        print(f"  {'u_sbn (incertitude)':<28} {u_mean:>12.4f}")
        print(f"  Signal nouveauté (lr_nov>{_lr_thr:.2f})  : {n_novelty:>6} "
              f"({100 * n_novelty / max(n_qual, 1):>5.1f}%)")

        # Vérification contrainte SL Σb + u = 1 sur les fenêtres qualifiées
        # (exclure 'no_groups' où u_sbn=1.0 et b=0 → sum=1 trivial)
        b_sum = df_qual_only[[c for c in b_cols]].sum(axis=1)
        sl_errors = (b_sum + df_qual_only['u_sbn'] - 1.0).abs()
        print(f"\n  Contrainte SL  Σb + u = 1 : "
              f"max_err={sl_errors.max():.2e}  mean_err={sl_errors.mean():.2e}  "
              f"(objectif : max_err < 1e-6)")

    # ── Analyse de sensibilité SBN_COND (optionnelle) ─────────────────────
    if run_sensitivity:
        _sensitivity_analysis(sbn_cond, evidence_scale=evidence_scale)

    # ── Évaluation quantitative type vs ground-truth (si dispo) ───────────
    _eval_type_performance(df_out, df)

    # ── Comparaison avec qualify_anomaly.py si demandée ───────────────────
    if compare_csv and os.path.exists(compare_csv):
        _compare_outputs(df_out, compare_csv, n_triggered)

    df_out.to_csv(output_csv, index=False)
    print(f"\n[OK] CSV SBN sauvegarde : {output_csv}")
    return df_out


def _compare_outputs(df_sbn: pd.DataFrame, old_csv: str, n_total: int):
    """Compare les résultats SBN vs qualify_anomaly.py (heuristique LR)."""
    try:
        df_old = pd.read_csv(old_csv, parse_dates=['timestamp'])
    except Exception:
        return

    # Merge sur timestamp — normalisation à la seconde pour éviter les échecs sur
    # différences de précision entre les deux CSV (ex. "08:00:00" vs "08:00:00.000").
    df_sbn = df_sbn.copy()
    df_old = df_old.copy()
    df_sbn['timestamp'] = pd.to_datetime(df_sbn['timestamp']).dt.floor('s')
    df_old['timestamp']  = pd.to_datetime(df_old['timestamp']).dt.floor('s')
    merged = df_sbn.merge(df_old[['timestamp', 'top1_type', 'gate_open']],
                          on='timestamp', suffixes=('_sbn', '_old'), how='inner')
    both_open = merged[(merged['gate_open_sbn']) & (merged['gate_open_old'])]
    if len(both_open) == 0:
        print(f"\n  [Comparaison] Aucune fenêtre commune gate_open.")
        return

    n_agree  = (both_open['top1_type_sbn'] == both_open['top1_type_old']).sum()
    n_total_c = len(both_open)
    print(f"\n  [Comparaison SBN vs LR heuristique]")
    print(f"    Fenêtres avec gate_open dans les deux : {n_total_c}")
    print(f"    Accord top-1     : {n_agree}/{n_total_c} ({100*n_agree/n_total_c:.1f}%)")

    disagree = both_open[both_open['top1_type_sbn'] != both_open['top1_type_old']]
    if len(disagree) > 0:
        print(f"    Désaccords :")
        for _, r in disagree.head(10).iterrows():
            print(f"      {r['timestamp']}  SBN={r['top1_type_sbn']:20s}  LR={r['top1_type_old']}")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('SL-template qualifier (legacy SBN output names; '
                     'not a canonical Subjective Bayesian Network)'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input',     default=None,
                        help='CSV de compute_opinions_v3.py (detection_results*.csv)')
    parser.add_argument('--output',    default=None,
                        help='CSV de sortie (défaut : qualif_types_sbn.csv)')
    parser.add_argument('--threshold', type=float, default=_THRESHOLD,
                        help=f'Seuil gate P(Anom) δ')
    parser.add_argument('--temporal',  action='store_true', default=False,
                        help='Active le prior temporel Markovien (L5 : WBF + matrice '
                             'de transition kill chain). Recommandé pour attaques '
                             'persistantes. Désactivé par défaut pour reproductibilité.')
    parser.add_argument('--sensitivity', action='store_true', default=False,
                        help='Exécute l\'analyse de sensibilité SBN_COND (perturbation ±0.05)')
    parser.add_argument('--no_um',     action='store_true', default=False,
                        help='Désactive l\'Uncertainty Maximisation post-fusion')
    parser.add_argument('--lambda_t',  type=float, default=0.80,
                        help='Facteur de décroissance temporel λ')
    parser.add_argument('--w_temp',    type=float, default=0.30,
                        help='Poids maximal du prior temporel dans WBF')
    parser.add_argument('--novelty',   type=float, default=0.85,
                        help='Seuil novelty_lr pour signal de nouveauté (défaut=0.85, '
                             'calibré sur signatures théoriques : connu→0.47-0.62, inconnu→0.996)')
    parser.add_argument('--compare',   default=None, metavar='OLD_CSV',
                        help='CSV de qualify_anomaly.py pour comparaison')
    parser.add_argument('--balance_ratio', default=None,
                        help='Variante balance_ratio (ex: "auto")')
    args = parser.parse_args()

    # Résolution des chemins
    _bal_suffix = f'_balance_{args.balance_ratio}' if args.balance_ratio else ''

    if args.input is None:
        candidates = [
            os.path.join(_OUTPUT_DIR, f'detection_results_INJECTED{_bal_suffix}.csv'),
            os.path.join(_OUTPUT_DIR, 'detection_results_INJECTED.csv'),
            os.path.join(_OUTPUT_DIR, 'detection_results.csv'),
        ]
        for c in candidates:
            if os.path.exists(c):
                args.input = c
                break
        if args.input is None:
            raise FileNotFoundError(
                f"Aucun CSV trouvé dans {_OUTPUT_DIR}. Passe --input <chemin>.")

    if args.output is None:
        base_dir   = os.path.dirname(args.input)
        args.output = os.path.join(base_dir, f'qualif_types_sbn{_bal_suffix}.csv')

    run(
        input_csv         = args.input,
        output_csv        = args.output,
        threshold         = args.threshold,
        apply_temporal    = args.temporal,
        apply_um          = not args.no_um,
        lambda_temporal   = args.lambda_t,
        temporal_weight   = args.w_temp,
        novelty_threshold = args.novelty,
        compare_csv       = args.compare,
        run_sensitivity   = args.sensitivity,
    )
