"""inspect_qualify_types_during_attacks — ad-hoc inspection script.

Originally lived at the project root as ``test.py``.  Moved to
``investigations/`` during the Phase H reorganisation (2026-04-29)
and renamed to reflect what it actually does: load
``qualif_types.csv`` and pretty-print, for each known attack window,
the number of gate-open windows, mean ``u_qualif``, mean ``top1_b``
and the dominant ``top1_type``.

Run as::

    PYTHONPATH=src python investigations/inspect_qualify_types_during_attacks.py
"""
import pandas as pd
from sl_ads.config import CONFIG  # Phase H
from sl_ads.paths import get_version_names, get_results_dir  # Phase H


_version_name, _ = get_version_names(CONFIG)
_results_dir = get_results_dir(CONFIG, up_levels=1)
df = pd.read_csv(f'{_results_dir}/qualif_types.csv',
                 parse_dates=['timestamp'])


attacks = [
    ('UDP_FLOOD',  '2025-11-16 14:00:00', '2025-11-16 18:00:00'),
    ('SYN_FLOOD',  '2025-11-21 02:30:00', '2025-11-21 03:15:00'),
    ('PORT_SCAN',  '2025-11-28 10:15:00', '2025-11-28 12:45:00'),
    ('DATA_EXFIL', '2025-12-02 23:00:00', '2025-12-03 05:00:00'),
    ('HTTP_FLOOD', '2025-12-07 16:30:00', '2025-12-07 18:00:00'),
    ('DNS_AMP',    '2025-12-11 08:00:00', '2025-12-11 11:00:00'),
    ('SLOWLORIS',  '2025-12-15 22:00:00', '2025-12-16 06:00:00'),
    ('ICMP_FLOOD', '2025-12-18 11:30:00', '2025-12-18 12:00:00'),
    ('UNKNOWN',    '2025-12-20 10:00:00', '2025-12-20 12:00:00'),
    ('REAL_DDOS',  '2025-11-12 18:21:13', '2025-11-13 10:14:36'),
]

print(f"{'Attaque':<15} {'N_open':>7} {'u_qualif':>10} {'b_top1':>8} {'top1_type'}")
print("─" * 65)
for name, s, e in attacks:
    mask = ((df['timestamp'] >= pd.Timestamp(s)) &
            (df['timestamp'] <= pd.Timestamp(e)) &
            (df['gate_open'] == True))
    d = df[mask]
    if len(d) == 0:
        print(f"  {name:<13} {'0':>7}")
        continue
    u_mean  = d['u_qualif'].mean()
    b_mean  = d['top1_b'].mean()
    top1    = d['top1_type'].value_counts().index[0] if len(d) else '—'
    marker  = " ← INCONNU" if name == 'UNKNOWN' else ""
    print(f"  {name:<13} {len(d):>7} {u_mean:>10.4f} {b_mean:>8.4f}  {top1}{marker}")
