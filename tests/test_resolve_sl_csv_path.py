"""
test_resolve_sl_csv_path.py — Tests pour _resolve_sl_csv_path (PATCH-C6).

Vérifie que :
1. Le typo "dadza" est bien corrigé (clé RESULTS_CSV_NAME respectée)
2. La variante _INJECTED est essayée en premier
3. Un alias custom dans CONFIG["EVAL"]["RESULTS_CSV_NAME"] est respecté
"""
import os
import sys
import tempfile
import unittest
import shutil

# Ajuster le sys.path pour importer depuis le parent
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)


class TestResolveSlCsvPath(unittest.TestCase):

    def setUp(self):
        # Créer un répertoire temporaire pour les CSVs de test
        self.tmpdir = tempfile.mkdtemp(prefix="test_resolve_sl_")
        # Phase H: import the new package path directly (the legacy
        # ``import compare_if_fair`` is now a deprecation shim that drops
        # underscore-prefixed names like ``_resolve_sl_csv_path``).
        import sl_ads.compare.compare_if_fair as compare_if_fair
        self._orig_cfg = compare_if_fair.CONFIG
        self._module = compare_if_fair

    def tearDown(self):
        # Restaurer CONFIG
        self._module.CONFIG = self._orig_cfg
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _patch_config(self, results_dir, csv_name):
        import copy
        cfg = copy.deepcopy(self._orig_cfg)
        cfg.setdefault("EVAL", {})
        cfg["EVAL"]["RESULTS_DIR"] = results_dir
        cfg["EVAL"]["RESULTS_CSV_NAME"] = csv_name
        # VERSION_NAME unique pour éviter de tomber sur un vrai dossier d'artefacts
        cfg["VERSION_NAME"] = "__TEST_UNIQUE_XYZ_DO_NOT_EXIST__"
        self._module.CONFIG = cfg

    def test_typo_dadza_fixed_key_respected(self):
        """Le typo 'dadza' est corrigé : la clé RESULTS_CSV_NAME doit être lue."""
        csv_path = os.path.join(self.tmpdir, "my_custom_detection.csv")
        with open(csv_path, "w") as f:
            f.write("timestamp,score\n")
        self._patch_config(self.tmpdir, "my_custom_detection.csv")

        resolved = self._module._resolve_sl_csv_path("")
        self.assertEqual(os.path.abspath(resolved), os.path.abspath(csv_path))

    def test_injected_variant_preferred(self):
        """Si _INJECTED existe, il est préféré au non-injecté."""
        base_csv = os.path.join(self.tmpdir, "detection_results.csv")
        inj_csv = os.path.join(self.tmpdir, "detection_results_INJECTED.csv")
        with open(base_csv, "w") as f:
            f.write("timestamp,score\n")
        with open(inj_csv, "w") as f:
            f.write("timestamp,score\n")
        self._patch_config(self.tmpdir, "detection_results.csv")

        resolved = self._module._resolve_sl_csv_path("")
        self.assertEqual(os.path.abspath(resolved), os.path.abspath(inj_csv))

    def test_fallback_to_base_when_injected_missing(self):
        """Si _INJECTED n'existe pas, on retombe sur le non-injecté."""
        base_csv = os.path.join(self.tmpdir, "detection_results.csv")
        with open(base_csv, "w") as f:
            f.write("timestamp,score\n")
        self._patch_config(self.tmpdir, "detection_results.csv")

        resolved = self._module._resolve_sl_csv_path("")
        self.assertEqual(os.path.abspath(resolved), os.path.abspath(base_csv))

    def test_requested_path_priority(self):
        """Un chemin explicite est respecté même si d'autres existent."""
        explicit = os.path.join(self.tmpdir, "explicit.csv")
        with open(explicit, "w") as f:
            f.write("timestamp,score\n")
        resolved = self._module._resolve_sl_csv_path(explicit)
        self.assertEqual(os.path.abspath(resolved), os.path.abspath(explicit))


if __name__ == "__main__":
    unittest.main(verbosity=2)
