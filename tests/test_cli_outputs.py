"""CLI output-directory handling for the existing simulator entry point."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestSafetySimOutputDir(unittest.TestCase):
    def test_plots_dir_is_created(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / 'does' / 'not' / 'exist'
            r = subprocess.run([sys.executable, str(ROOT / 'src' / 'safety_sim.py'), '--n', '1', '--plots-dir', str(out)],
                               capture_output=True, text=True, timeout=300,
                               env={**__import__('os').environ, 'MPLBACKEND': 'Agg'})
            self.assertEqual(r.returncode, 0, r.stderr[-500:])
            self.assertTrue((out / 'safety_sim.json').is_file())
            self.assertTrue((out / 'safety_sim.png').is_file())


if __name__ == '__main__':
    unittest.main()
