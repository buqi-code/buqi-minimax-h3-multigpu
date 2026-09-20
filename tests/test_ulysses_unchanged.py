"""Guard against accidentally rewriting the author's Ulysses collective math."""
import ast
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = "bce083929c135bdabd41679da3ddf6f409336ed3"
NAMES = ("SPContext", "post_heads_to_seq", "a2a_scatter_seq_gather_heads",
         "attention_and_exchange", "sp_attention", "sp_block")


class UlyssesPreservationTests(unittest.TestCase):
    def test_author_algorithm_unchanged(self):
        original = subprocess.check_output(["git", "-C", str(ROOT), "show", BASE + ":minimax_sp/sp_forward.py"], text=True)
        current = (ROOT / "minimax_sp/sp_forward.py").read_text()
        def functions(source):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        before, after = functions(original), functions(current)
        for name in NAMES:
            with self.subTest(function=name):
                self.assertEqual(before[name], after[name])


if __name__ == "__main__":
    unittest.main()
