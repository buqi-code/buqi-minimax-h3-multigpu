"""Exercise the same package-style import ComfyUI uses for custom-node folders."""

import importlib.util
from pathlib import Path
import sys
import unittest

from bootstrap import setup
setup()


class PackageDiscoveryTests(unittest.TestCase):
    def test_root_package_exports_comfy_entrypoint(self):
        root = Path(__file__).resolve().parents[1]
        module_name = "custom_nodes.buqi_minimax_h3_multigpu"
        spec = importlib.util.spec_from_file_location(
            module_name,
            root / "__init__.py",
            submodule_search_locations=[str(root)],
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            self.assertTrue(callable(module.comfy_entrypoint))
            self.assertEqual(module.comfy_entrypoint.__module__, f"{module_name}.minimax_sp")
        finally:
            for name in tuple(sys.modules):
                if name == module_name or name.startswith(module_name + "."):
                    sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
