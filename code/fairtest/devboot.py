"""Import bootstrap for local development runs.

`src.environments.__init__` eagerly imports torch-based baselines. Local Windows torch
wheels may fail to load, and the simulator path does not need torch. Registering a bare
package module lets submodules import normally without executing that __init__.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if 'src.environments' not in sys.modules:
    import src  # noqa: F401
    pkg = types.ModuleType('src.environments')
    pkg.__path__ = [str(ROOT / 'src' / 'environments')]
    pkg.__package__ = 'src.environments'
    sys.modules['src.environments'] = pkg
