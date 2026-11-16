"""Exact B5 runtime validation for the registered lab Actions container.

Location/context checks prevent accidental local execution. Environment variables
and container markers are operational guards, not security authentication.
Scientific inputs, Python package versions and external Python source hashes are
verified separately and are never weakened by the selected execution location.
"""
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform


ROOT = Path(__file__).resolve().parents[2]
ACTIONS_ROOT = '/results/fairtest-b5-runtime'
PINNED_VERSIONS = {
    'numpy': '2.4.4',
    'torch': '2.11.0',
    'gymnasium': '1.2.3',
    'stable-baselines3': '2.8.0',
    'highway-env': '1.10.2',
}


def location_profile(resolved_root, system, environment, *, docker_present, results_present):
    """Pure operational context check; ``resolved_root`` must be resolved first."""
    root = str(resolved_root)
    if system != 'Linux':
        raise RuntimeError('Actual B native execution requires the registered Linux research runtime')
    run_id = environment.get('TUPA_GITHUB_RUN_ID', '')
    actions = (root == ACTIONS_ROOT
        and docker_present and results_present
        and environment.get('TUPA_SOURCE_REPOSITORY') == '<GPU_FLEET_REPO>'
        and isinstance(run_id, str) and run_id.isascii() and run_id.isdecimal()
        and environment.get('TUPA_RESULTS_DIR') == '/results')
    if not actions:
        raise RuntimeError('Actual B native execution requires the registered lab Actions context')
    return 'lab_github_actions_container'


def assert_native_location():
    return location_profile(ROOT.resolve(), platform.system(), os.environ,
                            docker_present=Path('/.dockerenv').exists(),
                            results_present=Path('/results').is_dir())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_runtime(data):
    """Retain the old exact source/data/package/external-source verifier semantics."""
    assert_native_location()
    data = Path(data)
    manifest = json.loads((data/'runtime_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['source_hashes'].items():
        if sha(ROOT/name) != expected:
            raise RuntimeError('Runtime source drift: '+name)
    for name, expected in manifest['data_hashes'].items():
        if sha(data/name) != expected:
            raise RuntimeError('Runtime data drift: '+name)
    if manifest['expected_versions'] != PINNED_VERSIONS:
        raise RuntimeError('Frozen B runtime version contract drift')
    versions = {name: importlib.metadata.version(name) for name in manifest['expected_versions']}
    if versions != manifest['expected_versions']:
        raise RuntimeError('Installed package version drift: '+str(versions))
    for package, files in manifest['external_python'].items():
        root = Path(importlib.import_module(package).__file__).resolve().parent
        actual = {f.relative_to(root).as_posix(): sha(f) for f in sorted(root.rglob('*.py'))}
        if actual != files:
            raise RuntimeError('External package source drift: '+package)
    return manifest
