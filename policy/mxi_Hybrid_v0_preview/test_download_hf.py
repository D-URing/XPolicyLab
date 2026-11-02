"""Offline download plumbing tests; not live Hub transfer or inference evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('download_hf', Path(__file__).with_name('download_hf.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        names = ['model/runtime.json', 'model/joint/manifest.json', 'model/joint/action.pt',
                 'model/joint/planner/adapter_model.safetensors', 'install_runtime.py',
                 'launch.py', 'verify.py', 'PRIVATE_DELIVERY.txt',
                 'runtime/mxi-runtime-linux-x86_64.tar.gz']
        self.files = {name: b'offline-test-only' for name in names}
        self.inventory = {name: hashlib.sha256(data).hexdigest() for name, data in self.files.items()}
        self.set_inventory()

    def set_inventory(self):
        self.files['DELIVERY_SHA256.json'] = json.dumps(self.inventory).encode()
        self.expected = hashlib.sha256(self.files['DELIVERY_SHA256.json']).hexdigest()

    def fetch(self, **kwargs):
        self.assertEqual(kwargs['revision'], 'a' * 40)
        self.assertIs(kwargs['token'], True)
        path = Path(kwargs['local_dir']) / kwargs['filename']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[kwargs['filename']])
        return str(path)

    def run_download(self, revision='a' * 40, expected=None):
        with patch.dict('sys.modules', {'huggingface_hub': types.SimpleNamespace(hf_hub_download=self.fetch)}):
            return module.download('test/private', revision, expected or self.expected, self.root / 'out', 2)

    def test_verified_and_same_identity_resume(self):
        self.assertEqual(self.run_download(), 9)
        self.assertEqual(self.run_download(), 9)

    def test_branch_refused(self):
        with self.assertRaises(ValueError): self.run_download(revision='main')

    def test_wrong_inventory_hash(self):
        with self.assertRaisesRegex(ValueError, 'inventory hash'): self.run_download(expected='0' * 64)

    def test_tampered_file(self):
        self.files['model/joint/action.pt'] = b'wrong'
        with self.assertRaisesRegex(ValueError, 'file hash'): self.run_download()

    def test_path_traversal(self):
        self.inventory['../escape'] = '0' * 64
        self.set_inventory()
        with self.assertRaisesRegex(ValueError, 'Unsafe'): self.run_download()

    def test_missing_planner(self):
        del self.inventory['model/joint/planner/adapter_model.safetensors']
        self.set_inventory()
        with self.assertRaisesRegex(ValueError, 'Incomplete'): self.run_download()

    def test_no_mixed_revision_resume(self):
        self.run_download()
        with self.assertRaisesRegex(ValueError, 'different delivery'):
            self.run_download(revision='b' * 40)

    def test_no_unowned_directory(self):
        (self.root / 'out').mkdir()
        (self.root / 'out/unrelated').touch()
        with self.assertRaisesRegex(ValueError, 'empty destination'): self.run_download()


if __name__ == '__main__':
    unittest.main()
