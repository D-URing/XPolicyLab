"""Download a pinned private delivery; verify before any installation/execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_inventory(path, expected):
    if not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError('Expected lowercase SHA-256 from the authorized handoff')
    if sha256(path) != expected:
        raise ValueError('Delivery inventory hash mismatch')
    def unique_pairs(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError('Duplicate inventory key')
            result[name] = value
        return result
    inventory = json.loads(Path(path).read_text(), object_pairs_hook=unique_pairs)
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError('Empty or invalid inventory')
    for name, digest in inventory.items():
        path_name = PurePosixPath(name)
        if (path_name.is_absolute() or '..' in path_name.parts or
                str(path_name) != name or '\\' in name or
                any(c in name for c in '*?[]') or name.startswith('.')):
            raise ValueError('Unsafe inventory path')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Invalid file hash')
    required = {'model/runtime.json', 'model/joint/manifest.json',
                'model/joint/action.pt', 'model/joint/planner/adapter_model.safetensors',
                'install_runtime.py', 'launch.py', 'verify.py', 'PRIVATE_DELIVERY.txt',
                'runtime/mxi-runtime-linux-x86_64.tar.gz'}
    if not required <= inventory.keys():
        raise ValueError('Incomplete joint-model/runtime delivery inventory')
    return inventory


def download(repo_id, revision, expected, output, workers=4):
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Pin an exact 40-character commit, not main or a tag')
    if not 1 <= workers <= 16:
        raise ValueError('workers must be between 1 and 16')
    from huggingface_hub import hf_hub_download
    output = Path(output).absolute()
    # Dedicated destination only. SDK-managed partial downloads can be resumed
    # only for the same repo, commit and independently provided inventory hash.
    identity = dict(repo_id=repo_id, revision=revision, inventory_sha256=expected)
    marker = output / '.mxi-download.json'
    if output.is_symlink():
        raise ValueError('Refusing symlink destination')
    output.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        if marker.is_symlink() or json.loads(marker.read_text()) != identity:
            raise ValueError('Destination belongs to a different delivery')
    else:
        if any(output.iterdir()):
            raise ValueError('Use a new empty destination')
        with marker.open('x') as stream:
            json.dump(identity, stream)
    if any(p.is_symlink() for p in output.rglob('*')):
        raise ValueError('Refusing symlinks in destination')
    def fetch(name):
        return hf_hub_download(repo_id=repo_id, filename=name, revision=revision,
                               repo_type='model', local_dir=str(output), token=True)
    inventory = read_inventory(fetch('DELIVERY_SHA256.json'), expected)
    def fetch_verified(item):
        name, digest = item
        path = Path(fetch(name))
        if path.is_symlink() or sha256(path) != digest:
            raise ValueError('Delivery file hash mismatch')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch_verified, inventory.items()))
    return len(inventory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--inventory-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    try:
        count = download(args.repo_id, args.revision, args.inventory_sha256,
                         args.output, args.workers)
    except Exception as exc:
        # SDK errors may contain signed URLs. Do not print exception payloads.
        raise SystemExit(f'Delivery failed ({type(exc).__name__}); do not install. '
                         'Check login/access, pinned identity, disk and file hashes. '
                         'Retry the identical command to resume.') from None
    print(f'PASS: {count} delivery files verified; nothing installed or executed')


if __name__ == '__main__':
    main()
