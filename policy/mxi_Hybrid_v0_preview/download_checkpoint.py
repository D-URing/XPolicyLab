"""Download an evaluator-authorized artifact without embedding access credentials."""
import argparse
import hashlib
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    args = parser.parse_args()
    url = os.environ.get('MXI_ARTIFACT_URL','')
    if urllib.parse.urlsplit(url).scheme != 'https':
        raise SystemExit('Provide an authorized HTTPS URL through MXI_ARTIFACT_URL')
    if len(args.sha256)!=64 or any(c not in '0123456789abcdef' for c in args.sha256):
        raise SystemExit('A lowercase SHA-256 supplied separately is required')
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*a,**kw):
            raise ValueError('Redirect refused; obtain the final authorized artifact URL')
    digest = hashlib.sha256()
    part = args.output.with_name(args.output.name+'.partial')
    if args.output.exists() or part.exists():
        raise SystemExit('Destination exists; refusing overwrite')
    # Signed URLs are never printed. No cross-host credential forwarding.
    try:
        with urllib.request.build_opener(NoRedirect).open(url,timeout=60) as response, part.open('xb') as out:
            while chunk := response.read(4*1024*1024):
                out.write(chunk); digest.update(chunk)
        if digest.hexdigest()!=args.sha256:
            raise ValueError('Artifact checksum mismatch')
        # Hard-link commit is atomic and refuses a raced pre-existing destination.
        os.link(part,args.output)
        part.unlink()
    except Exception as exc:
        raise SystemExit(f'Download failed ({type(exc).__name__}); partial artifact retained, do not install') from None
    print('Verified authorized artifact downloaded; no automatic extraction or execution')


if __name__=='__main__':
    main()
