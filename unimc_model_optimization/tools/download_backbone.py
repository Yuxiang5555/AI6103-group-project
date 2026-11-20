"""Download the recorded backbone revision and verify model/tokenizer hashes."""
import argparse
import hashlib
from pathlib import Path

REPO = 'IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English'
REVISION = 'e3aeebc7bd475d559bd0cbd8d17abf4e159d1260'
EXPECTED = {
    'pytorch_model.bin': '96b426d4f012d35bc012259bd99b2a55d635214ff3b2a1f6df42a31c83168e08',
    'config.json': '8c008d9c0b8e95f87b4bca61dbaff9993aea4a81c21d129e515bf3588d715926',
    'spiece.model': 'fefb02b667a6c5c2fe27602d28e5fb3428f66ab89c7d6f388e7c8d44a02d0336',
    'tokenizer.json': '769cfa7f8c60f978395ce84dd18106457d53d9d47d4366d5edfc490a67ef514c',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='../models/unimc-albert-235m')
    args=parser.parse_args()
    from huggingface_hub import snapshot_download
    folder=Path(args.output)
    snapshot_download(repo_id=REPO,revision=REVISION,local_dir=str(folder),allow_patterns=list(EXPECTED),max_workers=2)
    for name,expected in EXPECTED.items():
        digest=hashlib.sha256()
        with (folder/name).open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                digest.update(chunk)
        if digest.hexdigest()!=expected:
            raise ValueError(f'Backbone hash mismatch: {name}')
        print('VERIFIED',name,flush=True)
    (folder/'model_revision.txt').write_text(REVISION+'\n',encoding='utf-8')
    print('PINNED BACKBONE VERIFIED',folder.resolve(),flush=True)


if __name__=='__main__':
    main()
