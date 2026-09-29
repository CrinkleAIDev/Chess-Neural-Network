"""Build our own C++ engine and export our network to its simple binary format."""
import argparse
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / 'native' / 'scratchchess.exe'


def export(source, destination=None):
    source = Path(source)
    destination = Path(destination) if destination else source.with_suffix('.scn')
    with np.load(source, allow_pickle=False) as net:
        width = net['embedding.weight'].shape[1]
        hidden = net['fc1.weight'].shape[0]
        if not 1 <= width <= 512 or not 1 <= hidden <= 128:
            raise ValueError('Native engine supports width <=512 and hidden <=128')
        temp = destination.with_suffix('.tmp')
        with temp.open('wb') as output:
            output.write(b'SCRATCH1' + struct.pack('<III', int(net['feature_version']), width, hidden))
            for name in ('embedding.weight','bias','fc1.weight','fc1.bias','fc2.weight','fc2.bias','out.weight','out.bias'):
                output.write(net[name].astype('<f4').tobytes())
        temp.replace(destination)
    return destination


def command(model):
    model = Path(model).resolve()
    if model.suffix == '.npz':
        target = model.with_suffix('.scn')
        if not target.exists() or target.stat().st_mtime < model.stat().st_mtime:
            export(model, target)
        model = target
    if not ENGINE.exists():
        raise FileNotFoundError('Build first: python -m scratch_chess.native build')
    return [str(ENGINE), '--model', str(model)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['build','export'])
    p.add_argument('--model')
    p.add_argument('--out')
    args=p.parse_args()
    if args.action=='build':
        # --out lets a candidate build be compiled while matches still hold the main executable open.
        target=Path(args.out) if args.out else ENGINE
        subprocess.run([sys.executable,'-m','ziglang','c++','-std=c++17','-O3','-ffast-math','-march=x86_64_v3',str(ROOT/'native/engine.cpp'),'-o',str(target)],check=True)
        print(target)
    else:
        print(export(args.model,args.out))


if __name__=='__main__':
    main()
