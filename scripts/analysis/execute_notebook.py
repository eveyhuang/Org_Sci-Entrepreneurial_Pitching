#!/usr/bin/env python3
"""Execute a notebook with this Python interpreter; preserve outputs on success."""
import argparse,sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

def main():
    ap=argparse.ArgumentParser();ap.add_argument('notebook',type=Path);ap.add_argument('--timeout',type=int,default=300);args=ap.parse_args()
    path=args.notebook.resolve();root=Path(__file__).resolve().parents[2]
    nb=nbformat.read(path,as_version=4)
    km=KernelManager(kernel_name='python3');km.kernel_spec.argv[0]=sys.executable
    client=NotebookClient(nb,km=km,timeout=args.timeout,resources={'metadata':{'path':str(root)}},record_timing=True)
    try:client.execute()
    finally:
        if km.has_kernel:km.shutdown_kernel(now=True)
    nbformat.write(nb,path)
    print(f'Executed {path}: {sum(c.cell_type=="code" for c in nb.cells)} code cells, no errors')
if __name__=='__main__':main()
