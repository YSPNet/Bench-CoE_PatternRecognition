"""Install only in prediction subprocesses, before loading policy artifacts."""
import os
import sys
from pathlib import Path
from .runtime import ROOT, OUT


def install():
    denied_prefixes=[ROOT/"ORACLE", ROOT/"BENCHMARKS", OUT/"evaluation", OUT/"test"]
    denied_files={str((ROOT/"data/matrices/Y.npz").resolve()),str((ROOT/"data/matrices/C.npy").resolve())}
    attempts=[]
    def hook(event,args):
        if event != "open" or not args or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path=Path(os.fsdecode(args[0])).resolve()
        # Filename-level denial supplements immutable allowlisted manifests.
        forbidden=(str(path) in denied_files or path.name=="labels.npz" or
                   any(path.is_relative_to(p) for p in denied_prefixes))
        if forbidden:
            attempts.append(str(path))
            raise PermissionError("prediction process cannot open outcome stores: "+str(path))
    sys.addaudithook(hook)
    return attempts
