"""Small binding diagnostic; no simulator, policy, benchmark resource or environment edits."""
import ctypes
import importlib.metadata
import json
import sys
from pathlib import Path
import mujoco
import numpy as np

opt=mujoco.MjvOption()
def snapshot():
    a=opt.geomgroup
    return dict(view=a.tolist(),shape=list(a.shape),strides=list(a.strides),dtype=str(a.dtype),
        raw_six_bytes=list(ctypes.string_at(a.ctypes.data,6)))
result=dict(mujoco=mujoco.__version__,numpy=np.__version__,python=sys.version,executable=sys.executable,before=snapshot())
opt.geomgroup[0]=0
result['after_geomgroup_0_equals_0']=snapshot()
opt.geomgroup[1]=1
result['after_geomgroup_1_equals_1']=snapshot()
result['expected_visual_only_prefix']=[0,1]
result['index_assignments_match_expected']=result['after_geomgroup_1_equals_1']['raw_six_bytes'][:2]==[0,1]
out=next(p for p in Path(__file__).resolve().parents if (p / 'AGENTS.md').exists())/'experiments/audits/full_clean_20261014'/('mujoco_array_probe_'+sys.argv[1]+'.json' if len(sys.argv)>1 else 'mujoco_array_probe.json')
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
