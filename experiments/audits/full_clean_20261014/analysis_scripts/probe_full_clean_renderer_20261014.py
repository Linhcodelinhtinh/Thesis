"""Diagnostic render of an official state; no policy, no benchmark score, no env edits."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from src.simulator.libero_env import LiberoEnv

ROOT=next(p for p in Path(__file__).resolve().parents if (p / 'AGENTS.md').exists())
OUT=ROOT/'experiments/audits/full_clean_20261014/render_probe'
OUT.mkdir(parents=True,exist_ok=True)
env=LiberoEnv(benchmark_name='libero_object',task_id=0,mode='LIBERO-DERIVED',camera_height=256,camera_width=256,num_steps_wait=10)
obs=env.reset(initial_state_id=0)
sim=env.sim
ctx=sim._render_context_offscreen
data=dict(purpose='NON-COMPARABLE renderer diagnostic, zero policy rollouts',
    packages={x:importlib.metadata.version(x) for x in ['numpy','mujoco','robosuite','lerobot','bddl']},
    env={x:os.environ.get(x) for x in ['MUJOCO_GL','GALLIUM_DRIVER','MESA_D3D12_DEFAULT_ADAPTER_NAME']},
    collision_setting=env._env.env.render_collision_mesh,visual_setting=env._env.env.render_visual_mesh,
    original_geomgroup=np.array(ctx.vopt.geomgroup).tolist(),
    collision_colors={'robot':[0,.5,0,1],'gripper':[0,0,.5,1]})
try:
    from OpenGL import GL
    data['gl']={key:GL.glGetString(getattr(GL,key)).decode() for key in ['GL_VENDOR','GL_RENDERER','GL_VERSION']}
except Exception as exc:
    data['gl_error']=repr(exc)
for cam in ['agentview','robot0_eye_in_hand']:
    Image.fromarray(obs[cam+'_image']).save(OUT/(cam+'_original.png'))
before=np.array(sim.data.qpos).copy()
ctx.vopt.geomgroup[0]=0
ctx.vopt.geomgroup[1]=1
data['after_index_assignment']=np.array(ctx.vopt.geomgroup).tolist()
for cam in ['agentview','robot0_eye_in_hand']:
    im=sim.render(camera_name=cam,height=256,width=256)
    Image.fromarray(im).save(OUT/(cam+'_index_assignment.png'))
ctx.vopt.geomgroup=np.array([0,1,1,0,0,0],dtype=np.uint8)
data['after_full_assignment']=np.array(ctx.vopt.geomgroup).tolist()
for cam in ['agentview','robot0_eye_in_hand']:
    im=sim.render(camera_name=cam,height=256,width=256)
    Image.fromarray(im).save(OUT/(cam+'_full_assignment.png'))
data['qpos_unchanged']=bool(np.array_equal(before,sim.data.qpos))
data['files']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
    Path(__file__),ROOT/'src/simulator/libero_env.py',Path(__import__('robosuite').__file__).parent/'utils/binding_utils.py']}
(OUT/'probe.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data,indent=2),flush=True)
env.close()
