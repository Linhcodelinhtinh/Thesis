#!/usr/bin/env python
"""Real T1/T3 demo replay against locked upstream evaluator; no policy/fixture score."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import h5py
import numpy as np
from src.simulator.libero_mem_env import LiberoMemEnv
from src.evaluation.libero_mem_protocol import DEFAULT_PROTOCOL, source_digest
from src.utils.file_hash import sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-lock',required=True)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/libero_mem')
    p.add_argument('--tasks',nargs='+',default=['T1','T3'],choices=['T1','T3'])
    p.add_argument('--demo-ids',nargs='+',type=int,default=[1])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a fresh conformance output')
    a.output.mkdir(parents=True)
    report=dict(classification='REAL_DEMONSTRATION_REPLAY_NOT_POLICY_SCORE',status='running',
                protocol_sha256=source_digest(DEFAULT_PROTOCOL),split_file_sha256=sha256_file(a.split_lock),
                runtime={k:importlib.metadata.version(k) for k in ['robosuite','numpy','mujoco','bddl','robomimic']},episodes=[])
    for task in a.tasks:
        env=LiberoMemEnv(task,split_lock=a.split_lock,split='train',purpose='demo_replay')
        try:
            for demo_id in a.demo_ids:
                key='demo_'+str(demo_id)
                index=next(i for i,e in enumerate(env.episodes) if e['demo']==key)
                first=env.reset(index)
                env.upstream._overshot=True  # reset isolation probe only, before replay begins
                obs=env.reset(index)
                reset_error=max(float(np.max(np.abs(first[k].astype(float)-obs[k].astype(float)))) for k in first)
                if reset_error>1e-6 or env.upstream._overshot or env.upstream._satisfied_subgoals:
                    raise AssertionError('Repeated exact-state reset is not isolated/reproducible')
                original=env.upstream._check_success
                calls=[]
                def counted(*args,**kwargs):
                    calls.append(bool(kwargs.get('inc',args[0] if args else False)))
                    return original(*args,**kwargs)
                env.upstream._check_success=counted
                task_path=a.data_dir/(env.task_name+'_demo.hdf5')
                pose_errors=[];quat_errors=[];image_errors=[];timestamps=[]
                started=time.perf_counter()
                try:
                    with h5py.File(task_path,'r') as f, (a.output/(task+'_'+str(demo_id)+'_trace.jsonl')).open('w') as trace:
                        g=f['data'][key];actions=g['actions'][:];robot=g['robot_states'][:]
                        for t in range(len(actions)):
                            pose_errors.append(float(np.linalg.norm(obs['robot0_eef_pos']-robot[t,2:5])))
                            q=obs['robot0_eef_quat'];qt=robot[t,5:9]
                            quat_errors.append(float(min(np.linalg.norm(q-qt),np.linalg.norm(q+qt))))
                            if t%20==0:
                                actual=obs['agentview_image'][::-1].astype(float)
                                image_errors.append(float(np.mean(np.abs(actual-g['obs/agentview_rgb'][t]))))
                            info=dict(env.last_info);info['eval_result']=asdict(info['eval_result'])
                            trace.write(json.dumps(dict(frame=t,info=info,eef_error=pose_errors[-1]))+'\n')
                            if t%50==0:print(task,key,t,'count',len(env.upstream._satisfied_subgoals),'eef_error',pose_errors[-1],flush=True)
                            if env._is_done:break
                            # The final target is not executed: no corresponding source next observation.
                            if t+1<len(actions):obs,_,_,_=env.step(actions[t])
                        count=len(env.upstream._satisfied_subgoals)
                        result=dict(task=task,demo=key,frames_observed=len(pose_errors),
                            source_frames=len(actions),completed=count,target=env.target_cycles,
                            final_predicate=env.last_info['final_predicate'],overshot=env.upstream._overshot,
                            reset_max_error=reset_error,inc_true_calls=sum(calls),inc_false_calls=len(calls)-sum(calls),
                            eef_error_median_m=float(np.median(pose_errors)),eef_error_max_m=max(pose_errors),
                            quaternion_error_max=max(quat_errors),rgb_mae_sampled=float(np.mean(image_errors)),
                            elapsed_seconds=time.perf_counter()-started)
                        if sum(calls)!=len(pose_errors)-1:raise AssertionError('Incorrect explicit increment cadence')
                        result['sequence_replay_passed']=count==env.target_cycles and not env.upstream._overshot and bool(env.last_info['final_predicate'])
                        report['episodes'].append(result)
                        print(json.dumps(result),flush=True)
                finally:env.upstream._check_success=original
                (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        finally:env.close()
    report['status']='passed' if all(e['sequence_replay_passed'] for e in report['episodes']) else 'replay_mismatch'
    report['scope']='This verifies selected demonstration replay and reset/call schedule, not a policy benchmark or author-runtime equivalence.'
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'])
    if report['status']!='passed':raise SystemExit(2)


if __name__=='__main__':main()
