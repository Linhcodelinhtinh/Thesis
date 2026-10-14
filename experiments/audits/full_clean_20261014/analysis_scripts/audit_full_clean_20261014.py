"""Read-only forensic analysis of full-clean; writes derived audit artifacts only."""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'AGENTS.md').exists())
BASE = ROOT / 'experiments/results/full-clean'
OUT = ROOT / 'experiments/audits/full_clean_20261014'
OUT.mkdir(parents=True, exist_ok=True)

def counts(items):
    return dict(collections.Counter(str(x) for x in items))

def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    c = (p+z*z/(2*n))/(1+z*z/n)
    r = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(c-r), float(c+r)]

rows = []
all_inf, all_sim, all_actions = [], [], []
hashes = {}
issues = []
for ep in sorted(BASE.glob('*/init_*/episode.json')):
    raw = ep.read_bytes()
    hashes[str(ep.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    e = json.loads(raw)
    p = e['provenance']
    d = e['diagnostics']
    logs = [json.loads(x) for x in (ep.parent/'model_output.jsonl').read_text().splitlines()]
    a = np.array([x['action'] for x in logs])
    traj = np.load(ep.parent/'trajectory.npz', allow_pickle=False)
    if len(logs) != e['num_steps'] or not np.allclose(a, traj['actions']):
        issues.append(str(ep))
    success_flags = [x['success'] for x in logs]
    if any(success_flags) != e['success']:
        issues.append('success mismatch '+str(ep))
    diags = [x['contact_diagnostics'] for x in logs]
    inf = e['inference_latencies_ms']
    sim = e['simulation_latencies_ms']
    all_inf.extend(inf)
    all_sim.extend(sim)
    all_actions.extend(a)
    suite, tid = ep.parent.parent.name.rsplit('_task', 1)
    evidence = d['evidence']
    milestones = d['subtask_milestones']
    missing_obj = sum(x['object_pos'] is None for x in diags)
    boundary, within = [], []
    for j in range(1,len(a)):
        (boundary if logs[j]['is_replanned'] else within).append(float(np.linalg.norm(a[j,:6]-a[j-1,:6])))
    r = dict(id=str(ep.parent.relative_to(BASE)), suite=suite, task_id=int(tid), task_name=e['task_name'],
        init=e['initial_state_id'], seed=e['seed'], success=e['success'], steps=e['num_steps'],
        phase=d['failure_phase'], termination=d['termination_reason'], attribution=d['primary_failure_code'],
        reached=evidence['ever_reached'], grasped=evidence['ever_grasped'], lifted=evidence['ever_lifted'],
        transported=evidence['ever_transported'], placed=evidence['ever_placed'],
        missing_object_steps=missing_obj, missing_goal_steps=sum(x['goal_pos'] is None for x in diags),
        first_object=diags[0]['object_pos'], first_goal=diags[0]['goal_pos'],
        milestone_count=len(milestones), achieved_count=sum(m['achieved'] for m in milestones),
        milestones=milestones, success_first_step=next((x['step'] for x in logs if x['success']), None),
        calls=len(inf), infer_total_ms=sum(inf), sim_total_ms=sum(sim),
        first_infer_ms=inf[0], infer_mean_ms=np.mean(inf),
        video_claimed=bool(e.get('video_path')), video_exists=(ep.parent/'video.mp4').exists(),
        raw_equal_action=bool(np.allclose(a,np.array([x['unnormalized_action'] for x in logs]))),
        nonfinite_actions=int((~np.isfinite(a)).sum()),
        arm_outside_steps=int((np.abs(a[:,:6])>1).any(axis=1).sum()),
        gripper_outside_steps=int((np.abs(a[:,6])>1).sum()),
        boundary_delta_mean=float(np.mean(boundary)) if boundary else None,
        within_delta_mean=float(np.mean(within)) if within else None,
        drift_mm=e['settling_snapshot']['eef_drift_mm'], provenance=p)
    rows.append(r)

summary = dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
    task_count=len(set((r['suite'],r['task_id']) for r in rows)),
    episode_wilson_95=wilson(sum(r['success'] for r in rows),len(rows)), consistency_issues=issues)
summary['suites']={}
for suite in sorted(set(r['suite'] for r in rows)):
    rr=[r for r in rows if r['suite']==suite]
    limit = {'libero_spatial':280,'libero_object':280,'libero_goal':300,'libero_10':520}[suite]
    summary['suites'][suite]=dict(episodes=len(rr), successes=sum(r['success'] for r in rr),
        ci95=wilson(sum(r['success'] for r in rr),len(rr)),
        phases=counts(r['phase'] for r in rr if not r['success']),
        reached=sum(r['reached'] for r in rr),grasped=sum(r['grasped'] for r in rr),
        lifted=sum(r['lifted'] for r in rr),placed=sum(r['placed'] for r in rr),
        missing_object_episodes=sum(r['missing_object_steps']>0 for r in rr),
        mean_sim_ms=sum(r['sim_total_ms'] for r in rr)/sum(r['steps'] for r in rr),
        mean_infer_ms=sum(r['infer_total_ms'] for r in rr)/sum(r['calls'] for r in rr),
        mean_success_steps=float(np.mean([r['steps'] for r in rr if r['success']])),
        posthoc_limit=limit,successes_by_limit=sum(r['success'] and r['steps']<=limit for r in rr))
summary['termination']=counts(r['termination'] for r in rows)
summary['phases']=counts(r['phase'] for r in rows if not r['success'])
summary['attribution']=counts(r['attribution'] for r in rows if not r['success'])
summary['success_by_init']={str(i):sum(r['success'] for r in rows if r['init']==i) for i in range(5)}
summary['protocol_fields']={k:counts(r['provenance'].get(k) for r in rows) for k in [
    'git_commit','git_clean_working_tree','git_diff_sha256','baseline_lock','model_config_sha256',
    'python_version','robosuite_version','bddl_version','execution_horizon_s','policy_chunk_size',
    'num_steps_wait','base_seed','camera_resolution','memory_condition']}
summary['missing_object_episodes']=[r['id'] for r in rows if r['missing_object_steps']]
summary['all_milestones_but_failure']=[r['id'] for r in rows if r['achieved_count']==r['milestone_count'] and not r['success']]
summary['milestones_at_step_zero']=[dict(id=r['id'],predicates=[m['predicate_key'] for m in r['milestones'] if m['first_achieved_step']==0]) for r in rows if any(m['first_achieved_step']==0 for m in r['milestones'])]
summary['video_claimed']=sum(r['video_claimed'] for r in rows)
summary['video_exists']=sum(r['video_exists'] for r in rows)
summary['timing']=dict(calls=len(all_inf),steps=len(all_sim),
    mean_infer_ms=float(np.mean(all_inf)),p95_infer_ms=float(np.percentile(all_inf,95)),
    mean_sim_ms=float(np.mean(all_sim)),infer_hours=sum(all_inf)/3600000,sim_hours=sum(all_sim)/3600000,
    episode_mean_of_call_means=float(np.mean([r['infer_mean_ms'] for r in rows])),
    first_infer_p95=float(np.percentile([r['first_infer_ms'] for r in rows],95)))
aa=np.array(all_actions)
summary['actions']=dict(min=aa.min(axis=0).tolist(),max=aa.max(axis=0).tolist(),
    outside_per_dim=(np.abs(aa)>1).sum(axis=0).tolist(),nonfinite=int((~np.isfinite(aa)).sum()),
    passthrough_episodes=sum(r['raw_equal_action'] for r in rows),
    near_zero_arm_steps=int((np.linalg.norm(aa[:,:6],axis=1)<.01).sum()))
summary['all_steps_failed']=sum(r['steps'] for r in rows if not r['success'])
summary['late_successes']=[dict(id=r['id'],steps=r['steps']) for r in rows if r['success'] and r['steps']>{'libero_spatial':280,'libero_object':280,'libero_goal':300,'libero_10':520}[r['suite']]]
summary['tasks_zero_success']=[f'{suite}_task{tid}' for suite,tid in sorted(set((r['suite'],r['task_id']) for r in rows)) if not any(r['success'] for r in rows if (r['suite'],r['task_id'])==(suite,tid))]
(OUT/'audit_metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(OUT/'episode_audit.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
(OUT/'input_episode_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
