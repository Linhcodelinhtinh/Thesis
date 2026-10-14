"""Render existing audit snapshots and selected rollout frames, without simulation."""
import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'AGENTS.md').exists())
OUT = ROOT/'experiments/audits/full_clean_20261014'
rows=json.loads((OUT/'episode_audit.json').read_text())
metrics=json.loads((OUT/'audit_metrics.json').read_text())
snapdir=ROOT/'experiments/audits/smolvla_parity_20261013/snapshots'
snaps=[torch.load(snapdir/name,map_location='cpu',weights_only=False) for name in ['snap_init_state0.pt','snap_wsl_seed1042.pt']]
canvas=Image.new('RGB',(640,660),'white')
draw=ImageDraw.Draw(canvas)
image_stats={}
for row,(name,snap) in enumerate(zip(['Windows audit snapshot','WSL audit snapshot'],snaps)):
    for col,cam in enumerate(['agentview_image','robot0_eye_in_hand_image']):
        img=snap[cam]
        x,y=col*320,row*330
        canvas.paste(Image.fromarray(np.asarray(img,dtype=np.uint8)).resize((300,300)),(x+10,y+28))
        draw.text((x+10,y+2),f'{name} / {cam}',fill='black')
        image_stats[f'{name}/{cam}']=dict(shape=list(img.shape),mean_rgb=img.mean(axis=(0,1)).tolist())
canvas.save(OUT/'prior_snapshot_comparison.png')
(OUT/'snapshot_image_stats.json').write_text(json.dumps(image_stats,indent=2))

selected=[]
for task,success in [('libero_goal_task8',True),('libero_object_task9',True),('libero_10_task2',True),('libero_object_task0',False),('libero_spatial_task0',False),('libero_10_task4',False)]:
    candidates=[r for r in rows if r['id'].split('\\')[0]==task and r['success']==success and r['video_exists']]
    r=candidates[0]
    selected.append(r['id'])
    path=ROOT/'experiments/results/full-clean'/r['id']/'video.mp4'
    cap=cv2.VideoCapture(str(path))
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    points=sorted(set([0,min(50,n-1),min(100,n-1),min(250,n-1),n-1]))
    canvas=Image.new('RGB',(256*len(points),310),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((5,3),r['id'].replace('\\',' / ')+f" | success={success} | {r['steps']} actions",fill='black')
    for col,frame_no in enumerate(points):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no)
        ok,im=cap.read()
        if not ok: raise RuntimeError(f'Cannot read {path} frame {frame_no}')
        canvas.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((256,256)),(col*256,50))
        draw.text((col*256+5,30),f'Video frame {frame_no}',fill='black')
    canvas.save(OUT/(r['id'].replace('\\','_')+'_frames.png'))
    cap.release()
(OUT/'selected_videos.json').write_text(json.dumps(selected,indent=2))

fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1,1.4]})
suites=['libero_spatial','libero_object','libero_goal','libero_10']
vals=[metrics['suites'][s]['successes']/50*100 for s in suites]
axes[0].bar(range(4),vals,color=['#467b9b','#b35a42','#4c8974','#aa6645'])
axes[0].set_xticks(range(4),['Spatial','Object','Goal','Long'])
axes[0].set_ylim(0,100)
axes[0].set_ylabel('Success rate (%)')
axes[0].set_title('200 episodes / 40 tasks / states 0-4')
for i,s in enumerate(suites):
    lo,hi=np.array(metrics['suites'][s]['ci95'])*100
    axes[0].errorbar(i,vals[i],yerr=[[vals[i]-lo],[hi-vals[i]]],fmt='none',color='black',capsize=4)
    axes[0].text(i,vals[i]+4,f'{int(vals[i])}%',ha='center')
mat=np.array([[sum(r['success'] for r in rows if r['suite']==s and r['task_id']==t) for t in range(10)] for s in suites])
axes[1].pcolormesh(np.arange(11)-.5,np.arange(5)-.5,mat,vmin=0,vmax=5,cmap='Blues',shading='flat')
axes[1].invert_yaxis()
axes[1].set_xticks(range(10))
axes[1].set_yticks(range(4),['Spatial','Object','Goal','Long'])
axes[1].set_xlabel('Task ID')
axes[1].set_title('Successful episodes per task (out of 5)')
for i in range(4):
    for j in range(10):
        axes[1].text(j,i,str(mat[i,j]),ha='center',va='center',color='white' if mat[i,j]>=3 else 'black')
fig.suptitle('full-clean: 46% overall | LIBERO-derived / non-comparable to official paper')
fig.text(.5,.015,'Error bars: descriptive episode-level Wilson 95% CI; task/state clustering is not modeled.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.035,1,.94))
fig.savefig(OUT/'success_overview.png',dpi=180)
plt.close(fig)
print(json.dumps(dict(selected_videos=selected,image_stats=image_stats),indent=2))
