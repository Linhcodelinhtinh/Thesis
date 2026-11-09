# LIBERO-Mem author-source runtime track

This track is isolated from `envs/libero_reference.yml` and the existing T1/T3
benchmark protocol. It uses the pinned LIBERO-Mem source revision
`0eee0defa4024e51511b6e15a79f30e9620209de`, including the author's vendored
robosuite 1.4.1 and robomimic 0.3.0 forks.

## Runtime provenance

The author's README instructs users to install `scripts/requirements.txt` before
collecting demonstrations. That requirements file pins MuJoCo 3.2.2, NumPy
1.23.5, BDDL 3.5.0, and other simulator-side packages copied into
`envs/libero_mem_author_candidate.txt`. This is the simulator-relevant subset,
not the full policy/training dependency set. The candidate environment uses Python
3.10.22 because the public setup does not specify a Python version and its
requirements need Python 3.9 or newer. The setup script writes the installed
versions and source paths to
`experiments/runtime/libero_mem_author_candidate/runtime-manifest.json` and
`pip-freeze.txt`.

The published HDF5 demonstration inspected for this replay does not contain a
MuJoCo version/build field. The pinned repository also has no Dockerfile,
Conda environment, or lockfile that records the actual data-generation host.
Therefore the author's requirements file is the available source-level runtime
pin, but it cannot prove which environment produced this particular released
HDF5 file.

The source's `experiments/robot/libero/libero_utils.py` currently has an active
`1/0` in `get_libero_env()`, so that helper cannot be called unchanged. The
diagnostic constructs the same task class and uses Panda, OSC_POSE, the default
gripper, and 20 Hz, without modifying vendored source. The image window is
disabled for headless WSL; offscreen camera rendering remains enabled.

## Install and replay

From WSL at the project root:

```bash
bash scripts/install_libero_mem_author_candidate.sh
MUJOCO_GL=osmesa .venv_libero_mem_author_candidate/bin/python scripts/diagnose_libero_mem_author_bottle.py --demo demo_33
```

The diagnostic restores the metadata-backed T1 initial-state vector and checks
exact state restoration. It then records the bottle pose and camera frame
before and after each of 20 no-op actions. This mirrors the warm-up in the author's
`mem_3_refine_demonstration.py`, which advances the environment by 20 dummy
actions after restoring each source episode's initial state and before
recording the refined trajectory. For frame comparison, the diagnostic applies
the author's `get_visual()` transform (horizontal flip followed by a 180-degree
rotation, equivalent to a vertical flip). Images and the pose trace are written
under `scratch/libero_mem_author_candidate/bottle_warmup/`.

## Comparison history

An earlier comparison used MuJoCo 3.2.3, which was mistakenly treated as a
project comparison pin before the author's script requirements were checked.
In that run, the bottle started upright at z=1.080 m and lay sideways after 20
no-op actions at z about 0.9534 m, while the first stored refined-demo frame
showed it upright. The author-vendored robosuite 1.4.1 and the earlier 1.4.0
run produced matching bottle trajectories under MuJoCo 3.2.3 (maximum position
difference 8.9e-16 m; quaternion-component difference 8.5e-15). This rules out
the robosuite fork change alone as the explanation, but that run is superseded
for provenance purposes by the MuJoCo 3.2.2 replay.

### Author-pinned dependency replay (2026-11-09)

- Runtime: Python 3.10.22 (project-selected; author did not pin Python),
  author-vendored robosuite 1.4.1, MuJoCo 3.2.2, BDDL 3.5.0, NumPy 1.23.5,
  plus the simulator/render pins listed in the candidate requirements file.
- Demo: T1 `demo_33`; initial-state SHA-256
  `b8a0272f0b2aa2472e5d549823e2b3168ccd19954c3f91fc4e72503de365e499`; state
  restore max error `0.0`.
- The bottle starts upright (tilt 0 degrees, z=1.080 m), tilts past 90 degrees
  by warm-up step 13, and settles at about 107.26 degrees and z=0.953400 m by
  step 20. The released refined HDF5 frame 0 shows the bottle upright.
- After applying the author's image transform, replay/frame mean absolute RGB
  error is 6.37/255. The scene and camera align closely, but the bottle pose
  does not; the remaining pixel difference is visibly concentrated around the
  bottle and rendering details. The released file stores no physical object
  pose, so this pose comparison is visual rather than a numeric HDF5 pose
  comparison.
- Repeating after pinning the remaining renderer/science packages produced
  the same trajectory to floating-point noise. The environment is therefore
  reproducible on this machine, but it does not reproduce the bottle state in
  this released demo.
- Repeating with the author's `has_renderer=True` setting under WSLg/GLFW also
  produced the same bottle pose and frame comparison. The headless/on-screen
  renderer choice does not explain this mismatch.

Do not call the dynamics equivalent. The source pin identifies MuJoCo 3.2.2 as
the intended script dependency, but the released artifact does not identify its
actual binary build or Python environment, and the exact-source replay still
does not match. The author's own README notes that replay can be nondeterministic
for a few episodes. Preserve this mismatch and obtain the original data-run
environment or a per-run environment manifest before attributing it to a
specific simulator difference. Do not change bottle geometry, contacts, state
vectors, or warm-up actions to force a match.

LIBERO-Mem/custom results from this track must be reported separately from
official LIBERO results. The official LIBERO environment and scores remain
governed by `AGENTS.md`.
