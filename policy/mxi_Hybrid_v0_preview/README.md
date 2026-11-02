# mxi_Hybrid_v0_preview

**Contributor:** MXI team | **Paper:** Pending | **arXiv:** Not available | **Original code:** [MXI](https://github.com/D-URing/MXI) (restricted access)

MXI combines a Qwen3-VL planner and a conditioned video/action model. This adapter
targets the complete execution-query s4000 checkpoint on RoboDojo ARX X5 in
absolute end-effector mode. No model implementation or weights are vendored here.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use the evaluator-provided, versioned inference environment and MXI wheel. Its
checksum is delivered separately through the authorized evaluation channel.
No public artifact URL or model redistribution permission is implied.

```bash
cd XPolicyLab/policy/mxi_Hybrid_v0_preview
export MXI_WHEEL=/path/to/authorized/mxi_hybrid.whl
export MXI_WHEEL_SHA256=<independently-provided-sha256>
bash install.sh
```

`install.sh` verifies the wheel before installation and does not upgrade shared
GPU dependencies. The exact environment recipe and real-weight validation report
must accompany the final submission; the current adapter remains pre-submission.

## Data Processing

Evaluation-only: no training-data conversion is supported here. The deployment
accepts official RGB observations, raw EE poses/gripper states, and the original
task instruction. Training uses an upstream LeRobot export plus causal planning
records; these are not interchangeable with an unmodified generic converter.
`process_data.sh` is intentionally absent, not a fake successful conversion stub.

## Training

Evaluation-only: this entry does not start training. `train.sh` is absent.
Maintainer agreement for the evaluation-only exception and a training-code
release timeline are still required before the official application. No release
date or approval has been assumed on the authors' behalf.

## Evaluation

```bash
bash eval.sh RoboDojo stack_bowls /path/to/authorized/mxi-s4000 arx_x5 ee 1 \
  0 1 <policy_conda_env> <simulator_conda_env>
```

`ckpt_name` resolves through the official checkpoint resolver and points to the
deployment bundle, not a single action file. Supported: `arx_x5`, `ee`,
`eval_batch=false`. Each server runs one episode at a time; scale with independent
servers, not interleaved sessions in one model. `MXI_OUTPUT_ROOT` sets the policy
decision-log directory (default `./mxi-results`). Official scores/videos remain
in the environment client's result directory.

The custom `deploy.py` is required: it sends every intermediate observation,
requests a chunk with an execution receipt, and acknowledges the actual number
of executed actions, including terminal truncation and interrupted execution.
It reads the official environment's `step_lim`; no example wall-clock budget is
substituted. KEEP regenerates actions. No teacher/script/zero-action fallback is
implemented. The official debug environment's own 20-step budget is used only
when `EVAL_ENV_TYPE=debug`; it is not a simulator result.

The client forwards only the three RGB streams, raw robot state, original task
instruction and transport identifiers. Depth, object poses, simulator annotations
and task metrics are not supplied to the policy.

Plain and encoded-image debug checks must both be run before submission:

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls /path/to/authorized/mxi-s4000 arx_x5 ee 1 0 0 <policy_conda_env> <client_conda_env>
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls /path/to/authorized/mxi-s4000 arx_x5 ee 1 0 0 <policy_conda_env> <client_conda_env>
```

## Model Assets

The complete bundle contains `runtime.json`, `assets.json`, `joint/manifest.json`,
`joint/action.pt`, `joint/planner/`, `base/`, `qwen/`, and `wan/`. Student planner,
execution queries, bridge and motion adapters must come from the same s4000.
Relative asset references, hashes and the normalization identity are verified;
loading only the action adapter is not supported.

An authorized evaluator can download a supplied bundle with:

```bash
# Set MXI_ARTIFACT_URL privately to the issued HTTPS download URL.
python download_checkpoint.py --output mxi-s4000.bundle --sha256 <issued-sha256>
```

The downloader never prints the signed URL, refuses redirects/overwrites, and
checks the complete download before installation. Access is evaluation-only;
the public adapter's license does not grant rights to model assets.

## Configuration

Frozen profile: schema decoding; absolute EE16, xyz + qwxyz + gripper per arm;
BF16, Euler 10, sigma shift 1, native inference seed 1; prediction 32, submission
at most 24; visual history 8, stride 25, episode anchor preserved. The CLI seed
selects the environment trial; it does not silently change the native sampler.
Full task text remains the video/action branch condition. No extra task-rule
prompt injection is enabled. Robot dimensions use XPolicyLab's shared lookup.

## Notes

Status: pre-submission, not a certified closed-loop deployment.
On 2026-11-02 the complete s4000 model loaded and passed two official-WebSocket
debug episodes (20 executed actions each), using plain and encoded RGB inputs.
These used real model weights, not a policy test double, and are protocol checks,
not task scores. Two infrastructure failures were retained: renderer/GPU
initialization and a simulator CLI abbreviation conflict, both before any robot
action. After correcting GPU selection, writable Kit cache paths and argument
parsing in a separate simulator snapshot, one bounded `stack_blocks` rollout
(seed 0, layout 0) completed successfully: official Q=1.0, 250 executed actions
against a 550-step budget, 11 decision/receipt pairs, and three unedited official
videos of 251 frames each. The terminal receipt correctly acknowledged only 10
actions from the final 24-action chunk. There were no policy/protocol errors.

This is a one-instance engineering acceptance, not a leaderboard score or proof
of broad task performance. It used the pinned official server and client loop
through a private reproducible acceptance runner; the generic shell entrypoint
has not separately been certified in a clean evaluator environment. Fixed-input
original-versus-exported numerical regression, portable artifact delivery and
maintainer approval remain outstanding. No official application has been filed.
The publicly contributed adapter follows XPolicyLab's Apache-2.0 license; core
implementation and weights are separate artifacts with their own terms.
