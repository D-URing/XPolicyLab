# mxi_Hybrid_v0_preview

**Contributor:** MXI team | **Paper:** Pending | **arXiv:** Not available | **Original code:** [MXI](https://github.com/D-URing/MXI) (restricted access)

MXI combines a Qwen3-VL planner and a conditioned video/action model. This adapter
targets the complete execution-query s4000 checkpoint on RoboDojo ARX X5 in
absolute end-effector mode. No model implementation or weights are vendored here.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

For the private S4000 handoff, prefer the complete, hash-verified delivery below.
It contains the full joint model, base assets, normalization, packaged MXI core,
runtime and pinned XPolicyLab integration; no extra model download is required.

```bash
# Run in the downloaded delivery directory, after download_hf.py verifies it.
python3 verify.py
python3 install_runtime.py
```

The runtime installs at `/opt/mxi/runtime` and requires Linux x86_64 compatible
with Ubuntu 22.04, a working NVIDIA container runtime/driver compatible with
Torch 2.11.0+cu130, and permission to create that prefix. Validation used a 96 GB
GPU; smaller devices are not certified. The simulator/environment is supplied
by the evaluator, not by this package. Follow `PRIVATE_DELIVERY.txt` for details.

The alternative wheel installation below is for an already provisioned matching
environment, not a substitute for its GPU and dependency requirements:

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

Use this adapter at upstream XPolicyLab revision
`0075cc96a1bf24c5d66a4a3a5e29eae6d739fef3` plus the MXI adapter changes. The
XPolicyLab checkout must live inside the evaluator's benchmark workspace with
its normal sibling `env_cfg/`; do not replace that with a guessed robot config.

```bash
cd XPolicyLab/policy/mxi_Hybrid_v0_preview
bash eval.sh RoboDojo stack_bowls /path/to/delivery/model arx_x5 ee 1 \
  0 1 /opt/mxi/runtime <simulator_conda_env>
```

`ckpt_name` resolves through the official checkpoint resolver and points to the
delivery's **model subdirectory**, not its top-level directory or a single action
file. Policy environment accepts a conda name, a directory containing
`bin/python`, or an executable Python path. The simulator argument remains a
conda environment name, as required by the official client scripts.
Supported: `arx_x5`, `ee`,
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

Plain and encoded-image debug checks on the recipient's environment:

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls /path/to/delivery/model arx_x5 ee 1 0 0 /opt/mxi/runtime <client_conda_env>
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls /path/to/delivery/model arx_x5 ee 1 0 0 /opt/mxi/runtime <client_conda_env>
```

## Model Assets

The complete bundle contains `runtime.json`, `assets.json`, `joint/manifest.json`,
`joint/action.pt`, `joint/planner/`, `base/`, `qwen/`, and `wan/`. Student planner,
execution queries, bridge and motion adapters must come from the same s4000.
Relative asset references, hashes and the normalization identity are verified;
loading only the action adapter is not supported.

An authorized evaluator can download a supplied bundle with:

```bash
# Preferred: private Hugging Face repository, once access is provisioned.
# Install huggingface_hub in a download-only environment, not the policy runtime.
hf auth login
python download_hf.py --repo-id <authorized-owner>/<private-model-repo> \
  --revision <exact-40-character-commit> \
  --inventory-sha256 fb23b6be2a189719b9617b693793079c54b5486aa49df0d0c42ed7fabe067cee \
  --output /path/to/new-delivery-directory

# Alternative: a separately supplied single HTTPS archive.
# Set MXI_ARTIFACT_URL privately to the issued HTTPS download URL.
python download_checkpoint.py --output mxi-s4000.bundle --sha256 <issued-sha256>
```

The HF repository and immutable revision are supplied separately; no uploaded
repository or access grant is implied by these placeholders. Authenticate with
the recipient's authorized account/token; never share an owner's write token.
A private organization repository with evaluator read access is preferred.
The HF downloader resumes only the same repo/revision/inventory identity,
downloads exact inventoried files, and verifies every SHA-256. It never executes
downloaded code. Its eight offline tests are not a live HF transfer test.
The HTTPS downloader never prints the signed URL, refuses redirects/overwrites,
and checks the complete archive before installation. Access is evaluation-only;
the public adapter's license does not grant rights to model assets.

## Configuration

Frozen profile: schema decoding; absolute EE16, xyz + qwxyz + gripper per arm;
BF16, Euler 10, sigma shift 1, native inference seed 1; prediction 32, submission
at most 24; visual history 8, stride 25, episode anchor preserved. The CLI seed
selects the environment trial; it does not silently change the native sampler.
Full task text remains the video/action branch condition. No extra task-rule
prompt injection is enabled. Robot dimensions use XPolicyLab's shared lookup.
`request_timeout_s: 300` is the client RPC response timeout, not an episode
wall-clock cutoff or a substitute for the official step budget.

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
original-versus-exported numerical regression subsequently passed on one fixed
real observation repeated twice (decisions equal, action maximum absolute error
zero); a clean-container packaged-runtime run reproduced those outputs exactly.
The model/runtime package is materialized and locally validated. Recipient
download/access, their environment acceptance, and any publication agreement
remain outstanding. No official leaderboard application has been filed.
The publicly contributed adapter follows XPolicyLab's Apache-2.0 license; core
implementation and weights are separate artifacts with their own terms.
