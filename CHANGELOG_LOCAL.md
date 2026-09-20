# Local compatibility changelog

Original work: buqi-code/buqi-minimax-h3-multigpu, MIT, Copyright (c) 2026 buqi-code.
Upstream baseline `bce0839`. LICENSE is unchanged. Local changes do not claim
ownership of the author's Ulysses implementation.

## 2026-09-21 — review candidate

Runtime/test commit: `e5f08a124af6b6a01af4ebc8028cd209f6c32c26`.

Follow-up runtime fix: `cbb34be`, explicitly catch ComfyUI's cancellation type
(`InterruptProcessingException`, a BaseException rather than Exception) around
distributed operations. The original catch skipped group teardown on cancellation,
and the following prompt hung. Failure logs are retained; this is separate from
the successful normal generation/LoRA paths. No ComfyUI core changes were needed.

Lifecycle follow-ups: `d0533a5` / `9bfaf92` release retained control-group/store
references, unregister stale exit callbacks, clean up failed constructors, and
bind a fresh TCPStore before worker startup. Both ranks use explicit stores;
mixing an explicit store with tcp:// added a key prefix on only the latter path
in the tested PyTorch. Intermediate failed attempts remain in local results.
The store uses the [documented ephemeral-port allocation](https://docs.pytorch.org/docs/2.14/distributed.html#torch.distributed.TCPStore),
not a probe-and-release free-port race. This changes rendezvous lifecycle only,
not NCCL tensor collectives or the Ulysses algorithm.

| File | Origin | Change / reason |
| --- | --- | --- |
| minimax_sp/__init__.py | Author, modified | Preserve loader and native world_size=1; pass masks; ON_PRE_RUN sync of the actual cloned patcher; current API/signature diagnostics |
| minimax_sp/sp_forward.py | Author, modified | Current H3 PackedLayout/modality/mask/final-layer semantics; audio velocity contract; use existing all-to-all without changing collective/attention math |
| minimax_sp/sp_group.py | Author, modified | Actual root/model paths, runtime flags, complete layout transport, patch ACK gate, native numerical comparison, discard failed/dead workers; fix recursive optional VAE fallback |
| minimax_sp/sp_worker.py | Author, modified | Match native DynamicVRAM startup and model options; dispatch patch prepare/apply/ACK; transport masks |
| minimax_sp/sp_patches.py | New local | One shared patch-only IPC boundary; preserves strengths/offsets and checks hash/keys/version/rank |
| minimax_sp/sp_vae.py | Author, unchanged | Optional algorithm left intact; not part of validated baseline |
| tests/test_current_api.py | New local | Tiny-model two-rank native parity for T2V/I2V/reference/audio/masks |
| tests/test_api_compat.py, tests/test_lora_sync.py | New local | Known API incompatibility, native bypass, unsupported exchange, patch corruption/rejection and lifecycle checks |
| run_comfy.sh, tests/run_baseline.py, tests/baseline_*_gpu.py | New local | Portable environment selection, exact prompt/history/GPU telemetry capture |
| tests/benchmark.py, tests/inspect_outputs.py, tests/runtime_recovery.py | New local | Unified records, decode checks, explicitly scoped interruption/worker-failure tests |
| README.md, INSTALL.md, ARCHITECTURE.md, UPSTREAM_PLAN.md, PR drafts | Author README / new local docs | Reproducibility, attribution, review boundaries and explicit unverified scope |

The runtime has one new patch-sync module, not a new framework. Expanded compat/
adapters and the whole-file H3 hash lock remain only on the archived experiment
branch (`archive/engineering-wip-20260921`, `5728bbb`). Unknown revisions now warn
instead of being mistaken for proven incompatibility. Known incompatible API
signatures still fail. FP8 loader inputs retain upstream behavior but are not
claimed as tested on SM75.

## Validation status

Final runtime source: `9bfaf92`. Checks pass: 6 API boundary checks, 11 patch and
lifecycle checks, 5 benchmark checks, and one AST check covering six unchanged
author Ulysses/attention functions. Ten two-rank tiny-model stream comparisons
pass, maximum absolute error 7.45e-9.

Fresh candidate GPU runs at `e5f08a1` completed native single GPU, dual Base,
4-step Turbo (186.36 s) and 8-step Turbo (277.90 s) at 832×480 / 124 frames,
then LoRA clearing / I2V and repeated generation. Both LoRAs acknowledged 208
keys on both ranks; Base acknowledged zero. Timings include native verification,
so these are correctness regressions, not new performance claims.

At `9bfaf92`, controlled cancellation → actual generation and idle worker death
→ explicit error → worker recreation → actual generation passed without
restarting ComfyUI. Startup failure cleanup also has a mocked regression test.
The final source also passed 320×192 / 22-frame 4-step → 8-step → Base/I2V
regression (208 → 208 → 0 keys on both ranks); maximum first-forward video
relative error against native H3 was below 3e-7 on these final runs.
This does not prove recovery from arbitrary in-collective process death. The
test runner explicitly targets the dedicated server's worker across all its
threads and records seeds/progress to distinguish generation from cache hits.
Machine-readable records, including earlier failed attempts, are exported to
`evidence/pr_validation.json`; raw logs remain local.

Historical local compatibility results (before the candidate commit): 832×480,
124 frames, Base 20 steps, 935.28 s single / 534.30 s dual (1.75× end-to-end,
1.86× steady step); INT8 ConvRot, FP16 video VAE, 4/8-step Turbo, I2V all completed.
Those numbers must not be relabeled as newly measured performance of this commit.

Failure retained in the historical audit: legacy two-rank LoRA preparation ran
out of host RAM; native DynamicVRAM fixed the tested path. The subsequent peer
errors were effects of process death, not proof of a P2P/NCCL defect.

Local installation side effect from the earlier first ComfyUI launch: ComfyUI
itself migrated its asset database; its existing backup is retained. This PR
does not modify ComfyUI core, conda dependencies or databases.
