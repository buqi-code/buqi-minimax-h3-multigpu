# Changelog

## 0.1.0

- Add a ComfyUI-discoverable package entrypoint and Comfy Registry metadata.
- Require the MiniMax H3 API introduced at or after ComfyUI commit `8d534945`, enforce it with runtime capability checks, and test against current 0.37 development commit `7fbcfa8b` without claiming an exact release-tag range.
- Cover T2V, first/first+last-frame I2V, reference/AddGuide, masks, PDD, and static Turbo LoRA workflows.
- Retire parallel VAE handover/decode; official INT8 and FP16 VAEs use standard `VAEDecode`, while `MiniMaxH3SPVAEDecode` remains only as a deprecated compatibility alias.
- Add CPU unit/two-rank checks, workflow validation, and opt-in self-hosted GPU parity plus real loader/SPGroup lifecycle acceptance.

Local acceptance on two NVIDIA L20 GPUs against ComfyUI `7fbcfa8b`:

- Real INT8 DiT SP2 parity passed (`video rel=1.199e-6`, audio bit-exact).
- Real loader, Gloo/NCCL group creation, all-reduce/all-to-all startup probe, and clean shutdown passed.
- One-step T2V, first-frame I2V, first+last-frame I2V, AddGuide, and R2V completed through the ComfyUI server with SP2.
- Four-GPU execution and a real Turbo LoRA checkpoint were not available in this environment and remain self-hosted acceptance items.

Breaking scope: the main branch no longer supports the old ComfyUI 0.30 MiniMax API. Use an older repository release with an old ComfyUI checkout if that API is required.
