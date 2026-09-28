"""Same-host patch manifest and ModelPatcher adapter.

ComfyUI has no public patch-only export/reset API. Access to patches and its
version is intentionally contained here. Never accept manifests from API users:
torch.load consumes a trusted private file from our parent process, not a sandbox.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import os
import tempfile

import torch
import comfy.model_management


@dataclass(frozen=True)
class PatchManifest:
    protocol: int
    path: str
    sha256: str
    version: str
    keys: tuple[str, ...]

    def message(self):
        return {"op": "patches", **asdict(self)}

    @classmethod
    def from_message(cls, message):
        if message.get("protocol") != 1:
            raise RuntimeError("unsupported MiniMax SP patch protocol")
        return cls(1, message["path"], message["sha256"], message["version"], tuple(message["keys"]))

    def ready(self, rank):
        return {"rank": rank, "ready": True, "version": self.version,
                "sha256": self.sha256, "keys": list(self.keys)}


def validate_replies(manifest, replies, world):
    if len(replies) != world or any(reply != manifest.ready(rank) for rank, reply in enumerate(replies)):
        raise RuntimeError(f"MiniMax SP patch synchronization failed; denoise prohibited: {replies}")


def file_digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_patcher(patcher):
    if patcher.weight_wrapper_patches or patcher.hook_patches:
        raise RuntimeError("MiniMax SP dynamic weight wrappers/hooks are not synchronized")
    if patcher.get_additional_models_with_key("multigpu"):
        raise RuntimeError("Do not combine ComfyUI thread-based MultiGPU clones with MiniMax SP workers")
    for entries in patcher.patches.values():
        if any(len(entry) != 5 or entry[4] is not None for entry in entries):
            raise RuntimeError("MiniMax SP supports native static patches without custom patch functions")


def patch_version(patcher):
    validate_patcher(patcher)
    return str(patcher.patches_uuid)


def memory_budget(patcher):
    if patcher.is_dynamic():
        return 0
    return max(0, comfy.model_management.get_total_memory(patcher.load_device)
               - patcher.loaded_size() - comfy.model_management.extra_reserved_memory())


@contextmanager
def export_patches(patcher):
    version = patch_version(patcher)
    keys = tuple(sorted(patcher.patches))
    if not keys:
        yield PatchManifest(1, "", hashlib.sha256(b"").hexdigest(), version, keys)
        return
    with tempfile.TemporaryDirectory(prefix="minimax_sp_patches_") as directory:
        os.chmod(directory, 0o700)
        path = os.path.join(directory, "patches.pt")
        torch.save(patcher.patches, path)
        os.chmod(path, 0o600)
        yield PatchManifest(1, path, file_digest(path), version, keys)


class WorkerPatches:
    def __init__(self, patcher, rank):
        self.patcher = patcher
        self.rank = rank
        self.memory_required = 0
        self.version = None
        self.ready = False

    def prepare(self, memory_required, version):
        self.ready = False
        self.memory_required = memory_required
        comfy.model_management.load_models_gpu([self.patcher], memory_required=memory_required)
        self.ready = self.version == version
        return {"ready": True}

    def apply(self, manifest):
        self.ready = False
        patcher = self.patcher
        if not manifest.keys:
            patches = {}
            if self.version is None and not patcher.patches:
                patcher.patches_uuid = manifest.version
                patcher.model.current_weight_patches_uuid = manifest.version
                self.version = manifest.version
                self.ready = True
                return manifest.ready(self.rank)
        else:
            # Hash and deserialize the same open file. The parent keeps its private directory alive until ACK.
            with open(manifest.path, "rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != manifest.sha256:
                    raise RuntimeError("patch file digest mismatch")
                stream.seek(0)
                patches = torch.load(stream, map_location="cpu", weights_only=True)
        if tuple(sorted(patches)) != manifest.keys:
            raise RuntimeError("patch manifest keys mismatch")
        patcher.unpatch_model(patcher.offload_device)
        # No public reset/export/version setter exists. Preserve the tested native
        # patch representation (including offsets and strengths) without re-encoding LoRA.
        patcher.patches = patches
        patcher.patches_uuid = manifest.version
        comfy.model_management.load_models_gpu([patcher], memory_required=self.memory_required)
        if patcher.model.current_weight_patches_uuid != patcher.patches_uuid:
            raise RuntimeError("ModelPatcher did not apply the requested patch version")
        self.version = manifest.version
        self.ready = True
        return manifest.ready(self.rank)

    def require_ready(self):
        if not self.ready:
            raise RuntimeError("MiniMax SP forward before successful patch verification")
