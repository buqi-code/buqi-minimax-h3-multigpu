"""Protocol tests without loading H3 weights. Real Turbo runs are separate evidence."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bootstrap import setup
setup()
import torch
from minimax_sp import sp_patches as sync
from minimax_sp.sp_group import SPGroup
from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import ModelPatcher


def fake_patcher():
    return SimpleNamespace(patches={"weight": [(0.7, ("diff", (torch.ones(2),)), 1.0, None, None)]},
                           patches_uuid="four", weight_wrapper_patches={}, hook_patches={},
                           get_additional_models_with_key=lambda key: [], offload_device="cpu",
                           model=SimpleNamespace(current_weight_patches_uuid=None), unpatch_model=Mock())


class PatchSyncTests(unittest.TestCase):
    def test_switch_and_clear(self):
        parent, child = fake_patcher(), fake_patcher()
        worker = sync.WorkerPatches(child, 1)

        def load(models, **kwargs):
            for p in models:
                p.model.current_weight_patches_uuid = p.patches_uuid

        with patch.object(sync.comfy.model_management, "load_models_gpu", side_effect=load):
            for version, keys in (("four", 1), ("eight", 1), ("base", 0)):
                parent.patches_uuid = version
                if not keys:
                    parent.patches = {}
                worker.prepare(0, version)
                self.assertFalse(worker.ready)
                with sync.export_patches(parent) as manifest:
                    path = Path(manifest.path)
                    reply = worker.apply(sync.PatchManifest.from_message(manifest.message()))
                    sync.validate_replies(manifest, [manifest.ready(0), reply], 2)
                    self.assertEqual(len(child.patches), keys)
                    if keys:
                        self.assertEqual(child.patches["weight"][0][0], 0.7)
                    worker.require_ready()
                self.assertFalse(path.exists())

    def test_bad_hash_and_keys_cannot_load(self):
        worker = sync.WorkerPatches(fake_patcher(), 1)
        with sync.export_patches(fake_patcher()) as manifest:
            for bad in (replace(manifest, sha256="bad"), replace(manifest, keys=("wrong",))):
                with patch.object(sync.comfy.model_management, "load_models_gpu") as load:
                    with self.assertRaisesRegex(RuntimeError, "mismatch"):
                        worker.apply(bad)
                    load.assert_not_called()
                    with self.assertRaisesRegex(RuntimeError, "before successful"):
                        worker.require_ready()

    def test_rank_and_version_mismatch(self):
        with sync.export_patches(fake_patcher()) as manifest:
            for replies in ([manifest.ready(0)], [manifest.ready(0), manifest.ready(0)],
                            [manifest.ready(0), {**manifest.ready(1), "version": "wrong"}]):
                with self.assertRaisesRegex(RuntimeError, "denoise prohibited"):
                    sync.validate_replies(manifest, replies, 2)

    def test_load_failure_not_ready(self):
        worker = sync.WorkerPatches(fake_patcher(), 1)
        with sync.export_patches(fake_patcher()) as manifest:
            with patch.object(sync.comfy.model_management, "load_models_gpu", side_effect=RuntimeError("load failure")):
                with self.assertRaisesRegex(RuntimeError, "load failure"):
                    worker.apply(manifest)
        self.assertFalse(worker.ready)

    def test_unsupported_hooks_rejected(self):
        parent = fake_patcher()
        parent.hook_patches = {"dynamic": True}
        with self.assertRaisesRegex(RuntimeError, "hooks"):
            sync.patch_version(parent)

    def test_actual_modelpatcher_clone_keeps_callback(self):
        base = torch.nn.Module()
        base.diffusion_model = torch.nn.Linear(2, 2)
        base.diffusion_model._forward = base.diffusion_model.forward
        patcher = ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))
        group = Mock()
        bound = patcher.clone()
        bound.add_callback_with_key(CallbacksMP.ON_PRE_RUN, "minimax_sp_lora", group.sync_patches)
        clone = bound.clone()
        clone.add_patches({"diffusion_model.weight": ("diff", (torch.ones(2, 2),))}, 0.5)
        clone.pre_run()
        group.sync_patches.assert_called_once_with(clone)
        self.assertNotEqual(clone.patches_uuid, bound.patches_uuid)
        self.assertEqual(len(clone.get_callbacks(CallbacksMP.ON_PRE_RUN, "minimax_sp_lora")), 1)

    def test_rank0_denies_forward_after_rejected_ack(self):
        group = SPGroup.__new__(SPGroup)
        group.world, group.patch_uuid, group.obj_pg = 2, None, None
        group.check_alive, group.destroy = Mock(), Mock()

        def gather(replies, local, **kwargs):
            replies[:] = [local, {"error": "worker rejected patches"}]

        with patch.object(sync, "memory_budget", return_value=0), \
                patch("minimax_sp.sp_group.dist.broadcast_object_list"), \
                patch("minimax_sp.sp_group.dist.all_gather_object", side_effect=gather):
            with self.assertRaisesRegex(RuntimeError, "preparation failed"):
                group.sync_patches(fake_patcher())
        group.destroy.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "denoise prohibited"):
            group.forward(None, None, None, None, {}, None)

    def test_dead_worker_drops_group_before_next_prompt(self):
        group = SPGroup.__new__(SPGroup)
        group.procs = [SimpleNamespace(poll=lambda: 7, returncode=7)]
        group.log_paths, group.destroy = {1: "worker.log"}, Mock()
        with self.assertRaisesRegex(RuntimeError, "exit 7"):
            group.check_alive()
        group.destroy.assert_called_once()

    def test_comfy_cancellation_is_not_an_exception(self):
        import comfy.model_management
        cancelled = comfy.model_management.InterruptProcessingException
        self.assertFalse(issubclass(cancelled, Exception))
        group = SPGroup.__new__(SPGroup)
        group.world, group.obj_pg, group.patches_ready = 2, None, True
        group.check_alive, group.destroy = Mock(), Mock()
        with patch("minimax_sp.sp_group.build_meta", return_value={}), \
                patch("minimax_sp.sp_group.ordered_tensors", return_value=[]), \
                patch("minimax_sp.sp_group.dist.broadcast_object_list"), \
                patch("minimax_sp.sp_group.spf.sp_forward", side_effect=cancelled):
            with self.assertRaises(cancelled):
                group.forward(None, [torch.zeros(1)], None, None, {}, None)
        group.destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
