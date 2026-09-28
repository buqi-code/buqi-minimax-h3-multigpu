"""Protocol tests without loading H3 weights. Real Turbo runs are separate evidence."""
from dataclasses import replace
from pathlib import Path
import stat
from types import SimpleNamespace
import unittest
from unittest.mock import call, Mock, patch

from bootstrap import setup
setup()
import torch
from minimax_sp import sp_patches as sync
from minimax_sp.sp_group import SPGroup
from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import ModelPatcher


def fake_patcher(with_patches=True):
    patches = {"weight": [(0.7, ("diff", (torch.ones(2),)), 1.0, None, None)]} if with_patches else {}
    return SimpleNamespace(patches=patches, patches_uuid="four",
                           weight_wrapper_patches={}, hook_patches={},
                           get_additional_models_with_key=lambda key: [], offload_device="cpu",
                           model=SimpleNamespace(current_weight_patches_uuid=None), unpatch_model=Mock())


class PatchSyncTests(unittest.TestCase):
    def test_initial_empty_manifest_only_marks_worker_ready(self):
        parent, child = fake_patcher(False), fake_patcher(False)
        worker = sync.WorkerPatches(child, 1)
        with patch.object(sync.comfy.model_management, "load_models_gpu") as load:
            worker.prepare(0, parent.patches_uuid)
            load.reset_mock()
            with sync.export_patches(parent) as manifest:
                self.assertEqual(manifest.path, "")
                reply = worker.apply(sync.PatchManifest.from_message(manifest.message()))
            load.assert_not_called()
        self.assertEqual(reply, manifest.ready(1))
        self.assertEqual(worker.version, parent.patches_uuid)
        child.unpatch_model.assert_not_called()
        worker.require_ready()

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
                    path = Path(manifest.path) if manifest.path else None
                    if path is not None:
                        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                        self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
                    reply = worker.apply(sync.PatchManifest.from_message(manifest.message()))
                    sync.validate_replies(manifest, [manifest.ready(0), reply], 2)
                    self.assertEqual(len(child.patches), keys)
                    if keys:
                        self.assertEqual(child.patches["weight"][0][0], 0.7)
                    worker.require_ready()
                if path is not None:
                    self.assertFalse(path.exists())
        self.assertEqual(child.unpatch_model.call_count, 3)

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
        group.world, group.patch_uuid, group.control_pg = 2, None, None
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
        group.world, group.control_pg, group.tensor_pg = 2, object(), object()
        group.patches_ready, group.operation_id, group.verify_pending = True, 0, False
        group.check_alive, group.destroy = Mock(), Mock()
        meta = {"video": None, "audio": None, "timestep": None, "context": None, "tags": None,
                "cond_video": [], "cond_audio": [], "sample_sigmas": None, "denoise_mask": None,
                "audio_denoise_mask": None, "options": {}}

        def gather(replies, local, **kwargs):
            replies[:] = [local, local]

        with patch("minimax_sp.sp_group.build_meta", return_value=meta), \
                patch("minimax_sp.sp_group.ordered_tensors", return_value=[]), \
                patch("minimax_sp.sp_group.dist.broadcast_object_list"), \
                patch("minimax_sp.sp_group.dist.all_gather_object", side_effect=gather), \
                patch("minimax_sp.sp_group.spf.sp_forward", side_effect=cancelled):
            with self.assertRaises(cancelled):
                group.forward(None, [torch.zeros(1)], None, None, {}, None)
        group.destroy.assert_called_once()

    def test_failed_start_cleans_worker_store_and_groups(self):
        from minimax_sp import sp_group
        worker = Mock()
        worker.poll.return_value = None
        captured = []
        tensor_pg, control_pg = object(), object()

        def fail_start(group, *args):
            captured.append(group)
            group.procs = [worker]
            group.tensor_pg, group.control_pg, group.store = tensor_pg, control_pg, object()
            group._owns_default_pg = True
            raise RuntimeError("startup probe failed")

        with patch.object(SPGroup, "_start", fail_start), \
                patch.object(sp_group.dist, "is_initialized", side_effect=[False, True]), \
                patch.object(sp_group.dist, "destroy_process_group") as cleanup:
            with self.assertRaisesRegex(RuntimeError, "startup probe failed"):
                SPGroup(2, "model", "default", ["0", "1"])
        self.assertEqual(cleanup.call_args_list, [call(tensor_pg), call(control_pg)])
        worker.kill.assert_called_once()
        worker.wait.assert_called_once()
        self.assertIsNone(captured[0].tensor_pg)
        self.assertIsNone(captured[0].control_pg)
        self.assertIsNone(captured[0].store)
        self.assertFalse(captured[0].patches_ready)

    def test_destroy_removes_cached_group_and_is_idempotent(self):
        from minimax_sp import sp_group
        group = SPGroup.__new__(SPGroup)
        tensor_pg, control_pg = object(), object()
        group.procs, group.tensor_pg, group.control_pg, group.store = [], tensor_pg, control_pg, object()
        group._owns_default_pg = True
        with patch.dict(sp_group._GROUPS, {"test": group}, clear=True), \
                patch.object(sp_group.dist, "is_initialized", side_effect=[True, False]), \
                patch.object(sp_group.dist, "destroy_process_group") as cleanup:
            group.destroy()
            group.destroy()
            self.assertFalse(sp_group._GROUPS)
        self.assertEqual(cleanup.call_args_list, [call(tensor_pg), call(control_pg)])
        self.assertIsNone(group.store)
        self.assertIsNone(group.tensor_pg)
        self.assertIsNone(group.control_pg)


if __name__ == "__main__":
    unittest.main()
