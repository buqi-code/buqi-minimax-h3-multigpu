"""CPU-only protocol tests for SP process-group fail-fast behavior."""
from datetime import timedelta
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bootstrap import setup
setup()
import torch
from minimax_sp import sp_group as spg
from minimax_sp import sp_worker


def forward_meta(operation_id=1):
    return {
        "op": "forward",
        "operation_id": operation_id,
        "video": {"shape": [1], "dtype": "float32"},
        "audio": None,
        "timestep": None,
        "context": None,
        "tags": None,
        "cond_video": [],
        "cond_audio": [],
        "sample_sigmas": None,
        "denoise_mask": None,
        "audio_denoise_mask": None,
        "keyframes": None,
        "refs": None,
        "frame_count": None,
        "seed": 0,
        "visual_cond_noise_aug": 0,
        "audio_cond_noise_aug": 0,
        "options": {},
        "layout": None,
    }


def successful_gather(replies, local, **kwargs):
    replies[:] = [local for _ in replies]


class SPFailFastTests(unittest.TestCase):
    def test_timeout_defaults_and_overrides(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(spg.positive_timeout("MINIMAX_SP_STARTUP_TIMEOUT", 300), timedelta(seconds=300))
            self.assertEqual(spg.positive_timeout("MINIMAX_SP_COLLECTIVE_TIMEOUT", 120), timedelta(seconds=120))
        with patch.dict(os.environ, {"MINIMAX_SP_STARTUP_TIMEOUT": "17"}, clear=True):
            self.assertEqual(spg.positive_timeout("MINIMAX_SP_STARTUP_TIMEOUT", 300), timedelta(seconds=17))

    def test_timeout_rejects_non_positive_and_invalid_values(self):
        for value in ("0", "-1", "1.5", "soon"):
            with self.subTest(value=value), patch.dict(
                    os.environ, {"MINIMAX_SP_STARTUP_TIMEOUT": value}, clear=True):
                with self.assertRaisesRegex(ValueError, "MINIMAX_SP_STARTUP_TIMEOUT.*positive integer"):
                    spg.positive_timeout("MINIMAX_SP_STARTUP_TIMEOUT", 300)

    def test_startup_probe_uses_tensor_group_and_reports_actionable_failure(self):
        tensor_pg = object()

        def all_reduce(tensor, **kwargs):
            self.assertIs(kwargs["group"], tensor_pg)
            tensor.fill_(3)

        def all_to_all(received, source, **kwargs):
            self.assertIs(kwargs["group"], tensor_pg)
            received.copy_(torch.tensor([0, 2]))

        with patch.object(spg.dist, "all_reduce", side_effect=all_reduce), \
                patch.object(spg.dist, "all_to_all_single", side_effect=all_to_all):
            spg.startup_probe(0, 2, tensor_pg, "cpu")
        with patch.object(spg.dist, "all_reduce", side_effect=RuntimeError("transport error")):
            with self.assertRaisesRegex(RuntimeError, "GPU visibility.*P2P.*NCCL logs"):
                spg.startup_probe(0, 2, tensor_pg, "cpu")

    def test_rank0_and_worker_use_the_same_tensor_group(self):
        tensor_pg, control_pg = object(), object()
        meta = forward_meta()
        source = torch.ones(1)
        group = spg.SPGroup.__new__(spg.SPGroup)
        group.world, group.control_pg, group.tensor_pg = 2, control_pg, tensor_pg
        group.patches_ready, group.operation_id, group.verify_pending = True, 0, False
        group.check_alive, group.destroy = Mock(), Mock()

        with patch.object(spg, "build_meta", return_value=meta), \
                patch.object(spg, "ordered_tensors", return_value=[source]), \
                patch.object(spg.dist, "broadcast_object_list"), \
                patch.object(spg.dist, "all_gather_object", side_effect=successful_gather), \
                patch.object(spg.dist, "broadcast") as rank0_broadcast, \
                patch.object(spg.spf, "sp_forward", return_value="rank0") as rank0_forward:
            self.assertEqual(group.forward(None, [source], None, None, {}, None), "rank0")
        rank0_broadcast.assert_called_once_with(source, src=0, group=tensor_pg)
        self.assertIs(rank0_forward.call_args.args[8], tensor_pg)

        patches = SimpleNamespace(require_ready=Mock())
        worker_spf = SimpleNamespace(sp_forward=Mock())
        with patch.object(sp_worker.dist, "all_gather_object", side_effect=successful_gather), \
                patch.object(sp_worker.dist, "broadcast") as worker_broadcast:
            operation_id = sp_worker.worker_forward(
                forward_meta(), 0, 1, 2, "cpu", control_pg, tensor_pg,
                object(), patches, spg, worker_spf)
        self.assertEqual(operation_id, 1)
        self.assertIs(worker_broadcast.call_args.kwargs["group"], tensor_pg)
        self.assertIs(worker_spf.sp_forward.call_args.args[8], tensor_pg)

    def test_sample_sigmas_uses_validated_tensor_protocol(self):
        meta = forward_meta()
        meta["sample_sigmas"] = {"shape": [3], "dtype": "float32"}
        status = spg.forward_status(meta)
        self.assertEqual(status["tensor_count"], 2)
        self.assertIn(meta["sample_sigmas"], status["tensor_metadata"])

        worker_spf = SimpleNamespace(sp_forward=Mock())
        with patch.object(sp_worker.dist, "all_gather_object", side_effect=successful_gather), \
                patch.object(sp_worker.dist, "broadcast") as broadcast:
            sp_worker.worker_forward(
                meta, 0, 1, 2, "cpu", object(), object(), object(),
                SimpleNamespace(require_ready=Mock()), spg, worker_spf)
        self.assertEqual(broadcast.call_count, 2)
        options = worker_spf.sp_forward.call_args.args[4]
        self.assertEqual(options["sample_sigmas"].device.type, "cpu")
        self.assertEqual(tuple(options["sample_sigmas"].shape), (3,))

    def test_probe_failure_destroys_group(self):
        def start(group, *args):
            group._startup_probe()

        with patch.object(spg.dist, "is_initialized", return_value=False), \
                patch.object(spg.SPGroup, "_start", start), \
                patch.object(spg.SPGroup, "_startup_probe", side_effect=RuntimeError("probe failed")), \
                patch.object(spg.SPGroup, "destroy") as destroy:
            with self.assertRaisesRegex(RuntimeError, "probe failed"):
                spg.SPGroup(2, "model", "default", ["0", "1"])
        destroy.assert_called_once()

    def test_metadata_mismatch_stops_before_rank0_tensor_collective(self):
        tensor_pg, control_pg = object(), object()
        meta = forward_meta()
        source = torch.ones(1)
        group = spg.SPGroup.__new__(spg.SPGroup)
        group.world, group.control_pg, group.tensor_pg = 2, control_pg, tensor_pg
        group.patches_ready, group.operation_id, group.verify_pending = True, 0, False
        group.check_alive, group.destroy = Mock(), Mock()

        def mismatch(replies, local, **kwargs):
            bad = dict(local)
            bad["tensor_metadata"] = []
            replies[:] = [local, bad]

        with patch.object(spg, "build_meta", return_value=meta), \
                patch.object(spg, "ordered_tensors", return_value=[source]), \
                patch.object(spg.dist, "broadcast_object_list"), \
                patch.object(spg.dist, "all_gather_object", side_effect=mismatch), \
                patch.object(spg.dist, "broadcast") as tensor_broadcast, \
                patch.object(spg.spf, "sp_forward") as rank0_forward:
            with self.assertRaisesRegex(RuntimeError, "before tensor collectives"):
                group.forward(None, [source], None, None, {}, None)
        tensor_broadcast.assert_not_called()
        rank0_forward.assert_not_called()
        group.destroy.assert_called_once()

    def test_worker_rejects_skipped_operation_before_tensor_collective(self):
        meta = forward_meta(operation_id=2)

        def gather(replies, local, **kwargs):
            replies[:] = [spg.forward_status(meta), local]

        with patch.object(sp_worker.dist, "all_gather_object", side_effect=gather), \
                patch.object(sp_worker.dist, "broadcast") as tensor_broadcast:
            with self.assertRaisesRegex(RuntimeError, "not ready.*expected operation_id 1"):
                sp_worker.worker_forward(
                    meta, 0, 1, 2, "cpu", object(), object(), object(),
                    SimpleNamespace(require_ready=Mock()), spg, SimpleNamespace(sp_forward=Mock()))
        tensor_broadcast.assert_not_called()

    def test_destroy_does_not_destroy_external_default_group(self):
        with patch.object(spg.dist, "is_initialized", return_value=True), \
                patch.object(spg.dist, "destroy_process_group") as destroy_pg:
            with self.assertRaisesRegex(RuntimeError, "cannot adopt an existing process group"):
                spg.SPGroup(2, "model", "default", ["0", "1"])
            destroy_pg.assert_not_called()

            group = spg.SPGroup.__new__(spg.SPGroup)
            group.procs, group.tensor_pg, group.control_pg = [], None, object()
            group.store, group._owns_default_pg = object(), False
            group.destroy()
            group.destroy()
        destroy_pg.assert_not_called()
        self.assertIsNone(group.control_pg)
        self.assertIsNone(group.store)


if __name__ == "__main__":
    unittest.main()
