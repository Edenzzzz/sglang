import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.io_struct import GetLoadReqOutput, LoadSnapshotUpdate
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_communicator_mixin import (
    TokenizerCommunicatorMixin,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager


def _load(dp_rank=0, num_reqs=1):
    return GetLoadReqOutput(
        dp_rank=dp_rank,
        num_reqs=num_reqs,
        num_waiting_reqs=0,
        num_tokens=10,
    )


class TestLoadSnapshot(unittest.IsolatedAsyncioTestCase):
    async def test_get_load_returns_cached_snapshots_without_communicator(self):
        communicator = Mock()
        manager = SimpleNamespace(
            cached_load_snapshots={1: _load(1, 2), 0: _load(0, 1)},
            get_load_communicator=communicator,
        )

        loads = await TokenizerCommunicatorMixin.get_load(manager)

        self.assertEqual([load.dp_rank for load in loads], [0, 1])
        self.assertEqual([load.num_reqs for load in loads], [1, 2])
        communicator.assert_not_called()

    async def test_get_load_falls_back_until_first_publication(self):
        async def communicate(_):
            return [_load(0, 3)]

        manager = SimpleNamespace(
            cached_load_snapshots={},
            get_load_communicator=communicate,
        )

        loads = await TokenizerCommunicatorMixin.get_load(manager)

        self.assertEqual(loads[0].num_reqs, 3)
        self.assertEqual(manager.cached_load_snapshots[0].num_reqs, 3)

    def test_unsolicited_update_replaces_rank_snapshot(self):
        manager = SimpleNamespace(cached_load_snapshots={0: _load(0, 1)})

        TokenizerManager._handle_load_snapshot_update(
            manager, LoadSnapshotUpdate(load=_load(0, 4))
        )

        self.assertEqual(manager.cached_load_snapshots[0].num_reqs, 4)

    def test_scheduler_publication_is_rate_limited(self):
        sender = Mock()
        load = _load()
        scheduler = SimpleNamespace(
            enable_load_snapshot_publication=True,
            next_load_snapshot_time=0.0,
            load_snapshot_interval=0.1,
            send_to_tokenizer=sender,
            get_load=Mock(return_value=load),
        )

        with patch(
            "sglang.srt.managers.scheduler.time.monotonic",
            side_effect=[1.0, 1.05, 1.11],
        ):
            Scheduler.publish_load_snapshot_if_due(scheduler)
            Scheduler.publish_load_snapshot_if_due(scheduler)
            Scheduler.publish_load_snapshot_if_due(scheduler)

        self.assertEqual(sender.send_output.call_count, 2)
        scheduler.get_load.assert_called_with()

    def test_scheduler_does_not_publish_without_explicit_checkpoint(self):
        scheduler = SimpleNamespace(
            enable_load_snapshot_publication=False,
            send_to_tokenizer=Mock(),
            get_load=Mock(),
        )

        Scheduler.publish_load_snapshot_if_due(scheduler)

        scheduler.send_to_tokenizer.send_output.assert_not_called()
        scheduler.get_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
