"""A failed pilot must never stop an existing service or leak its own helper."""
import json
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

from experiments.scripts import run_gate_template_pilot as pilot


class FakeClient:
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def get(self, url): return SimpleNamespace(raise_for_status=lambda: None)


class OwnershipTests(unittest.TestCase):
    def test_shutdown_timeout_escalates_only_the_owned_popen(self):
        service=MagicMock()
        service.poll.return_value=None
        service.wait.side_effect=[subprocess.TimeoutExpired('owned',20),0]
        self.assertEqual(pilot.stop_owned_service(service),'sigterm_after_grace')
        service.send_signal.assert_called_once_with(signal.SIGINT)
        service.terminate.assert_called_once()
        service.kill.assert_not_called()

    def test_slow_termination_has_bounded_kill_fallback(self):
        service=MagicMock()
        service.poll.return_value=None
        service.wait.side_effect=[subprocess.TimeoutExpired('owned',20),subprocess.TimeoutExpired('owned',10),0]
        self.assertEqual(pilot.stop_owned_service(service),'sigkill_after_grace')
        service.kill.assert_called_once()
        self.assertEqual(service.wait.call_args.kwargs,{'timeout':5})

    def setup_paths(self, folder):
        root = Path(folder)
        source, model = root/'source', root/'model'
        source.mkdir(); model.mkdir(); (source/'preparation_sources').mkdir()
        (model/'gate.pt').write_bytes(b'fixture-not-used-for-inference')
        (model/'metadata.json').write_text('{}')
        return source, model

    def test_existing_port_is_not_replaced_or_signaled(self):
        with tempfile.TemporaryDirectory() as folder:
            source, model = self.setup_paths(folder)
            probe = MagicMock(); probe.__enter__.return_value.connect_ex.return_value = 0
            with patch('sys.argv',['pilot','--source',str(source),'--checkpoint',str(model)]), \
                 patch.object(pilot.data,'seeds',return_value=[dict(synthetic_split='train')]), \
                 patch('httpx.Client',FakeClient), patch.object(pilot.socket,'socket',return_value=probe), \
                 patch.object(pilot.subprocess,'Popen') as start:
                with self.assertRaisesRegex(RuntimeError,'already owned by another process'):
                    pilot.main()
                start.assert_not_called()
            state=json.loads((source/'job_state.json').read_text())
            self.assertEqual(state['status'],'failed')
            self.assertNotIn('owned_tts_pid',state)

    def test_synthesis_failure_stops_only_owned_helper(self):
        with tempfile.TemporaryDirectory() as folder:
            source, model = self.setup_paths(folder)
            probe = MagicMock(); probe.__enter__.return_value.connect_ex.return_value = 1
            service=MagicMock(pid=4321)
            service.poll.return_value=None
            failure=subprocess.CalledProcessError(1,['fixture-synthesize'])
            with patch('sys.argv',['pilot','--source',str(source),'--checkpoint',str(model)]), \
                 patch.object(pilot.data,'seeds',return_value=[dict(synthetic_split='train')]), \
                 patch('httpx.Client',FakeClient), patch.object(pilot.socket,'socket',return_value=probe), \
                 patch.object(pilot.subprocess,'Popen',return_value=service), \
                 patch.object(pilot.subprocess,'run',side_effect=failure):
                with self.assertRaises(subprocess.CalledProcessError): pilot.main()
            service.send_signal.assert_called_once_with(signal.SIGINT)
            service.wait.assert_called_once_with(timeout=20)
            state=json.loads((source/'job_state.json').read_text())
            self.assertEqual(state['status'],'failed')
            self.assertEqual(state['stage'],'synthesize')
            self.assertEqual(state['owned_tts_pid'],4321)


if __name__ == '__main__': unittest.main()
