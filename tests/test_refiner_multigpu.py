import sys
from pathlib import Path
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/scripts'))
from generate_refiner_multigpu import EndpointTransport


class Response:
    def raise_for_status(self):
        pass

    def json(self):
        return {'choices': [{'finish_reason': 'stop'}]}


class TransportTest(unittest.TestCase):
    def test_single_tp2_server_accepts_32_slots_and_runtime_timeout(self):
        calls = []
        def post(url, **kwargs):
            calls.append((url, kwargs))
            return Response()
        transport = EndpointTransport('http://a/v1', ['http://a/v1'], limit=32,
                                      post=post, request_timeout=900)
        transport.post('http://a/v1/chat/completions', timeout=(10, 300))
        self.assertEqual(calls[0][1]['timeout'], (10, 900))
        self.assertEqual(transport.limit, 32)
        self.assertEqual(sum(transport.active.values()), 0)

    def test_both_servers_receive_unmodified_request_and_provenance(self):
        calls = []
        payload = {'seed': 42, 'messages': [{'content': '中文测试'}]}
        def post(url, **kwargs):
            calls.append((url, kwargs))
            return Response()
        transport = EndpointTransport('http://a/v1', ['http://a/v1', 'http://b/v1'],
                                      post=post, source_hash='abc')
        responses = [transport.post('http://a/v1/chat/completions', json=payload, timeout=(10, 300)) for _ in range(2)]
        self.assertEqual([x[0] for x in calls], ['http://a/v1/chat/completions', 'http://b/v1/chat/completions'])
        self.assertTrue(all(x[1]['json'] is payload for x in calls))
        self.assertEqual(responses[1].json()['_local_transport'],
                         {'endpoint': 'http://b/v1', 'source_sha256': 'abc'})
        self.assertEqual(responses[0].json()['choices'][0]['finish_reason'], 'stop')

    def test_failure_releases_capacity_and_next_attempt_uses_other_server(self):
        calls = []
        def post(url, **kwargs):
            calls.append(url)
            if len(calls) == 1:
                raise ConnectionError('offline')
            return Response()
        transport = EndpointTransport('http://a/v1', ['http://a/v1', 'http://b/v1'], limit=1, post=post)
        with self.assertRaises(ConnectionError):
            transport.post('http://a/v1/chat/completions')
        transport.post('http://a/v1/chat/completions')
        self.assertEqual(transport.active['http://a/v1'], 0)
        self.assertTrue(calls[-1].startswith('http://b/'))

    def test_concurrent_requests_obey_each_server_limit(self):
        first_wave = threading.Barrier(5)
        release = threading.Event()
        lock = threading.Lock()
        calls = []
        def post(url, **kwargs):
            with lock:
                calls.append(url)
                initial = len(calls) <= 4
            if initial:
                first_wave.wait(timeout=5)
                if not release.wait(timeout=5):
                    raise TimeoutError('test did not release requests')
            return Response()
        transport = EndpointTransport('http://a/v1', ['http://a/v1', 'http://b/v1'], limit=2, post=post)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(transport.post, 'http://a/v1/chat/completions') for _ in range(6)]
            try:
                first_wave.wait(timeout=5)
                with transport.condition:
                    self.assertEqual(dict(transport.active), {'http://a/v1': 2, 'http://b/v1': 2})
                self.assertEqual(len(calls), 4)
            finally:
                release.set()
            for future in futures:
                future.result(timeout=5)
        self.assertEqual(sum(transport.active.values()), 0)


if __name__ == '__main__':
    unittest.main()
