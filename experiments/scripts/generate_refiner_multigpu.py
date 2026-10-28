"""Run the unchanged dataset generator with explicit inference-server transport.

The dataset config and original code snapshot remain valid. Runtime routing and
its source snapshot are recorded separately; each raw response records its server.
Only this process's generator-module requests reference is injected, not the
installed requests library. One Runner still owns quotas, sequence IDs and dedup.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading

import requests

import generate_refiner_100k as generator
from refiner_scoped_runner import ScopedRunner


class RoutedResponse:
    def __init__(self, response, endpoint, transport_hash):
        self.response = response
        self.endpoint = endpoint
        self.transport_hash = transport_hash

    def raise_for_status(self):
        return self.response.raise_for_status()

    def json(self):
        value = self.response.json()
        if isinstance(value, dict):
            value = dict(value)
            value['_local_transport'] = {
                'endpoint': self.endpoint, 'source_sha256': self.transport_hash}
        return value


class EndpointTransport:
    """Bound concurrent requests per server and dispatch to the least busy one."""
    def __init__(self, logical_api, endpoints, limit=2, post=None, source_hash='', request_timeout=None):
        self.logical_api = logical_api.rstrip('/')
        self.endpoints = list(dict.fromkeys(e.rstrip('/') for e in endpoints))
        if not self.endpoints or limit < 1:
            raise ValueError('endpoints and positive per-server limit required')
        self.limit = limit
        self.active = Counter()
        self.condition = threading.Condition()
        self.cursor = 0
        self.post_request = post or requests.post
        self.source_hash = source_hash
        self.request_timeout = request_timeout

    def post(self, url, **kwargs):
        prefix = self.logical_api + '/'
        if not url.startswith(prefix):
            raise ValueError('unexpected_logical_api_url')
        path = url[len(self.logical_api):]
        with self.condition:
            self.condition.wait_for(
                lambda: any(self.active[e] < self.limit for e in self.endpoints))
            order = self.endpoints[self.cursor:] + self.endpoints[:self.cursor]
            endpoint = min(order, key=lambda e: self.active[e])
            self.active[endpoint] += 1
            self.cursor = (self.endpoints.index(endpoint) + 1) % len(self.endpoints)
        try:
            if self.request_timeout is not None:
                kwargs['timeout'] = (10, self.request_timeout)
            response = self.post_request(endpoint + path, **kwargs)
            return RoutedResponse(response, endpoint, self.source_hash)
        finally:
            with self.condition:
                self.active[endpoint] -= 1
                self.condition.notify_all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--api', default='http://127.0.0.1:8014/v1')
    parser.add_argument('--extra-api', action='append', default=[])
    parser.add_argument('--model', default='gemma4-31b-teacher')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--per-server-limit', type=int, default=2)
    parser.add_argument('--request-timeout', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026102401)
    parser.add_argument('--stop-after', type=int, default=0)
    parser.add_argument('--categories', help='Comma-separated coverage gaps; other categories are retained but not generated')
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or not 1 <= args.batch_size <= 4 or args.stop_after < 0:
        parser.error('workers 1..32, batch size 1..4, stop-after >=0')
    if not 1 <= args.per_server_limit <= 32:
        parser.error('per-server-limit must be 1..32')
    if not 30 <= args.request_timeout <= 3600:
        parser.error('request-timeout must be 30..3600 seconds')
    endpoints = list(dict.fromkeys([args.api.rstrip('/')] + [x.rstrip('/') for x in args.extra_api]))
    if args.extra_api and len(endpoints) < 2:
        parser.error('at least two distinct server endpoints are required')
    # Verify that replicas identify as the same model before resuming the ledger.
    for endpoint in endpoints:
        if not endpoint.startswith('http://127.0.0.1:') or not endpoint.endswith('/v1'):
            parser.error('this local driver only supports loopback /v1 endpoints')
        health = requests.get(endpoint[:-3] + '/health', timeout=10)
        health.raise_for_status()
        models = requests.get(endpoint + '/models', timeout=10)
        models.raise_for_status()
        if args.model not in {x['id'] for x in models.json()['data']}:
            raise ValueError('replica_model_name_mismatch:' + endpoint)
    if args.categories and (not all(args.categories.split(',')) or set(args.categories.split(',')) - set(generator.QUOTAS)):
        parser.error('categories must be known nonempty class IDs')
    runner = ScopedRunner(args) if args.categories else generator.Runner(args)
    source = Path(__file__).read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    runtime = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_config_hash': runner.config_hash,
        'logical_api_in_dataset_config': args.api,
        'actual_endpoints': endpoints, 'workers': args.workers,
        'per_server_limit': args.per_server_limit, 'batch_size': args.batch_size,
        'request_timeout_seconds': args.request_timeout,
        'active_categories': runner.categories if args.categories else list(generator.QUOTAS),
        'scope_maximum': runner.scope_target if args.categories else sum(generator.QUOTAS.values()),
        'phase_stop_after_additional': args.stop_after or None,
        'scoped_runner_source_sha256': hashlib.sha256(Path(__file__).with_name('refiner_scoped_runner.py').read_bytes()).hexdigest(),
        'transport_source_sha256': source_hash,
        'note': 'Transport-only change; original generation/audit rules, quotas and ledger retained. '
                'Raw responses identify the actual endpoint. Accepted remains pending human review.',
    }
    snapshots = runner.out / 'runtime_snapshots'
    snapshots.mkdir(exist_ok=True)
    (snapshots / f'generate_refiner_multigpu_{source_hash}.py').write_bytes(source)
    scoped_source = Path(__file__).with_name('refiner_scoped_runner.py').read_bytes()
    (snapshots / f'refiner_scoped_runner_{hashlib.sha256(scoped_source).hexdigest()}.py').write_bytes(scoped_source)
    with (runner.out / 'runtime_history.jsonl').open('a') as stream:
        stream.write(json.dumps(runtime, ensure_ascii=False) + '\n')
    generator.atomic_json(runner.out / 'runtime_current.json', runtime)
    generator.requests = EndpointTransport(args.api, endpoints, args.per_server_limit,
                                           source_hash=source_hash, request_timeout=args.request_timeout)
    print(json.dumps(runtime, ensure_ascii=False), flush=True)
    runner.run()


if __name__ == '__main__':
    main()
