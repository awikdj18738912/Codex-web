"""Export packet latency and sending lag over a paced ASR recording."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(directory):
    rows = [json.loads(line) for line in (directory / 'run/results.jsonl').read_text().splitlines() if line]
    assert {r['arm'] for r in rows} == {'raw', 'entity'}
    duration = max(r['duration_seconds'] for r in rows)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, constrained_layout=True)
    colors = {'raw': '#2664aa', 'entity': '#c85b22'}
    labels = {'raw': 'Qwen only', 'entity': 'Qwen + CTC + audio review'}
    for arm in ('raw', 'entity'):
        packets = [p for r in rows if r['arm'] == arm for p in r['chunk_submissions']]
        boundaries = np.arange(0, duration + 10, 10)
        times, request_p95, lag_max = [], [], []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            sample = [p for p in packets if left < p['audio_seconds'] <= right]
            if not sample:
                continue
            times.append((left + min(right, duration)) / 120)
            request_p95.append(1000 * np.percentile([p['chunk_request_seconds'] for p in sample], 95))
            lag_max.append(1000 * max(p['cumulative_schedule_lateness_seconds'] for p in sample))
        axes[0].plot(times, request_p95, color=colors[arm], label=labels[arm], linewidth=1.7)
        axes[1].plot(times, lag_max, color=colors[arm], label=labels[arm], linewidth=1.7)
    axes[0].set_ylabel('Packet request p95 (ms)\nwithin each 10-second bin')
    axes[1].set_ylabel('Maximum sending lag (ms)\nwithin each 10-second bin')
    axes[1].set_xlabel('Input audio position (minutes)')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, duration / 60)
    axes[0].legend(frameon=False, loc='upper right')
    fig.suptitle(f'{duration / 60:g}-minute continuous recording: paced streaming latency', fontsize=14)
    fig.savefig(directory / 'latency_over_time.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    main(parser.parse_args().directory.resolve())
