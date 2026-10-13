from experiments.scripts.stream_linear_gate_holdout import policy_order,paced_session


def test_each_clip_has_both_policy_orders_across_repetitions():
    for identity in ('clip_a','clip_b','real_speech'):
        assert policy_order(identity,0)==list(reversed(policy_order(identity,1)))


def test_paced_transport_reports_audio_clock_and_bounded_ack_backlog():
    # Transport fixture, not speech or ASR evidence.
    import queue
    class Socket:
        def __init__(self):self.messages=queue.Queue();self.messages.put({'event':'ready'});self.sent=0
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def receive_json(self):return self.messages.get(timeout=2)
        def send_bytes(self,_):
            self.sent+=1;self.messages.put(dict(event='chunk_ack',chunk_index=self.sent))
        def send_json(self,_):self.messages.put(dict(event='final',clean_text='fixture'))
    class Client:
        def websocket_connect(self,_):return Socket()
    final,_,timing=paced_session(Client(),bytes(160*4),chunk_samples=80)
    assert final['clean_text']=='fixture'
    assert timing['audio_seconds']==.01
    assert timing['sent_chunks']==timing['acknowledged_chunks']==2
    assert timing['max_inflight_chunks']<=2
    assert timing['completion_lag_from_audio_clock_seconds']>=timing['finish_tail_seconds']-.002
