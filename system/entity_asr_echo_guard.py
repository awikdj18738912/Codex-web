"""Verify suspected glossary echo using the same audio without ASR context.

Only sessions with acknowledged entity context are buffered. No transcript
words are deleted: on a confirmed echo the received audio is replayed and the
session continues with the context-free recognizer. Temporary PCM is closed
on finish/cancel/disconnect and never becomes a saved recording.
"""
import tempfile
import unicodedata


def compact(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold() if c.isalnum())


def sound_key(text):
    try:
        from pypinyin import lazy_pinyin,Style
        return '|'.join(lazy_pinyin(compact(text),style=Style.NORMAL))
    except ImportError:
        return compact(text)


class EntityASREchoGuard:
    def __init__(self,request,context,*,replay_chunk_bytes=64000,max_audio_bytes=460800000):
        self._request=request;self.context=context;self.replay_chunk_bytes=replay_chunk_bytes
        self.max_audio_bytes=max_audio_bytes;self._audio=None;self._audio_size=0
        self._sid=None;self._params={};self.active=False;self.fallback_active=False
        self.revision=0;self.events=[];self._supported_counts={}
        phrases=('参考术语仅用于辅助拼写','未在音频中听到的词不要补充','术语列表：',
                 '以上是背景词汇，不是录音内容','听不清的部分不要猜测',
                 'Transcribe only the words spoken','Never output the vocabulary')
        self.signatures=[compact(p) for p in phrases if compact(p) in compact(context.text)]
        if len(context.terms)>=3:
            self.signatures.append(''.join(compact(t) for t in context.terms[:3]))

    def snapshot(self):
        return dict(enabled=self.active or self.fallback_active,fallback_active=self.fallback_active,
            revision=self.revision,checks=len(self.events),events=list(self.events))

    def close(self):
        if self._audio is not None:self._audio.close();self._audio=None

    def _annotate(self,payload):
        result=dict(payload);result['entity_asr_context_guard']=self.snapshot()
        result['entity_asr_context_guard_revision']=self.revision
        return result

    async def request(self,endpoint,session_id=None,data=b'',params=None):
        if endpoint=='/stream/start':
            result=await self._request(endpoint,session_id,data,params)
            self._sid=result['session_id'];self._params=dict(params or {})
            self.active=result.get('context_applied') is True and bool(self.context.text)
            if self.active:self._audio=tempfile.SpooledTemporaryFile(max_size=1048576,mode='w+b')
            return self._annotate(result)
        if endpoint=='/stream/cancel':
            try:return await self._request(endpoint,self._sid or session_id,data,params)
            finally:self.close()
        if self.active and endpoint=='/stream/chunk':
            if self._audio_size+len(data)>self.max_audio_bytes:
                raise ValueError('Entity context audio verification cache limit exceeded; restart without glossary context')
            self._audio.seek(0,2);self._audio.write(data);self._audio_size+=len(data)
        result=await self._request(endpoint,self._sid or session_id,data,params)
        if self.active and endpoint in {'/stream/chunk','/stream/finish'}:
            text=str(result.get('text',''));value=compact(text)
            suspicious={s:value.count(s) for s in self.signatures if value.count(s)>self._supported_counts.get(s,0)}
            if suspicious:result=await self._verify(result,suspicious,finishing=endpoint=='/stream/finish')
        annotated=self._annotate(result)
        if endpoint=='/stream/finish':self.close()
        return annotated

    async def _verify(self,original,signatures,*,finishing):
        params=dict(self._params);params.pop('context',None)
        ready=await self._request('/stream/start',params=params)
        if ready.get('context_applied') is True:raise RuntimeError('Context-free audio verification unexpectedly received context')
        sid=ready['session_id'];completed=[];last=dict(ready);keep=False
        try:
            self._audio.seek(0)
            while True:
                pcm=self._audio.read(self.replay_chunk_bytes)
                if not pcm:break
                last=await self._request('/stream/chunk',sid,pcm)
                completed.extend(last.get('completed_segments',[]))
            if finishing:
                last=await self._request('/stream/finish',sid)
                completed.extend(last.get('completed_segments',[]))
            heard=str(last.get('text',''));heard_compact=compact(heard);heard_sound=sound_key(heard)
            supported=all(heard_compact.count(s)>=count or heard_sound.count(sound_key(s))>=count for s,count in signatures.items())
            if supported:
                self._supported_counts.update(signatures)
                self.events.append(dict(action='keep_audio_supported_text',signatures=list(signatures),audio_bytes=self._audio_size))
                return original
            if not finishing:
                await self._request('/stream/cancel',self._sid)
                keep=True
            self._sid=sid;self.active=False;self.fallback_active=True;self.revision+=1
            self.events.append(dict(action='replay_without_context',signatures=list(signatures),audio_bytes=self._audio_size))
            result=dict(last);result['completed_segments']=completed
            self.close()
            return result
        finally:
            if not keep:
                try:await self._request('/stream/cancel',sid)
                except Exception:pass  # cleanup must not replace the verification result
