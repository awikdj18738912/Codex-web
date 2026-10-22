(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const startButton = $('translation-start');
  const stopButton = $('translation-stop');
  const downloadButton = $('translation-download');
  const textTab = $('translation-text-tab');
  const audioTab = $('translation-audio-tab');
  const textView = $('translation-text-view');
  const audioView = $('translation-audio-view');
  const status = $('translation-status');
  const log = $('translation-log');
  const audioFile = $('translation-audio-file');
  let runCounter = 0;
  let activeRun = null;
  let audioReady = false;
  let serviceReady = false;
  let isAudioTab = false;
  let events = [];
  let serializedEvents = [];
  let lastSourceRevision = 0;
  let logRenderScheduled = false;
  let initialHealthComplete = false;

  log.title = '仅展示最近60条，下载包含全部';
  log.setAttribute('aria-label', '翻译事件日志。仅展示最近60条，下载包含全部。');

  function setStatus(message, isError = false) {
    status.textContent = message;
    status.classList.toggle('error', isError);
  }
  function setTab(audio) {
    if (activeRun) return;
    isAudioTab = audio;
    textTab.setAttribute('aria-selected', String(!audio));
    audioTab.setAttribute('aria-selected', String(audio));
    textView.hidden = audio;
    audioView.hidden = !audio;
    startButton.disabled = !serviceReady || (audio && !audioReady);
    startButton.textContent = audio ? '开始音频翻译' : '开始翻译';
  }
  textTab.addEventListener('click', () => setTab(false));
  audioTab.addEventListener('click', () => setTab(true));

  function populateLanguages(languages) {
    if (!Array.isArray(languages) || !languages.length) return;
    for (const id of ['translation-language', 'translation-audio-language']) {
      const select = $(id);
      const previous = select.value;
      select.replaceChildren();
      for (const item of languages) {
        if (!item || !item.code || !item.name) continue;
        const option = document.createElement('option');
        option.value = String(item.code);
        option.textContent = String(item.name);
        select.append(option);
      }
      if ([...select.options].some(option => option.value === previous)) select.value = previous;
      else if ([...select.options].some(option => option.value === 'en')) select.value = 'en';
    }
  }
  async function refreshHealth() {
    try {
      const response = await fetch('/api/translation/health');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      populateLanguages(data.languages);
      serviceReady = data.ready === true;
      audioReady = data.audio_ready === true;
      audioTab.disabled = !audioReady;
      $('translation-audio-help').textContent = audioReady
        ? '选择一个音频文件，服务会自动分块回放并推送翻译结果。'
        : '语音转录服务正在准备，可先使用文本回放。';
      if (!activeRun) {
        if (!events.length || !serviceReady) {
          setStatus(serviceReady
            ? (audioReady ? '翻译与音频服务已就绪。' : '翻译服务已就绪；音频翻译暂不可用。')
            : '翻译服务尚未就绪。', !serviceReady);
        }
        startButton.disabled = !serviceReady || (isAudioTab && !audioReady);
      }
    } catch (error) {
      if (!initialHealthComplete) {
        serviceReady = false;
        audioReady = false;
        audioTab.disabled = true;
        $('translation-audio-help').textContent = '暂时无法连接翻译服务，请稍后重试。';
      }
      if (!activeRun) {
        startButton.disabled = !serviceReady || (isAudioTab && !audioReady);
        setStatus(`无法检查翻译服务：${error.message}`, true);
      }
    } finally {
      initialHealthComplete = true;
    }
  }
  refreshHealth();
  window.setInterval(refreshHealth, 10000);

  function addLog(event) {
    events.push(event);
    serializedEvents.push(JSON.stringify(event));
    if (!logRenderScheduled) {
      logRenderScheduled = true;
      const render = () => {
        logRenderScheduled = false;
        const recent = serializedEvents.slice(-60);
        log.textContent = `仅展示最近60条，下载包含全部\n${recent.length ? recent.join('\n') : '等待服务端事件…'}`;
      };
      if (typeof window.requestAnimationFrame === 'function') window.requestAnimationFrame(render);
      else window.setTimeout(render, 100);
    }
    downloadButton.disabled = events.length === 0;
  }
  function resetRun() {
    events = [];
    serializedEvents = [];
    lastSourceRevision = 0;
    log.textContent = '仅展示最近60条，下载包含全部\n等待服务端事件…';
    $('translation-source').textContent = '';
    $('translation-original').textContent = '';
    $('translation-clean').textContent = '';
    $('translation-calls').textContent = '0 / 0';
    $('translation-guard').textContent = '等待运行';
    $('translation-time').textContent = '—';
    $('refinement-time').textContent = '—';
    downloadButton.disabled = true;
  }
  function wsUrl(path) {
    const url = new URL(path, location.href);
    url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return url.href;
  }
  function setRunning(running, run) {
    if (activeRun !== run) return;
    if (running) {
      startButton.disabled = true;
      stopButton.disabled = false;
    } else {
      activeRun = null;
      stopButton.disabled = true;
      startButton.disabled = !serviceReady || (isAudioTab && !audioReady);
      if (run.asrLock) {
        $('record').disabled = run.asrLock.recordDisabled;
        $('mode').disabled = run.asrLock.modeDisabled;
        run.asrLock = null;
      }
    }
  }
  function updateFromEvent(event) {
    if (['translation', 'update', 'final'].includes(event.event)
      && Number.isFinite(event.revision) && event.revision < lastSourceRevision) return false;
    if (event.event === 'source') {
      const revision = Number.isFinite(event.revision) ? event.revision : lastSourceRevision + 1;
      if (revision > lastSourceRevision) {
        lastSourceRevision = revision;
        $('translation-original').textContent = '等待完整片段…';
        $('translation-clean').textContent = '等待精修…';
        $('translation-guard').textContent = '等待运行';
        $('translation-time').textContent = '—';
        $('refinement-time').textContent = '—';
      }
    }
    if (Number.isFinite(event.audio_processed_seconds) && Number.isFinite(event.audio_total_seconds)) {
      setStatus(`音频已处理 ${event.audio_processed_seconds.toFixed(1)} / ${event.audio_total_seconds.toFixed(1)} 秒`);
    }
    if (event.event === 'source') {
      $('translation-original').textContent = '等待完整片段…';
      $('translation-clean').textContent = '等待精修…';
    }
    if (typeof event.source_text === 'string') $('translation-source').textContent = event.source_text;
    if (typeof event.translation === 'string') $('translation-original').textContent = event.translation;
    if (typeof event.clean_text === 'string') $('translation-clean').textContent = event.clean_text;
    if (event.model_calls !== undefined || event.total_model_calls !== undefined) {
      $('translation-calls').textContent = `${event.model_calls ?? 0} / ${event.total_model_calls ?? 0}`;
    }
    if (event.refinement_state !== undefined) {
      const stateLabels = {
        waiting_for_context: '等待后文', waiting_for_group: '等待完整片段',
        window_refined: '精修完成', waiting_for_complete_segment: '等待完整片段',
        waiting_for_translation: '等待翻译', refining: '精修中', completed: '精修完成',
        complete: '精修完成', accepted: '精修完成', rejected: '守卫拒绝精修', skipped: '跳过精修',
      };
      const reasonLabels = {
        numeric_value_mismatch: '数量或单位校验未通过', numeric_mismatch: '数字校验未通过',
        entity_mismatch: '实体校验未通过', meaning_drift: '语义校验未通过',
        empty_refinement: '精修结果为空', invalid_refinement: '精修结果无效',
      };
      const rawReasons = Array.isArray(event.refiner_reject_reasons) ? event.refiner_reject_reasons : [];
      const reasons = rawReasons.length ? `：${rawReasons.map(item => {
        const key = typeof item === 'string' ? item : (item && (item.code || item.reason || item.key)) || String(item);
        return reasonLabels[key] || key;
      }).join('、')}` : '';
      const accepted = event.refiner_accepted === true ? '已接受' : event.refiner_accepted === false ? '已回退' : '';
      const state = stateLabels[event.refinement_state] || event.refinement_state;
      $('translation-guard').textContent = `${state}${accepted ? ` · ${accepted}` : ''}${reasons}`;
    } else if (event.refiner_executed !== undefined) {
      $('translation-guard').textContent = event.refiner_executed ? (event.refiner_accepted ? '已执行并接受' : '已执行，结果未接受') : '未执行';
    }
    if (event.translation_ms !== undefined) $('translation-time').textContent = `${formatMs(event.translation_ms)} ms`;
    if (event.refinement_ms !== undefined) $('refinement-time').textContent = `${formatMs(event.refinement_ms)} ms`;
    return true;
  }
  function formatMs(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(1) : String(value);
  }
  function connect(path, textPayload, run) {
    const socket = new WebSocket(wsUrl(path));
    run.socket = socket;
    socket.addEventListener('open', () => {
      if (activeRun !== run || run.id !== runCounter) return socket.close();
      if (textPayload) socket.send(JSON.stringify(textPayload));
      setStatus(textPayload ? '已连接，正在回放文本…' : '已连接，正在处理音频…');
    });
    socket.addEventListener('message', message => {
      if (activeRun !== run || run.id !== runCounter) return;
      let event;
      try { event = JSON.parse(message.data); }
      catch (_) { addLog({ event: 'parse_error', detail: String(message.data) }); return; }
      addLog(event);
      if (!updateFromEvent(event)) return;
      if (event.event === 'ready') setStatus('服务已就绪，正在处理…');
      else if (event.event === 'error') {
        setStatus(event.detail || '翻译过程中发生错误。', true);
        run.terminal = true;
      } else if (event.event === 'final') {
        setStatus('翻译与中文精修完成。');
        run.terminal = true;
      } else if (event.event === 'update') setStatus('收到新的翻译与精修结果…');
    });
    socket.addEventListener('error', () => {
      if (activeRun === run && run.id === runCounter) setStatus('WebSocket 连接发生错误。', true);
    });
    socket.addEventListener('close', () => {
      if (activeRun !== run || run.id !== runCounter) return;
      if (!run.terminal && !run.cancelled) setStatus('连接已关闭。', true);
      setRunning(false, run);
    });
  }
  async function startText(run) {
    const text = $('translation-input').value.trim();
    if (!text) throw new Error('请先输入待翻译文本。');
    const language = $('translation-language').value;
    const domain = $('translation-domain').value;
    connect('/ws/translation', { event: 'start', language, domain, text, chunk_chars: 40, interval_ms: 500 }, run);
  }
  async function startAudio(run) {
    const file = audioFile.files && audioFile.files[0];
    if (!file) throw new Error('请先选择音频文件。');
    if (!audioReady) throw new Error('音频翻译服务未就绪。');
    const form = new FormData();
    form.append('file', file);
    form.append('language', $('translation-audio-language').value);
    form.append('domain', $('translation-audio-domain').value);
    form.append('chunk_seconds', '2');
    form.append('interval_ms', '0');
    const controller = new AbortController();
    run.uploadController = controller;
    const response = await fetch('/api/translation/audio', { method: 'POST', body: form, signal: controller.signal });
    run.uploadResponseSeen = true;
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try { const body = await response.json(); detail = body.detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    const result = await response.json();
    if (!result.id || !result.websocket) throw new Error('音频服务未返回任务 ID 或 WebSocket 地址。');
    if (activeRun !== run || run.id !== runCounter || run.cancelled) {
      fetch(`/api/translation/audio/${encodeURIComponent(result.id)}`, { method: 'DELETE' }).catch(() => {});
      return;
    }
    connect(result.websocket, null, run);
  }
  startButton.addEventListener('click', async () => {
    if (activeRun) return;
    const recordButton = $('record');
    const modeSelect = $('mode');
    if (recordButton.classList.contains('recording') || modeSelect.disabled) {
      setStatus('请先结束当前 ASR 转写，再启动翻译。', true);
      return;
    }
    const run = {
      id: ++runCounter, socket: null, terminal: false, cancelled: false,
      asrLock: { recordDisabled: recordButton.disabled, modeDisabled: modeSelect.disabled },
    };
    activeRun = run;
    recordButton.disabled = true;
    modeSelect.disabled = true;
    resetRun();
    setRunning(true, run);
    setStatus(isAudioTab ? '正在上传音频…' : '正在连接翻译服务…');
    try {
      if (isAudioTab) await startAudio(run);
      else await startText(run);
    } catch (error) {
      if (activeRun !== run) return;
      addLog({ event: 'error', detail: error.message });
      run.terminal = true;
      setStatus(error.message, true);
      setRunning(false, run);
      if (run.socket && run.socket.readyState < WebSocket.CLOSING) run.socket.close();
      if (run.uploadController) run.uploadController.abort();
    }
  });
  stopButton.addEventListener('click', () => {
    const run = activeRun;
    if (!run) return;
    run.cancelled = true;
    run.terminal = true;
    setStatus('已取消。');
    if (run.uploadController && !run.uploadResponseSeen) run.uploadController.abort();
    if (run.socket) run.socket.close();
    else setRunning(false, run);
  });
  downloadButton.addEventListener('click', () => {
    if (!events.length) return;
    const blob = new Blob([JSON.stringify(events, null, 2)], { type: 'application/json' });
    const anchor = document.createElement('a');
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `translation-log-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    anchor.click();
    URL.revokeObjectURL(anchor.href);
  });
})();
