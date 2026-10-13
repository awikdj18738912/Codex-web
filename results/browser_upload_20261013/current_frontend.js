
    const recordButton = document.querySelector('#record');
    const audioFileInput = document.querySelector('#audio-file');
    const modeSelect = document.querySelector('#mode');
    const transcriptionDomain = document.querySelector('#transcription-domain');
    const entityAsrContext = document.querySelector('#entity-asr-context');
    const entityAudioRepair = document.querySelector('#entity-audio-repair');
    let entityContextTouched = false;
    let entityAudioTouched = false;
    entityAsrContext.addEventListener('change', () => {
      entityContextTouched = true;
      if (entityAsrContext.checked) { entityAudioRepair.checked = false; entityAudioTouched = true; }
    });
    entityAudioRepair.addEventListener('change', () => {
      entityAudioTouched = true;
      if (entityAudioRepair.checked) { entityAsrContext.checked = false; entityContextTouched = true; }
    });
    fetch('/health').then(response => response.json()).then(data => {
      if (!entityContextTouched && !running && !finishing) {
        entityAsrContext.checked = data.entity_asr_context_enabled === true;
      }
      if (!entityAudioTouched && !running && !finishing) {
        entityAudioRepair.checked = data.entity_audio_repair_enabled === true && !entityAsrContext.checked;
      }
    }).catch(() => {});
    const status = document.querySelector('#status');
    const timer = document.querySelector('#timer');
    const raw = document.querySelector('#raw');
    const clean = document.querySelector('#clean');
    const refinerExecuted = document.querySelector('#refiner-executed');
    const refinerResult = document.querySelector('#refiner-result');
    const refinerType = document.querySelector('#refiner-type');
    const refinerCallCount = document.querySelector('#refiner-call-count');
    const gateSkipCount = document.querySelector('#gate-skip-count');
    const historyCount = document.querySelector('#history-count');
    const historyEmpty = document.querySelector('#history-empty');
    const refinementList = document.querySelector('#refinement-list');
    const manageEntitiesButton = document.querySelector('#manage-entities');
    const entityPanel = document.querySelector('#entity-panel');
    const closeEntitiesButton = document.querySelector('#close-entities');
    const refreshEntitiesButton = document.querySelector('#refresh-entities');
    const addEntityButton = document.querySelector('#add-entity');
    const entitySearch = document.querySelector('#entity-search');
    const entityTypeFilter = document.querySelector('#entity-type-filter');
    const entityStatusFilter = document.querySelector('#entity-status-filter');
    const entityRows = document.querySelector('#entity-rows');
    const entityNotice = document.querySelector('#entity-notice');
    const entityDialog = document.querySelector('#entity-dialog');
    const entityForm = document.querySelector('#entity-form');
    const entityDialogTitle = document.querySelector('#entity-dialog-title');
    const entityFormError = document.querySelector('#entity-form-error');
    const entityCanonical = document.querySelector('#entity-canonical');
    const entityType = document.querySelector('#entity-type');
    const entityDomain = document.querySelector('#entity-domain');
    const entityPolicy = document.querySelector('#entity-policy');
    const entityPriority = document.querySelector('#entity-priority');
    const entityAliases = document.querySelector('#entity-aliases');
    const entityEnabled = document.querySelector('#entity-enabled');
    const targetRate = 16000;
    // Microphone updates remain responsive at 0.5 s. File streaming uses the
    // Qwen backend's native 1 s chunk size, halving request overhead on long
    // recordings without changing the model's streaming cadence.
    const microphoneChunkSamples = 8000;
    const fileChunkSamples = 16000;
    const FILE_UPLOAD_WINDOW = 4;
    const FILE_CHUNK_TIMEOUT_MS = 330000;
    // ASR finalization can take longer than a single Refiner pass (notably
    // for Whisper rolling windows). Keep the client alive long enough to
    // receive the explicit raw-transcript and refinement status events.
    // The final Refiner pass scales with the number of text segments.  The
    // backend sends periodic progress events and has its own hard deadline.
    const FINISH_WATCHDOG_MS = 360000;
    let context, source, processor, stream, socket, buffered = new Float32Array(0), running = false, finishing = false, startedAt, ticker, finishWatchdog, selectedMode = 'online', entities = [], refinementHistory = [], seenRefinementChanges = new Set(), latestRefinedText = '', lastAppliedRefinementRevision = 0;
    let fileRunSerial = 0, currentFileTiming = null;
    const fileTimingLabel = document.querySelector('#file-timing');
    const downloadFileTiming = document.querySelector('#download-file-timing');

    function saveFileTiming(outcome) {
      const timing = currentFileTiming;
      if (!timing || timing.finishedAt != null) return;
      timing.finishedAt = performance.now();
      const duration = (end, start) => end != null && start != null
        ? Math.round((end - start) * 1000) / 1000 : null;
      const summary = {
        schema_version: 1, outcome, trace_id: timing.traceId || null,
        mode: timing.mode, entity_asr_context: timing.contextEnabled,
        entity_audio_repair: timing.repairEnabled, upload_window: FILE_UPLOAD_WINDOW,
        audio_seconds: timing.samples == null ? null : timing.samples / targetRate,
        total_chunks: timing.totalChunks || 0, acknowledged_chunks: timing.acks.length,
        preparation_ms: duration(timing.firstSendAt, timing.startedAt),
        decode_ms: duration(timing.decodedAt, timing.startedAt),
        upload_and_recognition_ms: duration(timing.allAckedAt, timing.firstSendAt),
        finish_wait_ms: duration(timing.finishedAt, timing.finishSentAt),
        total_ms: duration(timing.finishedAt, timing.startedAt),
        message_handler_ms: timing.handlerMs, messages: timing.messages,
        chunks: timing.acks.map(ack => ({
          index: ack.index, sent_ms: duration(ack.sentAt, timing.startedAt),
          acknowledged_ms: duration(ack.ackAt, timing.startedAt),
          round_trip_ms: duration(ack.ackAt, ack.sentAt), bytes: ack.bytes,
        })),
      };
      // Client-only diagnostics: no transcript/audio is included or uploaded.
      window.__asrLastFileTransfer = summary;
      const seconds = value => value == null ? '—' : (value / 1000).toFixed(2);
      fileTimingLabel.textContent = `文件准备 ${seconds(summary.preparation_ms)} 秒 · 传输与识别 ${seconds(summary.upload_and_recognition_ms)} 秒 · 收尾 ${seconds(summary.finish_wait_ms)} 秒`;
      fileTimingLabel.hidden = false;
      downloadFileTiming.hidden = false;
    }

    downloadFileTiming.addEventListener('click', () => {
      if (!window.__asrLastFileTransfer) return;
      const url = URL.createObjectURL(new Blob([
        JSON.stringify(window.__asrLastFileTransfer, null, 2),
      ], { type: 'application/json' }));
      const link = document.createElement('a');
      link.href = url; link.download = 'asr-file-timing.json'; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });

    function escapeHtml(value) {
      return String(value).replace(/[&<>'"]/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[character]);
    }

    function setEntityNotice(value = '', isError = false) {
      entityNotice.textContent = value;
      entityNotice.classList.toggle('error', isError);
    }

    async function entityApi(path, options = {}) {
      const response = await fetch(path, options);
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || '术语库请求失败');
      return payload;
    }

    function displayDate(value) {
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString('zh-CN', { hour12: false });
    }

    function renderEntities() {
      if (!entities.length) {
        entityRows.innerHTML = '<tr><td colspan="8">没有符合条件的实体。</td></tr>';
        return;
      }
      entityRows.innerHTML = entities.map(entity => {
        const aliases = entity.aliases.length
          ? entity.aliases.map(alias => `<span class="chip">${escapeHtml(alias)}</span>`).join('')
          : '—';
        const status = entity.enabled
          ? '<span class="chip">启用</span>'
          : '<span class="chip disabled">停用</span>';
        const action = entity.enabled ? '停用' : '启用';
        return `<tr class="${entity.enabled ? '' : 'muted-row'}">
          <td>${status}</td><td>${escapeHtml(entity.canonical_text)}</td><td><div class="chips">${aliases}</div></td>
          <td>${escapeHtml(entity.entity_type)}<br><span class="field-help">${escapeHtml(entity.domain)}</span></td>
          <td>${entity.normalization_policy === 'normalize' ? '统一标准名' : '保留原写法'}</td><td>${entity.priority}</td>
          <td>${displayDate(entity.updated_at)}</td>
          <td><div class="row-actions"><button class="small-button" type="button" data-action="edit" data-id="${entity.entity_id}">编辑</button><button class="small-button" type="button" data-action="toggle" data-id="${entity.entity_id}">${action}</button><button class="small-button danger" type="button" data-action="delete" data-id="${entity.entity_id}">删除</button></div></td>
        </tr>`;
      }).join('');
    }

    async function loadEntities() {
      setEntityNotice('正在加载本地术语库…');
      const params = new URLSearchParams({ include_disabled: 'true' });
      if (entitySearch.value.trim()) params.set('query', entitySearch.value.trim());
      if (entityTypeFilter.value) params.set('entity_type', entityTypeFilter.value);
      try {
        const payload = await entityApi(`/api/entities?${params.toString()}`);
        entities = payload.entities.filter(entity => entityStatusFilter.value === 'all' || (entityStatusFilter.value === 'enabled' ? entity.enabled : !entity.enabled));
        renderEntities();
        setEntityNotice(`已加载 ${entities.length} 条实体。`);
      } catch (error) {
        entities = []; renderEntities(); setEntityNotice(error.message, true);
      }
    }

    function openEntityDialog(entity = null) {
      entityForm.dataset.entityId = entity ? String(entity.entity_id) : '';
      entityDialogTitle.textContent = entity ? '编辑实体' : '新增实体';
      entityCanonical.value = entity ? entity.canonical_text : '';
      entityType.value = entity ? entity.entity_type : 'TERM';
      entityDomain.value = entity ? entity.domain : 'general';
      entityPolicy.value = entity ? entity.normalization_policy : 'preserve';
      entityPriority.value = entity ? String(entity.priority) : '0';
      entityAliases.value = entity ? entity.aliases.join('\n') : '';
      entityEnabled.checked = entity ? entity.enabled : true;
      entityFormError.textContent = '';
      entityDialog.showModal();
      entityCanonical.focus();
    }

    function setStatus(value, isError = false) {
      status.textContent = value;
      status.classList.toggle('error', isError);
    }

    function setButton(isRecording, disabled = false) {
      const startLabels = {
        online: '开始在线转写',
        offline: '选择整段音频文件',
        streaming: '选择流式音频文件',
      };
      const startLabel = startLabels[isRecording ? selectedMode : modeSelect.value] || startLabels.online;
      recordButton.disabled = disabled;
      recordButton.classList.toggle('recording', isRecording);
      recordButton.title = isRecording ? '停止转写' : startLabel;
      recordButton.innerHTML = isRecording
        ? '<i data-lucide="square"></i><span>停止转写</span>'
        : `<i data-lucide="mic"></i><span>${startLabel}</span>`;
      lucide.createIcons();
    }

    function resetRefinerStatus() {
      refinerExecuted.textContent = '未执行';
      refinerResult.textContent = '等待识别文本';
      refinerResult.classList.remove('warning');
      refinerType.textContent = '—';
      refinerCallCount.textContent = '0';
      gateSkipCount.textContent = '0';
    }

    function updateSessionMetrics(data) {
      const stats = data && typeof data.refiner_session_stats === 'object'
        ? data.refiner_session_stats
        : null;
      const calls = stats ? Number(stats.call_count) : Number(data.session_refiner_call_count);
      const skips = stats ? Number(stats.gate_skipped_segment_count) : Number(data.session_gate_skipped_segment_count);
      if (Number.isFinite(calls) && calls >= 0) refinerCallCount.textContent = String(Math.trunc(calls));
      if (Number.isFinite(skips) && skips >= 0) gateSkipCount.textContent = String(Math.trunc(skips));
    }

    function refinementStageLabel(data) {
      if (data.event === 'final') return '最终精修';
      if (selectedMode === 'streaming') return '流式异步精修';
      return '增量精修';
    }

    function sentenceParts(text) {
      return text.match(/[^。！？!?；;\n]+[。！？!?；;\n]?/g) || (text ? [text] : []);
    }

    function sentenceMatches(rawParts, cleanParts) {
      const table = Array.from({ length: rawParts.length + 1 }, () => new Uint16Array(cleanParts.length + 1));
      for (let row = rawParts.length - 1; row >= 0; row -= 1) {
        for (let column = cleanParts.length - 1; column >= 0; column -= 1) {
          table[row][column] = rawParts[row] === cleanParts[column]
            ? table[row + 1][column + 1] + 1
            : Math.max(table[row + 1][column], table[row][column + 1]);
        }
      }
      const matches = [];
      let row = 0; let column = 0;
      while (row < rawParts.length && column < cleanParts.length) {
        if (rawParts[row] === cleanParts[column]) { matches.push([row, column]); row += 1; column += 1; }
        else if (table[row + 1][column] >= table[row][column + 1]) row += 1;
        else column += 1;
      }
      return matches;
    }

    function limitChangedText(value) {
      if (value.length <= 300) return escapeHtml(value);
      return escapeHtml(value.slice(0, 145)) + '…' + escapeHtml(value.slice(-145));
    }

    function compactChange(rawText, cleanText) {
      let prefix = 0;
      while (prefix < rawText.length && prefix < cleanText.length && rawText[prefix] === cleanText[prefix]) prefix += 1;
      let rawEnd = rawText.length;
      let cleanEnd = cleanText.length;
      while (rawEnd > prefix && cleanEnd > prefix && rawText[rawEnd - 1] === cleanText[cleanEnd - 1]) { rawEnd -= 1; cleanEnd -= 1; }
      const beforeStart = Math.max(0, prefix - 16);
      const afterEnd = Math.min(rawText.length, rawEnd + 16);
      const before = (beforeStart ? '…' : '') + escapeHtml(rawText.slice(beforeStart, prefix));
      const after = escapeHtml(rawText.slice(rawEnd, afterEnd)) + (afterEnd < rawText.length ? '…' : '');
      const rawChanged = rawText.slice(prefix, rawEnd);
      const cleanChanged = cleanText.slice(prefix, cleanEnd);
      return {
        key: `${rawChanged}\u0000${cleanChanged}`,
        raw: before + (rawChanged ? `<del class="diff-delete">${limitChangedText(rawChanged)}</del>` : '') + after,
        clean: before + (cleanChanged ? `<ins class="diff-insert">${limitChangedText(cleanChanged)}</ins>` : '') + after,
      };
    }

    function collectChangeHunks(rawText, cleanText) {
      if (rawText === cleanText) return [];
      const rawParts = sentenceParts(rawText);
      const cleanParts = sentenceParts(cleanText);
      const matches = [...sentenceMatches(rawParts, cleanParts), [rawParts.length, cleanParts.length]];
      const changes = [];
      let rawStart = 0; let cleanStart = 0;
      for (const [rawEnd, cleanEnd] of matches) {
        const rawBlock = rawParts.slice(rawStart, rawEnd).join('');
        const cleanBlock = cleanParts.slice(cleanStart, cleanEnd).join('');
        if (rawBlock || cleanBlock) changes.push(compactChange(rawBlock, cleanBlock));
        rawStart = rawEnd + 1;
        cleanStart = cleanEnd + 1;
      }
      return changes;
    }

    function renderRefinementHistory() {
      historyCount.textContent = `${refinementHistory.length} 处有效修改`;
      historyEmpty.hidden = refinementHistory.length > 0;
      refinementList.innerHTML = refinementHistory.map((entry, index) => {
        const notes = [];
        if (entry.entityChanges.length) notes.push(...entry.entityChanges.map(change => `<span class="entity-change">${change.stage === 'asr_audio_repair' ? '原音实体修正' : '实体规范'}：${escapeHtml(change.observed)} → ${escapeHtml(change.canonical)}</span>`));
        if (entry.entityCandidates.length) notes.push(`<details><summary>实体候选 ${entry.entityCandidates.length} 条（Matcher ${entry.matcherLatency} ms）</summary>${entry.entityCandidates.map(candidate => `<span>${escapeHtml(candidate.observed)} → ${escapeHtml(candidate.canonical)} · 拼音 ${Number(candidate.pinyin_score || 0).toFixed(2)} / 字符 ${Number(candidate.char_score || 0).toFixed(2)} / 总分 ${Number(candidate.final_score || 0).toFixed(3)} / 间隔 ${candidate.margin == null ? '—' : Number(candidate.margin).toFixed(3)} · ${escapeHtml(candidate.decision)}</span>`).join('')}</details>`);
        if (entry.entityHints.length) notes.push(`<span>术语提示：${escapeHtml(entry.entityHints.join('、'))}</span>`);
        if (entry.detectedEntities.length) notes.push(`<span>命中实体：${escapeHtml(entry.detectedEntities.join('、'))}</span>`);
        if (entry.auditIssues.length) notes.push(`<span class="audit-note">审计提示：${escapeHtml(entry.auditIssues.join('、'))}</span>`);
        if (entry.rejectReasons.length) notes.push(`<span class="quality-note">质量回退：${escapeHtml(entry.rejectReasons.join('、'))}</span>`);
        if (!notes.length) notes.push('<span>常规文本精修。</span>');
        return `<article class="refinement-card ${entry.final ? 'final' : ''}">
          <div class="history-meta"><span class="stage-badge ${entry.final ? 'final' : ''}">${entry.stage}</span><span>第 ${index + 1} 处</span><span>${entry.time}</span><span>Refiner ${entry.latency} ms</span></div>
          <div class="change-list">${entry.changes.map(change => `<div class="change-hunk"><div class="change-row"><span class="change-label">原文</span><span>${change.raw || '—'}</span></div><div class="change-row"><span class="change-label">精修</span><span>${change.clean || '—'}</span></div></div>`).join('') || '<p class="change-empty">本条没有可展示的文字替换，仅记录实体或质量状态。</p>'}</div>
          <div class="history-note">${notes.join('')}</div>
        </article>`;
      }).join('');
    }

    function resetRefinementHistory() {
      refinementHistory = [];
      seenRefinementChanges = new Set();
      renderRefinementHistory();
    }

    function recordRefinementStep(data) {
      if (typeof data.raw_text !== 'string' || typeof data.clean_text !== 'string' || data.refiner_latency_ms === undefined) return;
      const entities = Array.isArray(data.protected_entities) ? data.protected_entities : [];
      const changes = collectChangeHunks(data.raw_text, data.clean_text).filter(change => {
        const key = `text:${change.key}`;
        if (seenRefinementChanges.has(key)) return false;
        seenRefinementChanges.add(key);
        return true;
      });
      const entityChanges = (Array.isArray(data.entity_normalizations) ? data.entity_normalizations : []).filter(change => {
        const key = `entity:${change.observed}\u0000${change.canonical}`;
        if (seenRefinementChanges.has(key)) return false;
        seenRefinementChanges.add(key);
        return true;
      });
      const rejectReasons = (Array.isArray(data.refiner_reject_reasons) ? data.refiner_reject_reasons : []).filter(reason => {
        const key = `quality:${reason}`;
        if (seenRefinementChanges.has(key)) return false;
        seenRefinementChanges.add(key);
        return true;
      });
      const entityCandidates = (Array.isArray(data.entity_candidates) ? data.entity_candidates : []).filter(candidate => {
        const key = `candidate:${candidate.start}:${candidate.end}:${candidate.entity_id}:${candidate.decision}`;
        if (seenRefinementChanges.has(key)) return false;
        seenRefinementChanges.add(key);
        return true;
      });
      if (!changes.length && !entityChanges.length && !rejectReasons.length && !entityCandidates.length) {
        if (data.event === 'final' && refinementHistory.length) {
          refinementHistory[refinementHistory.length - 1].final = true;
          refinementHistory[refinementHistory.length - 1].stage = refinementStageLabel(data);
          renderRefinementHistory();
        }
        return;
      }
      refinementHistory.push({
        changes,
        stage: refinementStageLabel(data),
        final: data.event === 'final',
        latency: data.refiner_latency_ms,
        time: new Date().toLocaleTimeString('zh-CN', { hour12: false }),
        entityHints: Array.isArray(data.entity_refinement_hints) ? data.entity_refinement_hints : [],
        entityChanges,
        entityCandidates,
        matcherLatency: Number(data.entity_matcher_latency_ms || 0).toFixed(3),
        detectedEntities: entities.filter(entity => /^(database|fuzzy):/.test(String(entity.source || ''))).map(entity => `${entity.original}（${entity.replacement}）`),
        auditIssues: Array.isArray(data.entity_audit_issues) ? data.entity_audit_issues : [],
        rejectReasons,
      });
      renderRefinementHistory();
    }

    function updateRefinerStatus(data) {
      if (data.refiner_latency_ms === undefined) return;
      const isFinal = data.event === 'final';
      const modeType = selectedMode === 'offline'
        ? '离线最终精修'
        : (selectedMode === 'streaming'
          ? (isFinal ? '流式最终精修' : '流式异步精修')
          : (isFinal ? '在线最终精修' : '在线增量精修'));
      const auditIssues = Array.isArray(data.entity_audit_issues) ? data.entity_audit_issues : [];
      const entityHints = Array.isArray(data.entity_refinement_hints) ? data.entity_refinement_hints : [];
      const entityChanges = Array.isArray(data.entity_normalizations) ? data.entity_normalizations : [];
      const pendingChunks = Number(data.pending_refinement_chunks || 0);
      const refinementBatchSize = Number(data.refinement_batch_size || data.window_size || 3);
      if (!isFinal && data.refinement_state === 'waiting_for_group') {
        refinerExecuted.textContent = '流式精修已发布';
        refinerResult.textContent = `等待凑齐 ${refinementBatchSize} 个片段 · 当前还有 ${pendingChunks} 个待精修`;
        refinerResult.classList.remove('warning');
        refinerType.textContent = `${modeType} · 分组窗口 ${refinementBatchSize}`;
        return;
      }
      if (data.refiner_executed === false) {
        const decisions = Array.isArray(data.refinement_gate_decisions) ? data.refinement_gate_decisions : [];
        const skipped = decisions.filter(item => item.action === 'skip');
        const reasons = [...new Set(skipped.flatMap(item => Array.isArray(item.reasons) ? item.reasons : []))];
        refinerExecuted.textContent = `门控跳过 · ${skipped.length || data.refinement_gate_skipped_segments || 0} 段`;
        refinerResult.textContent = reasons.length
          ? `未调用模型，已保留确定性结果：${reasons.join('、')}`
          : '未调用模型，已保留确定性结果';
        refinerResult.classList.remove('warning');
        refinerType.textContent = `${modeType} · ${data.refinement_gate_mode || 'conservative'} 门控`;
        return;
      }
      refinerExecuted.textContent = `已执行 · ${data.refiner_latency_ms} ms`;
      refinerType.textContent = modeType;
      if (data.refiner_accepted === false) {
        const reasons = Array.isArray(data.refiner_reject_reasons) ? data.refiner_reject_reasons : [];
        refinerResult.textContent = reasons.length
          ? `检测到异常精修片段，已回退对应原文：${reasons.join('、')}`
          : '未通过，已回退原始文本';
        refinerResult.classList.add('warning');
      } else if (entityChanges.length) {
        refinerResult.textContent = '已完成 · 实体已规范：' + entityChanges.map(change => `${change.observed} → ${change.canonical}`).join('；');
        refinerResult.classList.remove('warning');
        refinerType.textContent += ' · 实体约束';
      } else if (entityHints.length) {
        refinerResult.textContent = '已完成 · 已使用实体提示：' + entityHints.join('、');
        refinerResult.classList.remove('warning');
        refinerType.textContent += ' · 实体约束';
      } else if (auditIssues.length) {
        refinerResult.textContent = `已完成 · 实体审计提示：${auditIssues.join('、')}`;
        refinerResult.classList.add('warning');
      } else {
        refinerResult.textContent = isFinal ? '已完成' : (finishing ? '流式窗口已精修，等待最终结果' : '活动窗口已精修，转录继续中');
        refinerResult.classList.remove('warning');
      }
    }

    function updateTimer() {
      const seconds = Math.floor((Date.now() - startedAt) / 1000);
      timer.textContent = `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
    }

    function completeTranscription(status = '已完成', isError = false, preserveRaw = false) {
      saveFileTiming(isError ? 'error' : (preserveRaw ? 'fallback' : 'complete'));
      if (preserveRaw) {
        const rawText = raw.textContent.trim();
        // A partial/older refinement is not a valid final result. When the
        // final event is missing, always show the complete ASR text instead
        // of leaving a stale intermediate fragment in the refined pane.
        if (rawText) {
          clean.textContent = raw.textContent;
          refinerResult.textContent = '最终精修未返回，已保留完整原始转写';
          refinerResult.classList.add('warning');
        }
      }
      running = false;
      finishing = false;
      clearInterval(ticker);
      clearTimeout(finishWatchdog);
      ticker = null;
      finishWatchdog = null;
      modeSelect.disabled = false;
      transcriptionDomain.disabled = false; entityAsrContext.disabled = false;
      setButton(false);
      setStatus(status, isError);
      if (socket && socket.readyState !== WebSocket.CLOSED) socket.close();
      socket = null;
    }

    function renderRefinedText(data) {
      const hasStreamingDisplay = (
        (data.event === 'transcript' || data.event === 'update')
        && typeof data.display_refined_text === 'string'
        && typeof data.pending_raw_text === 'string'
      );
      if (!hasStreamingDisplay) {
        if (data.clean_text !== undefined) clean.textContent = data.clean_text;
        return;
      }
      clean.replaceChildren();
      if (data.display_refined_text) {
        clean.append(document.createTextNode(data.display_refined_text));
      }
      if (data.pending_raw_text) {
        const pending = document.createElement('span');
        pending.className = 'pending-refinement';
        pending.title = '等待后台精修';
        pending.setAttribute('aria-label', '待精修文本');
        pending.textContent = data.pending_raw_text;
        clean.append(pending);
      }
    }

    function join(a, b) {
      const output = new Float32Array(a.length + b.length);
      output.set(a); output.set(b, a.length);
      return output;
    }

    function resample(input, fromRate) {
      if (fromRate === targetRate) return input;
      const ratio = targetRate / fromRate;
      const output = new Float32Array(Math.round(input.length * ratio));
      for (let index = 0; index < output.length; index += 1) {
        const sourceIndex = index / ratio;
        const left = Math.floor(sourceIndex);
        const right = Math.min(left + 1, input.length - 1);
        output[index] = input[left] + (input[right] - input[left]) * (sourceIndex - left);
      }
      return output;
    }

    function websocketUrl(asrChunkSeconds) {
      const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
      const decodeInterval = asrChunkSeconds === undefined ? '' : '&asr_chunk_seconds=' + encodeURIComponent(asrChunkSeconds);
      return scheme + '://' + location.host + '/ws/stream?language=Chinese&mode=' + encodeURIComponent(selectedMode) + '&domain=' + encodeURIComponent(transcriptionDomain.value) + '&entity_asr_context=' + (entityAsrContext.checked ? '1' : '0') + '&entity_audio_repair=' + (entityAudioRepair.checked ? '1' : '0') + decodeInterval;
    }

    function connectSocket(asrChunkSeconds) {
      return new Promise((resolve, reject) => {
        socket = new WebSocket(websocketUrl(asrChunkSeconds));
        const currentSocket = socket;
        socket.binaryType = 'arraybuffer';
        if (currentFileTiming && currentFileTiming.finishedAt == null) currentFileTiming.socket = currentSocket;
        let connected = false;
        const timeout = setTimeout(() => {
          reject(new Error('连接超时')); currentSocket.close();
        }, 10000);
        socket.onopen = () => { connected = true; clearTimeout(timeout); resolve(currentSocket); };
        socket.onerror = () => { clearTimeout(timeout); reject(new Error('无法连接识别服务')); currentSocket.close(); };
        socket.onclose = () => {
          clearTimeout(timeout);
          if (!connected) reject(new Error('识别连接未能建立'));
          if (socket !== currentSocket) return;
          socket = null;
          if (!running && !finishing) return;
          if (finishing && raw.textContent.trim()) {
            completeTranscription('已完成（连接已关闭，已保留完整转写）', false, true);
          } else {
            completeTranscription('识别连接已关闭，请重试', true);
          }
        };
        socket.onmessage = event => {
          // A previous transcription can still have buffered WebSocket
          // messages after a new session starts.  Only the active socket may
          // update the shared transcript panes and refinement history.
          if (socket !== currentSocket) return;
          const data = JSON.parse(event.data);
          // Refill the bounded send window before any transcript/history DOM
          // work. Each ACK acknowledges exactly its own frame index.
          if (data.event === 'chunk_ack') {
            currentSocket.dispatchEvent(new CustomEvent('chunk-ack', {
              detail: { chunkIndex: data.chunk_index },
            }));
            return;
          }
          if (data.event === 'ready' && currentFileTiming && currentFileTiming.socket === currentSocket) {
            currentFileTiming.traceId = data.trace_id;
            currentFileTiming.contextEnabled = data.entity_asr_context_requested;
            currentFileTiming.repairEnabled = data.entity_audio_repair_requested;
          }
          const handlerStarted = performance.now();
          try {
          updateSessionMetrics(data);
          if (data.entity_audio_repair) {
            const repair = data.entity_audio_repair;
            const reasonMessages = {
              no_normalizable_entities: '该领域暂无可复核实体',
              backend_context_unsupported: '当前识别后端不支持原音复核',
              source_text_limit: '转录过长，本次保留初稿',
              audio_cache_limit: '音频过长，本次保留初稿',
              no_candidates: '未检索到待复核词条',
              weak_candidates: '候选证据较弱，已保留初稿',
            };
            let message;
            if (repair.phase === 'capturing') {
              message = '已开启，结束时复核';
            } else if (repair.phase === 'retrieving' || repair.phase === 'verifying') {
              message = `原音复核中 · ${repair.elapsed_seconds || 0} 秒`;
            } else if (repair.applied) {
              message = `原音已修正 ${repair.change_count || 0} 处`;
            } else if (repair.phase === 'fallback') {
              message = '复核未完成，已保留初稿';
            } else {
              message = reasonMessages[repair.reason]
                || (repair.phase === 'skipped' ? '本次未复核，已保留初稿' : '未确认可替换实体');
            }
            document.querySelector('#entity-audio-status').textContent = message;
          } else if (data.event === 'ready') {
            document.querySelector('#entity-audio-status').textContent = '';
          }
          if (data.entity_asr_context_guard?.fallback_active) {
            document.querySelector('#entity-context-status').textContent = '检测到提示复述，已用原音关闭词库重新识别';
          }
          // Keep validated streaming refinements visible through stream drain
          // and final reconciliation; suppress only stale ASR hypotheses.
          if (finishing && data.event === 'transcript' && data.refiner_deferred !== true) return;
          if (data.event === 'ready') {
            document.querySelector('#entity-context-status').textContent = data.entity_asr_context_applied
              ? `已使用 ${data.entity_asr_context_term_count || 0} 项术语`
              : (data.entity_asr_context_requested
                 ? (data.entity_asr_context_term_count ? '当前后端未应用术语提示' : '该领域暂无适合辅助识别的术语') : '');
            return;
          }
          if (data.event === 'error') { completeTranscription('识别服务错误：' + data.detail, true); return; }
          if (data.event === 'status' && data.stage === 'entity_audio_repair') {
            setStatus('正在用原音核对实体 · ' + (data.elapsed_seconds || 0) + ' 秒');
            return;
          }
          if (data.event === 'status' && data.stage === 'stream_refinement_drain') {
            refinerExecuted.textContent = 'ASR 已结束 · 流式精修收尾中';
            refinerResult.textContent = '正在完成活动窗口和最终 ASR 文本的流式精修';
            refinerResult.classList.remove('warning');
            refinerType.textContent = '流式窗口精修';
            setStatus('ASR 已完成，等待流式精修队列结束 · ' + (data.elapsed_seconds || 0) + ' 秒');
            return;
          }
          if (data.event === 'status' && data.stage === 'final_refinement') {
            refinerExecuted.textContent = '最终精修中 · ' + (data.elapsed_seconds || 0) + ' 秒';
            refinerResult.textContent = '正在处理完整转录，连接保持中';
            refinerResult.classList.remove('warning');
            refinerType.textContent = selectedMode === 'streaming' ? '流式最终精修' : '离线最终精修';
            setStatus('正在完成最终精修');
            return;
          }
          if (data.event === 'status' && data.stage === 'asr_chunk') {
            setStatus(`正在处理音频块 ${data.chunk_index || ''}· 已等待 ${data.elapsed_seconds || 0} 秒`);
            return;
          }
          if (data.event === 'update' || data.event === 'final') {
            const revision = Number(data.refinement_revision);
            if (Number.isFinite(revision)) {
              if (revision < lastAppliedRefinementRevision) return;
              lastAppliedRefinementRevision = revision;
            }
          }
          if (data.raw_text !== undefined && data.event !== 'update') raw.textContent = data.raw_text;
          renderRefinedText(data);
          if (data.clean_text !== undefined && (data.event === 'update' || data.event === 'final')) latestRefinedText = data.clean_text;
          if (data.event === 'transcript') {
            const deferred = data.refiner_deferred === true;
            const gateAction = String(data.refinement_gate_action || '');
            const gateLabels = { keep: 'KEEP · 保留', defer: 'DEFER · 等待稳定', refine: 'REFINE · 调用模型' };
            if (gateAction && data.refinement_gate_state) {
              const reasons = Array.isArray(data.refinement_gate_reasons) ? data.refinement_gate_reasons.join('、') : '';
              refinerExecuted.textContent = gateLabels[gateAction] || data.refinement_gate_state;
              refinerResult.textContent = reasons ? `三态门控：${reasons}` : '三态门控已完成判断';
              refinerResult.classList.remove('warning');
              refinerType.textContent = 'tri_state · 实时假设路由';
              setStatus(gateAction === 'defer' ? '等待 ASR 假设稳定' : (gateAction === 'keep' ? '保留当前转写，跳过模型' : '已进入精修队列'));
            } else {
              refinerExecuted.textContent = deferred ? '最终 ASR 已返回' : '后台精修排队中';
              refinerResult.textContent = deferred
                ? '完整文本将先进入流式精修，再进行最终精修'
                : '正在识别并更新活动窗口';
              refinerResult.classList.remove('warning');
              refinerType.textContent = deferred
                ? '流式窗口精修收尾中'
                : '流式异步精修';
              setStatus(
                deferred
                  ? 'ASR 已完成，继续流式精修'
                  : '离线流式转录中，精修异步跟进'
              );
            }
          }
          recordRefinementStep(data);
          updateRefinerStatus(data);
          if (data.event === 'update' && !finishing) {
            setStatus(
              selectedMode === 'streaming'
                ? '离线流式转录与精修中'
                : selectedMode === 'offline'
                  ? '离线模式滑动窗口精修中'
                  : '实时识别中'
            );
          }
          if (data.event === 'final') completeTranscription('已完成');
          } finally {
            if (currentFileTiming && currentFileTiming.socket === currentSocket) {
              currentFileTiming.handlerMs += performance.now() - handlerStarted;
              currentFileTiming.messages += 1;
            }
          }
        };
      });
    }

    function flushChunks() {
      while (running && socket && socket.readyState === WebSocket.OPEN && buffered.length >= microphoneChunkSamples) {
        socket.send(buffered.slice(0, microphoneChunkSamples).buffer);
        buffered = buffered.slice(microphoneChunkSamples);
      }
    }

    async function stopAudio() {
      if (processor) { processor.disconnect(); processor.onaudioprocess = null; }
      if (source) source.disconnect();
      if (stream) stream.getTracks().forEach(track => track.stop());
      if (context) await context.close();
      processor = null; source = null; stream = null; context = null;
    }

    async function decodeAudioFile(file) {
      const fileContext = new AudioContext();
      try {
        const decoded = await fileContext.decodeAudioData(await file.arrayBuffer());
        const mono = new Float32Array(decoded.length);
        for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
          const input = decoded.getChannelData(channel);
          for (let index = 0; index < input.length; index += 1) mono[index] += input[index] / decoded.numberOfChannels;
        }
        return resample(mono, decoded.sampleRate);
      } finally {
        await fileContext.close();
      }
    }

    async function sendFileChunks(samples) {
      const fileSocket = socket;
      const totalChunks = Math.ceil(samples.length / fileChunkSamples);
      const timing = currentFileTiming;
      if (timing) { timing.samples = samples.length; timing.totalChunks = totalChunks; }
      await new Promise((resolve, reject) => {
        const pending = new Map();
        let sent = 0, acknowledgedCount = 0, settled = false, timeout;
        const cleanup = () => {
          clearTimeout(timeout);
          fileSocket.removeEventListener('chunk-ack', acknowledged);
          fileSocket.removeEventListener('close', closed);
        };
        const fail = error => {
          if (settled) return;
          settled = true; cleanup(); reject(error);
        };
        const armTimeout = () => {
          clearTimeout(timeout);
          timeout = setTimeout(() => fail(new Error('音频传输确认超时，连接已停止')), FILE_CHUNK_TIMEOUT_MS);
        };
        const pump = () => {
          if (settled) return;
          if (socket !== fileSocket || fileSocket.readyState !== WebSocket.OPEN) {
            fail(new Error('音频传输连接已关闭')); return;
          }
          try {
            while (!settled && sent < totalChunks && pending.size < FILE_UPLOAD_WINDOW) {
              const offset = sent * fileChunkSamples;
              const bytes = samples.slice(offset, Math.min(offset + fileChunkSamples, samples.length)).buffer;
              const index = ++sent;
              const sentAt = performance.now();
              if (timing && timing.firstSendAt == null) timing.firstSendAt = sentAt;
              pending.set(index, { sentAt, bytes: bytes.byteLength });
              fileSocket.send(bytes);
            }
            if (sent === totalChunks && !pending.size) {
              if (timing) timing.allAckedAt = performance.now();
              settled = true; cleanup(); resolve();
            }
          } catch (error) { fail(error); }
        };
        const acknowledged = event => {
          if (settled || socket !== fileSocket) return;
          const index = Number(event.detail?.chunkIndex);
          if (!Number.isInteger(index) || index < 1 || index > sent) {
            fail(new Error('收到无效的音频确认编号')); return;
          }
          const frame = pending.get(index);
          if (!frame) return; // duplicate ACK; never frees another frame
          pending.delete(index); acknowledgedCount += 1;
          if (timing) timing.acks.push({ index, ...frame, ackAt: performance.now() });
          setStatus(`正在传输并识别音频 · 已确认 ${acknowledgedCount}/${totalChunks} 块`);
          armTimeout(); pump();
        };
        const closed = () => fail(new Error('音频传输连接已关闭'));
        fileSocket.addEventListener('chunk-ack', acknowledged);
        fileSocket.addEventListener('close', closed);
        armTimeout(); pump();
      });
    }

    async function startFile(file) {
      const generation = ++fileRunSerial;
      const previousSocket = socket;
      socket = null;
      if (previousSocket && previousSocket.readyState !== WebSocket.CLOSED) previousSocket.close();
      running = false; clearInterval(ticker);
      selectedMode = modeSelect.value;
      const timing = { startedAt: performance.now(), acks: [], handlerMs: 0, messages: 0,
        mode: selectedMode, contextEnabled: entityAsrContext.checked, repairEnabled: entityAudioRepair.checked };
      currentFileTiming = timing; fileTimingLabel.hidden = true; downloadFileTiming.hidden = true;
      window.__asrLastFileTransfer = null;
      let fileSocket;
      try {
        finishing = false;
        clearTimeout(finishWatchdog);
        modeSelect.disabled = true; transcriptionDomain.disabled = true; entityAsrContext.disabled = true;
        setButton(false, true); setStatus(`正在读取文件：${file.name}`);
        const samples = await decodeAudioFile(file);
        timing.decodedAt = performance.now();
        if (generation !== fileRunSerial) return;
        // File uploads decode every two audio seconds; transport stays at one
        // second per block. Microphone sessions retain the server default.
        fileSocket = await connectSocket(2);
        timing.socket = fileSocket;
        if (generation !== fileRunSerial) { fileSocket.close(); return; }
        raw.textContent = ''; clean.textContent = ''; latestRefinedText = ''; lastAppliedRefinementRevision = 0; resetRefinerStatus(); resetRefinementHistory();
        running = true; startedAt = Date.now(); ticker = setInterval(updateTimer, 250); updateTimer();
        setButton(false, true);
        if (selectedMode === 'offline') {
          setStatus('正在整段识别并最终精修');
        } else {
          setStatus('离线流式转录中，精修异步跟进');
        }
        await sendFileChunks(samples);
        if (socket === fileSocket && fileSocket.readyState === WebSocket.OPEN) {
          finishing = true;
          setStatus('音频已发送，等待 ASR 最终结果；流式精修继续运行');
          timing.finishSentAt = performance.now();
          fileSocket.send(JSON.stringify({ event: 'finish' }));
          // A broken proxy or dropped WebSocket must not leave the timer running forever.
          finishWatchdog = setTimeout(() => {
            if (finishing) completeTranscription(
              raw.textContent.trim()
                ? '已停止计时（精修响应超时，已保留完整转写）'
                : '精修服务响应超时，请重试',
              !raw.textContent.trim(),
              Boolean(raw.textContent.trim()),
            );
          }, FINISH_WATCHDOG_MS);
        }
      } catch (error) {
        if (generation !== fileRunSerial || timing.finishedAt != null) return;
        saveFileTiming('error');
        running = false; finishing = false; clearInterval(ticker); clearTimeout(finishWatchdog); modeSelect.disabled = false; transcriptionDomain.disabled = false; entityAsrContext.disabled = false; setButton(false);
        setStatus('文件处理失败：' + error.message, true);
        if (socket === fileSocket) { if (socket) socket.close(); socket = null; }
      }
    }

    async function startMicrophone() {
      fileRunSerial += 1; currentFileTiming = null;
      fileTimingLabel.hidden = true; downloadFileTiming.hidden = true;
      try {
        finishing = false;
        clearTimeout(finishWatchdog);
        selectedMode = modeSelect.value;
        modeSelect.disabled = true; transcriptionDomain.disabled = true; entityAsrContext.disabled = true;
        setButton(false, true); setStatus('正在连接识别服务');
        await connectSocket();
        stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
        context = new AudioContext();
        source = context.createMediaStreamSource(stream);
        processor = context.createScriptProcessor(4096, 1, 1);
        processor.onaudioprocess = event => {
          if (!running) return;
          buffered = join(buffered, resample(event.inputBuffer.getChannelData(0), context.sampleRate));
          flushChunks();
        };
        source.connect(processor); processor.connect(context.destination);
        buffered = new Float32Array(0); raw.textContent = ''; clean.textContent = ''; latestRefinedText = ''; lastAppliedRefinementRevision = 0; resetRefinerStatus(); resetRefinementHistory();
        running = true; startedAt = Date.now(); ticker = setInterval(updateTimer, 250); updateTimer();
        setButton(true); setStatus('实时识别中');
      } catch (error) {
        running = false; finishing = false; clearTimeout(finishWatchdog); modeSelect.disabled = false; transcriptionDomain.disabled = false; entityAsrContext.disabled = false; setButton(false); setStatus('启动失败：' + error.message, true);
        if (socket) socket.close(); socket = null; await stopAudio();
      }
    }

    async function stop() {
      running = false; finishing = true; clearInterval(ticker); setButton(true, true);
      setStatus('音频输入已结束，等待 ASR 最终结果；流式精修继续运行');
      await stopAudio();
      if (socket && socket.readyState === WebSocket.OPEN) {
        if (buffered.length) socket.send(buffered.buffer);
        socket.send(JSON.stringify({ event: 'finish' }));
        clearTimeout(finishWatchdog);
        finishWatchdog = setTimeout(() => {
          if (finishing) completeTranscription(
            raw.textContent.trim()
              ? '已停止计时（精修响应超时，已保留完整转写）'
              : '精修服务响应超时，请重试',
            !raw.textContent.trim(),
            Boolean(raw.textContent.trim()),
          );
        }, FINISH_WATCHDOG_MS);
      } else { completeTranscription('连接已关闭', true); }
      buffered = new Float32Array(0);
    }

    function start() {
      selectedMode = modeSelect.value;
      if (selectedMode === 'online') return startMicrophone();
      audioFileInput.value = '';
      audioFileInput.click();
    }

    manageEntitiesButton.addEventListener('click', async () => {
      entityPanel.hidden = false;
      await loadEntities();
      entityPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    closeEntitiesButton.addEventListener('click', () => { entityPanel.hidden = true; });
    refreshEntitiesButton.addEventListener('click', loadEntities);
    addEntityButton.addEventListener('click', () => openEntityDialog());
    document.querySelector('#cancel-entity').addEventListener('click', () => entityDialog.close());
    entitySearch.addEventListener('input', loadEntities);
    entityTypeFilter.addEventListener('change', loadEntities);
    entityStatusFilter.addEventListener('change', loadEntities);
    entityRows.addEventListener('click', async event => {
      const button = event.target.closest('button[data-action]');
      if (!button) return;
      const entity = entities.find(item => item.entity_id === Number(button.dataset.id));
      if (!entity) return;
      try {
        if (button.dataset.action === 'edit') return openEntityDialog(entity);
        if (button.dataset.action === 'toggle') {
          await entityApi('/api/entities/' + entity.entity_id + '/enabled', {
            method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: !entity.enabled }),
          });
          setEntityNotice(entity.enabled ? '实体已停用；下一次转录不再匹配。' : '实体已启用；下一次转录将匹配。');
        }
        if (button.dataset.action === 'delete') {
          if (!confirm('确定永久删除“' + entity.canonical_text + '”及其全部别名吗？')) return;
          await entityApi('/api/entities/' + entity.entity_id, { method: 'DELETE' });
          setEntityNotice('实体已永久删除。');
        }
        await loadEntities();
      } catch (error) { setEntityNotice(error.message, true); }
    });
    entityForm.addEventListener('submit', async event => {
      event.preventDefault();
      const aliases = entityAliases.value.split('\n').map(value => value.trim()).filter(Boolean);
      const payload = {
        canonical_text: entityCanonical.value.trim(), entity_type: entityType.value.trim(), domain: entityDomain.value.trim(),
        normalization_policy: entityPolicy.value, priority: Number(entityPriority.value), aliases,
      };
      if (!payload.canonical_text || !payload.entity_type || !payload.domain || !Number.isInteger(payload.priority)) {
        entityFormError.textContent = '请填写标准名称、类型、领域，并输入整数优先级。'; return;
      }
      const entityId = entityForm.dataset.entityId;
      const submitButton = entityForm.querySelector('button[type="submit"]');
      submitButton.disabled = true; entityFormError.textContent = '';
      try {
        const endpoint = entityId ? '/api/entities/' + entityId : '/api/entities';
        const saved = await entityApi(endpoint, {
          method: entityId ? 'PUT' : 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
        });
        if (saved.enabled !== entityEnabled.checked) {
          await entityApi('/api/entities/' + saved.entity_id + '/enabled', {
            method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: entityEnabled.checked }),
          });
        }
        entityDialog.close();
        setEntityNotice('已保存。变更将在下一次开始转录时生效。');
        await loadEntities();
      } catch (error) { entityFormError.textContent = error.message; }
      finally { submitButton.disabled = false; }
    });

    modeSelect.addEventListener('change', () => {
      if (!running) setButton(false);
    });
    audioFileInput.addEventListener('change', () => {
      const file = audioFileInput.files && audioFileInput.files[0];
      if (file) startFile(file);
    });
    recordButton.addEventListener('click', () => running ? stop() : start());
    lucide.createIcons();
  