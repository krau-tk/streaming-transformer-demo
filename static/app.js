const SAMPLE_RATE = 16000;
const CHUNK_DURATION_MS = 320;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_DURATION_MS / 1000);  // 5120

let ws = null;
let audioCtx = null;
let mediaStream = null;
let processor = null;
let source = null;
let monitorGain = null;
let isRecording = false;
let sampleBuffer = [];
let shouldReconnect = true;
let inputSampleRate = SAMPLE_RATE;
let resampleRemainder = new Float32Array(0);
let resampleOffset = 0;

const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const transcript = document.getElementById('transcript');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const modeInputs = Array.from(document.querySelectorAll('input[name="session-mode"]'));

function setStatus(state, text) {
    statusDot.className = 'status-dot ' + state;
    statusText.textContent = text;
}

function getSelectedMode() {
    const selected = document.querySelector('input[name="session-mode"]:checked');
    return selected ? selected.value : 'pseudo';
}

function setModeInputsDisabled(disabled) {
    modeInputs.forEach(input => {
        input.disabled = disabled;
    });
}

function getAsrWsUrl() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const mode = encodeURIComponent(getSelectedMode());
    return `${protocol}//${location.host}/ws/asr?mode=${mode}`;
}

function handleAsrMessage(event) {
    const data = JSON.parse(event.data);
    if (data.error) {
        transcript.textContent = '[错误] ' + data.error;
        setStatus('', '识别错误');
        return;
    }
    transcript.textContent = data.text;
    if (data.is_final) {
        setStatus('connected', '识别完成');
    }
}

function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        return;
    }

    ws = new WebSocket(getAsrWsUrl());

    ws.onopen = () => {
        setStatus('connected', '已连接，准备就绪');
        btnStart.disabled = false;
    };

    ws.onmessage = handleAsrMessage;

    ws.onclose = () => {
        ws = null;
        if (isRecording) {
            // Connection lost during recording
            cleanupAudio();
            isRecording = false;
            setModeInputsDisabled(false);
            btnStart.style.display = 'inline-block';
            btnStop.style.display = 'none';
        }
        setStatus('', '连接断开');
        btnStart.disabled = true;
        if (shouldReconnect) {
            setTimeout(connectWebSocket, 2000);
        }
    };

    ws.onerror = () => {
        setStatus('', '连接错误');
    };
}

async function startRecording() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: SAMPLE_RATE,
                echoCancellation: true,
                noiseSuppression: true,
            }
        });
    } catch (e) {
        alert('无法访问麦克风: ' + e.message);
        return;
    }

    // Close old WS and create fresh session
    shouldReconnect = false;
    if (ws) {
        ws.onclose = null;  // prevent reconnect loop
        ws.close();
        ws = null;
    }

    // Wait a tick then connect fresh
    await new Promise(r => setTimeout(r, 100));
    shouldReconnect = true;

    ws = new WebSocket(getAsrWsUrl());

    ws.onopen = () => {
        // Start audio processing now that WS is ready
        setupAudio();
        isRecording = true;
        sampleBuffer = [];
        transcript.textContent = '';
        setModeInputsDisabled(true);
        setStatus('recording', '正在录音...');
        btnStart.style.display = 'none';
        btnStop.style.display = 'inline-block';
    };

    ws.onmessage = handleAsrMessage;

    ws.onclose = () => {
        ws = null;
        if (isRecording) {
            cleanupAudio();
            isRecording = false;
            setModeInputsDisabled(false);
            btnStart.style.display = 'inline-block';
            btnStop.style.display = 'none';
        }
        setStatus('', '连接断开');
        btnStart.disabled = true;
        if (shouldReconnect) {
            setTimeout(connectWebSocket, 2000);
        }
    };

    ws.onerror = () => setStatus('', '连接错误');
}

function setupAudio() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    try {
        audioCtx = new AudioContextClass({ sampleRate: SAMPLE_RATE });
    } catch (e) {
        audioCtx = new AudioContextClass();
    }
    inputSampleRate = audioCtx.sampleRate;
    resampleRemainder = new Float32Array(0);
    resampleOffset = 0;
    source = audioCtx.createMediaStreamSource(mediaStream);
    processor = audioCtx.createScriptProcessor(4096, 1, 1);
    monitorGain = audioCtx.createGain();
    monitorGain.gain.value = 0;

    processor.onaudioprocess = (e) => {
        if (!isRecording) return;
        const float32 = e.inputBuffer.getChannelData(0);
        const pcm16k = resampleTo16k(float32, inputSampleRate);
        for (let i = 0; i < pcm16k.length; i++) {
            sampleBuffer.push(pcm16k[i]);
        }
        while (sampleBuffer.length >= CHUNK_SAMPLES) {
            const chunk = sampleBuffer.splice(0, CHUNK_SAMPLES);
            sendAudioChunk(chunk);
        }
    };

    source.connect(processor);
    processor.connect(monitorGain);
    monitorGain.connect(audioCtx.destination);
}

function resampleTo16k(input, sourceRate) {
    if (sourceRate === SAMPLE_RATE) {
        return input;
    }

    const ratio = sourceRate / SAMPLE_RATE;
    const combined = new Float32Array(resampleRemainder.length + input.length);
    combined.set(resampleRemainder);
    combined.set(input, resampleRemainder.length);

    const outputLength = Math.max(0, Math.floor((combined.length - 1 - resampleOffset) / ratio) + 1);
    const output = new Float32Array(outputLength);

    let sourceIndex = resampleOffset;
    for (let i = 0; i < outputLength; i++) {
        const left = Math.floor(sourceIndex);
        const right = Math.min(left + 1, combined.length - 1);
        const weight = sourceIndex - left;
        output[i] = combined[left] * (1 - weight) + combined[right] * weight;
        sourceIndex += ratio;
    }

    const consumed = Math.floor(sourceIndex);
    resampleRemainder = combined.slice(consumed);
    resampleOffset = sourceIndex - consumed;

    return output;
}

function sendAudioChunk(float32Array) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const int16 = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
        const s = Math.max(-1, Math.min(1, float32Array[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    ws.send(int16.buffer);
}

function cleanupAudio() {
    if (processor) {
        processor.disconnect();
        processor = null;
    }
    if (monitorGain) {
        monitorGain.disconnect();
        monitorGain = null;
    }
    if (source) {
        source.disconnect();
        source = null;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
    }
    if (audioCtx) {
        audioCtx.close();
        audioCtx = null;
    }
    resampleRemainder = new Float32Array(0);
    resampleOffset = 0;
}

function stopRecording() {
    if (!isRecording) return;
    isRecording = false;

    // Send remaining samples
    if (sampleBuffer.length > 0) {
        sendAudioChunk(sampleBuffer);
        sampleBuffer = [];
    }

    // Signal finalize
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send('finalize');
    }

    cleanupAudio();
    setModeInputsDisabled(false);
    setStatus('connected', '处理中...');
    btnStart.style.display = 'inline-block';
    btnStop.style.display = 'none';
}

btnStart.addEventListener('click', startRecording);
btnStop.addEventListener('click', stopRecording);
modeInputs.forEach(input => {
    input.addEventListener('change', () => {
        if (isRecording) return;
        transcript.textContent = '';
        shouldReconnect = false;
        if (ws) {
            ws.onclose = null;
            ws.close();
            ws = null;
        }
        shouldReconnect = true;
        btnStart.disabled = true;
        setStatus('', '正在连接...');
        connectWebSocket();
    });
});

// === File upload (offline recognition) ===
const fileInput = document.getElementById('file-input');
const btnUpload = document.getElementById('btn-upload');
const fileInfo = document.getElementById('file-info');

fileInput.addEventListener('change', () => {
    btnUpload.disabled = !fileInput.files.length;
    if (fileInput.files.length) {
        const f = fileInput.files[0];
        fileInfo.textContent = `${f.name} (${(f.size / 1024).toFixed(1)} KB)`;
    } else {
        fileInfo.textContent = '';
    }
});

btnUpload.addEventListener('click', async () => {
    if (!fileInput.files.length) return;

    const file = fileInput.files[0];
    btnUpload.disabled = true;
    btnUpload.textContent = '识别中...';
    fileInfo.innerHTML = '<span class="spinner"></span> 正在处理...';
    transcript.textContent = '';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const mode = encodeURIComponent(getSelectedMode());
        const resp = await fetch(`/api/recognize?mode=${mode}`, { method: 'POST', body: formData });
        const data = await resp.json();

        if (data.error) {
            transcript.textContent = '[错误] ' + data.error;
            fileInfo.textContent = '识别失败';
        } else {
            transcript.textContent = data.text || '(无识别结果)';
            fileInfo.textContent = `音频时长: ${data.duration}s`;
            if (data.partial_results && data.partial_results.length) {
                fileInfo.textContent += ` | 逐步输出: ${data.partial_results.length} 次`;
            }
        }
    } catch (e) {
        transcript.textContent = '[网络错误] ' + e.message;
        fileInfo.textContent = '请求失败';
    }

    btnUpload.disabled = false;
    btnUpload.textContent = '上传识别';
});

// Initial connection
btnStart.disabled = true;
setStatus('', '正在连接...');
connectWebSocket();
