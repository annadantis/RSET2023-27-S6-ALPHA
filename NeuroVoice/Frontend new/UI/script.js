// ===========================
// NeuroVoice script.js
// Frontend Demo Logic
// ===========================

// GLOBAL VARIABLES
let currentPage = "home";
let mediaRecorder;
let audioChunks = [];
let recordedAudio = null;
let isRecording = false;
let generatedAudio = null;


// ===========================
// PAGE NAVIGATION
// ===========================
function navigateTo(page) {

    document.querySelectorAll(".page").forEach(p => {
        p.classList.remove("active");
    });

    document.getElementById(page + "-page").classList.add("active");

    document.querySelectorAll(".nav-link").forEach(link => {
        link.classList.remove("active");
    });

    document.querySelectorAll(`.nav-link[data-page="${page}"]`).forEach(link => {
        link.classList.add("active");
    });

    currentPage = page;
}


// Navbar click support
document.querySelectorAll(".nav-link").forEach(link => {

    link.addEventListener("click", function (e) {

        e.preventDefault();

        const page = this.getAttribute("data-page");

        navigateTo(page);

    });

});


// ===========================
// TEXT CHARACTER COUNT
// ===========================

const textInput = document.getElementById("textInput");

if (textInput) {

    textInput.addEventListener("input", function () {

        document.getElementById("charCount").innerText = this.value.length;

    });

}


// ===========================
// RECORD VOICE
// ===========================

async function toggleRecording() {

    if (!isRecording) {

        try {

            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            mediaRecorder = new MediaRecorder(stream);

            audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {

                recordedAudio = new Blob(audioChunks, { type: "audio/webm" });

                showNotification("Recording saved", "success");

            };

            mediaRecorder.start();

            isRecording = true;

            document.getElementById("recordBtn").classList.add("recording");

            document.getElementById("recordingStatus").innerText = "Recording...";

        }

        catch {

            showNotification("Microphone access denied", "error");

        }

    }

    else {

        stopRecording();

    }

}


function stopRecording() {

    if (mediaRecorder && isRecording) {

        mediaRecorder.stop();

        isRecording = false;

        document.getElementById("recordBtn").classList.remove("recording");

        document.getElementById("recordingStatus").innerText = "Recording stopped";

    }

}


function playRecording() {

    if (!recordedAudio) {

        showNotification("No recording found", "warning");

        return;

    }

    const audio = new Audio(URL.createObjectURL(recordedAudio));

    audio.play();

}


// ===========================
// FILE UPLOAD
// ===========================

const audioFileInput = document.getElementById("audioFile");

if (audioFileInput) {

    audioFileInput.addEventListener("change", function () {

        if (this.files.length > 0) {

            recordedAudio = this.files[0];

            showNotification("Audio file loaded", "success");

        }

    });

}


// ===========================
// CONTINUE TO GENERATE
// ===========================

function proceedToGenerate() {

    if (!recordedAudio) {

        showNotification("Please upload or record voice first", "warning");

        return;

    }

    navigateTo("generate");

}

// ===========================
// GENERATE VOICE (REAL EMOTION AI)
// ===========================

// MP3 to WAV conversion using Web Audio API
async function convertMp3ToWav(mp3Blob) {
    return new Promise((resolve) => {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const reader = new FileReader();
        
        reader.onload = async () => {
            const arrayBuffer = reader.result;
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // Convert to WAV
            const length = audioBuffer.length;
            const sampleRate = audioBuffer.sampleRate;
            const numberOfChannels = audioBuffer.numberOfChannels;
            const buffer = new ArrayBuffer(44 + length * 2);
            const view = new DataView(buffer);
            
            // WAV header
            const writeString = (offset, string) => {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            };
            
            writeString(0, 'RIFF');
            view.setUint32(4, 36 + length * 2, true);
            writeString(8, 'WAVE');
            writeString(12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, numberOfChannels, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * numberOfChannels * 2, true);
            view.setUint16(32, numberOfChannels * 2, true);
            view.setUint16(34, 16, true);
            writeString(36, 'data');
            view.setUint32(40, length * 2, true);
            
            // Convert float samples to 16-bit PCM
            const volume = 0.8;
            let offset = 44;
            for (let i = 0; i < length; i++) {
                let sample = audioBuffer.getChannelData(0)[i] * volume;
                sample = Math.max(-1, Math.min(1, sample));
                view.setInt16(offset, sample * 0x7FFF, true);
                offset += 2;
            }
            
            resolve(new Blob([buffer], { type: 'audio/wav' }));
        };
        
        reader.readAsArrayBuffer(mp3Blob);
    });
}

async function generateVoice() {
    const text = document.getElementById("textInput").value;
    const language = document.getElementById("languageSelect").value;
    const alpha = parseFloat(document.getElementById("emotionSlider").value) / 100; // Convert 0-100 to 0-1
    const uploadedAudio = document.getElementById("audioFile").files[0];

    if (text.trim() === "") {
        showNotification("Enter text first", "warning");
        return;
    }

    if (!uploadedAudio && !recordedAudio) {
        showNotification("Please upload or record audio", "warning");
        return;
    }

    showLoading(true);
    const start = Date.now();

    try {
        let audioBlob = uploadedAudio || recordedAudio;
        
        // Convert MP3 to WAV if needed
        if (audioBlob.type.includes("mp3")) {
            audioBlob = await convertMp3ToWav(audioBlob);
        }

        const formData = new FormData();
        formData.append("text", text);
        formData.append("language", language);
        formData.append("alpha", alpha.toString());
        formData.append("audio", audioBlob, "reference.wav");

        const response = await fetch("http://localhost:8000/synthesize", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const responseAudioBlob = await response.blob();
        generatedAudio = responseAudioBlob;
        
        // Extract real metrics from headers
        const voiceSimilarity = response.headers.get("X-Voice-Similarity");
        const valence = response.headers.get("X-Valence");
        const arousal = response.headers.get("X-Arousal");
        const dominance = response.headers.get("X-Dominance");
        const emotionLabel = response.headers.get("X-Emotion-Label");
        const confidence = response.headers.get("X-Emotion-Confidence");
        
        console.log(`Metrics - Voice: ${voiceSimilarity}, Emotion: ${emotionLabel} (${confidence}), VAD: ${valence}, ${arousal}, ${dominance}`);
        
        const time = ((Date.now() - start) / 1000).toFixed(1);
        document.getElementById("genTime").innerText = time + "s";
        document.getElementById("audioLang").innerText = 
            document.getElementById("languageSelect").selectedOptions[0].text;
        
        // Update metrics with real values
        if (voiceSimilarity) {
            document.querySelector(".metric:nth-child(1) .metric-value").innerText = 
                (parseFloat(voiceSimilarity) * 100).toFixed(1) + "%";
        }
        
        if (valence && arousal) {
            document.querySelector(".metric:nth-child(2) .metric-value").innerText = 
                `V:${parseFloat(valence).toFixed(2)} A:${parseFloat(arousal).toFixed(2)}`;
        }
        
        // Update emotion visualization
        if (arousal) {
            const arousalValue = parseFloat(arousal);
            document.querySelector(".bg-animation").style.filter = 
                `hue-rotate(${arousalValue * 180}deg)`;
        }
        
        // Create emotion bars with real values
        createEmotionBars(
            valence ? parseFloat(valence) : null,
            arousal ? parseFloat(arousal) : null,
            dominance ? parseFloat(dominance) : null
        );
        
        // Update emotion label display
        if (emotionLabel && confidence) {
            // Add emotion label to metrics section
            let emotionDisplay = document.querySelector('.emotion-display');
            if (!emotionDisplay) {
                emotionDisplay = document.createElement('div');
                emotionDisplay.className = 'emotion-display';
                emotionDisplay.innerHTML = `
                    <h4>Detected Emotion</h4>
                    <div class="emotion-result">
                        <span class="emotion-name">${emotionLabel}</span>
                        <span class="emotion-confidence">${(parseFloat(confidence) * 100).toFixed(1)}%</span>
                    </div>
                `;
                const metricsSection = document.querySelector('.quality-metrics');
                metricsSection.appendChild(emotionDisplay);
            } else {
                emotionDisplay.querySelector('.emotion-name').innerText = emotionLabel;
                emotionDisplay.querySelector('.emotion-confidence').innerText = (parseFloat(confidence) * 100).toFixed(1) + '%';
            }
        }
        
        drawWaveform();
        navigateTo("results");
        showNotification("Emotion-aware voice generated successfully", "success");

    } catch (err) {
        showNotification("Error: " + err.message, "error");
        console.error(err);
    } finally {
        showLoading(false);
    }
}

// ===========================
// REAL AUDIO PLAYER SETUP
// ===========================

function setupRealAudioPlayer() {
    // Add HTML5 audio element if not exists
    if (!document.getElementById('resultAudio')) {
        const audioContainer = document.querySelector('.audio-result');
        const audioEl = document.createElement('audio');
        audioEl.id = 'resultAudio';
        audioEl.controls = true;
        audioEl.style.width = '100%';
        audioEl.style.marginTop = '20px';
        audioContainer.appendChild(audioEl);
    }
    
    // Set audio source
    const audioEl = document.getElementById('resultAudio');
    audioEl.src = URL.createObjectURL(generatedAudio);
    
    // Update duration display
    audioEl.addEventListener('loadedmetadata', () => {
        const duration = formatTime(audioEl.duration);
        document.getElementById('audioDuration').innerText = duration;
    });
    
    // Update play button
    audioEl.addEventListener('play', () => {
        const playBtn = document.querySelector('.play-btn-large');
        if (playBtn) playBtn.innerHTML = '<i class="fas fa-pause"></i>';
    });
    
    audioEl.addEventListener('pause', () => {
        const playBtn = document.querySelector('.play-btn-large');
        if (playBtn) playBtn.innerHTML = '<i class="fas fa-play"></i>';
    });
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// ===========================
// PLAY RESULT (Updated)
// ===========================

function playResult() {
    const audioEl = document.getElementById('resultAudio');
    if (audioEl) {
        if (audioEl.paused) {
            audioEl.play();
        } else {
            audioEl.pause();
        }
    } else if (generatedAudio) {
        // Fallback to old method
        const audio = new Audio(URL.createObjectURL(generatedAudio));
        audio.play();
    }
}

// ===========================
// REFERENCE AUDIO PLAYER (Real-time)
// ===========================

function setupReferencePlayer() {
    if (!recordedAudio) return;
    
    // Create real-time progress for reference audio
    const refAudio = new Audio(URL.createObjectURL(recordedAudio));
    
    refAudio.addEventListener('loadedmetadata', () => {
        const duration = formatTime(refAudio.duration);
        const durationEl = document.getElementById('refDuration');
        if (durationEl) durationEl.innerText = duration;
    });
    
    refAudio.addEventListener('timeupdate', () => {
        const percent = (refAudio.currentTime / refAudio.duration) * 100;
        const timeline = document.getElementById('refTimeline');
        if (timeline) timeline.style.width = percent + '%';
    });
    
    // Store reference for play function
    window.referenceAudio = refAudio;
}

function playReference() {
    if (window.referenceAudio) {
        if (window.referenceAudio.paused) {
            window.referenceAudio.play();
        } else {
            window.referenceAudio.pause();
        }
    } else if (recordedAudio) {
        setupReferencePlayer();
        window.referenceAudio.play();
    }
}


// ===========================
// DOWNLOAD AUDIO
// ===========================

function downloadAudio() {

    if (!generatedAudio) return;

    const link = document.createElement("a");

    link.href = URL.createObjectURL(generatedAudio);

    link.download = "neurovoice.wav";

    link.click();

}


// ===========================
// GENERATE ANOTHER
// ===========================

function generateAnother() {

    navigateTo("generate");

}


// ===========================
// DEMO BUTTON
// ===========================

function showDemo() {

    showNotification("Demo coming soon!", "info");

}


// ===========================
// SHARE
// ===========================

function shareResult() {

    showNotification("Share feature coming soon", "info");

}


// ===========================
// LOADING OVERLAY
// ===========================

function showLoading(show) {

    const overlay = document.getElementById("loadingOverlay");

    overlay.style.display = show ? "flex" : "none";

}


// ===========================
// REAL WAVEFORM (WaveSurfer.js)
// ===========================

let wavesurfer = null;

function drawWaveform() {
    // Initialize WaveSurfer if not already done
    if (!wavesurfer && generatedAudio) {
        wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: '#4facfe',
            progressColor: '#00f2fe',
            cursorColor: '#ffffff',
            barWidth: 2,
            barRadius: 3,
            cursorWidth: 1,
            height: 100,
            barGap: 3
        });
        
        // Load the generated audio
        wavesurfer.load(URL.createObjectURL(generatedAudio));
        
        // Play/pause button integration
        wavesurfer.on('play', () => {
            const playBtn = document.querySelector('.play-btn-large');
            if (playBtn) playBtn.innerHTML = '<i class="fas fa-pause"></i>';
        });
        
        wavesurfer.on('pause', () => {
            const playBtn = document.querySelector('.play-btn-large');
            if (playBtn) playBtn.innerHTML = '<i class="fas fa-play"></i>';
        });
    } else if (wavesurfer && generatedAudio) {
        // Load new audio if wavesurfer exists
        wavesurfer.load(URL.createObjectURL(generatedAudio));
    }
}

// ===========================
// DYNAMIC EMOTION BARS
// ===========================

function createEmotionBars(valence, arousal, dominance) {
    // Add emotion bars to metrics section if not exists
    if (!document.querySelector('.emotion-bars')) {
        const metricsSection = document.querySelector('.quality-metrics');
        const emotionBars = document.createElement('div');
        emotionBars.className = 'emotion-bars';
        emotionBars.innerHTML = `
            <h4>Emotion Profile</h4>
            <div class="emotion-bar-container">
                <div class="emotion-bar-item">
                    <span>Valence</span>
                    <div class="emotion-bar-track">
                        <div id="valenceBar" class="emotion-bar-fill"></div>
                    </div>
                    <span id="valenceValue">0.00</span>
                </div>
                <div class="emotion-bar-item">
                    <span>Arousal</span>
                    <div class="emotion-bar-track">
                        <div id="arousalBar" class="emotion-bar-fill"></div>
                    </div>
                    <span id="arousalValue">0.00</span>
                </div>
                <div class="emotion-bar-item">
                    <span>Dominance</span>
                    <div class="emotion-bar-track">
                        <div id="dominanceBar" class="emotion-bar-fill"></div>
                    </div>
                    <span id="dominanceValue">0.00</span>
                </div>
            </div>
        `;
        metricsSection.appendChild(emotionBars);
    }
    
    // Update emotion bars with real values
    if (valence !== null && arousal !== null && dominance !== null) {
        document.getElementById('valenceBar').style.width = (valence * 100) + '%';
        document.getElementById('valenceValue').innerText = valence.toFixed(2);
        
        document.getElementById('arousalBar').style.width = (arousal * 100) + '%';
        document.getElementById('arousalValue').innerText = arousal.toFixed(2);
        
        document.getElementById('dominanceBar').style.width = (dominance * 100) + '%';
        document.getElementById('dominanceValue').innerText = dominance.toFixed(2);
    }
}

// ===========================
// NOTIFICATIONS
// ===========================

function showNotification(message, type = "info") {

    const div = document.createElement("div");

    div.className = `notification notification-${type}`;

    div.innerHTML = `
        ${message}
        <button onclick="this.parentElement.remove()">✖</button>
    `;

    document.body.appendChild(div);

    setTimeout(() => {

        div.remove();

    }, 4000);

}
