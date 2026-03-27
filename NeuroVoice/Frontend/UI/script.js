let mediaRecorder;
let recordedChunks = [];
let recordedBlob = null;

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

async function startRecording() {
  const status = document.getElementById("recordingStatus");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    recordedChunks = [];

    mediaRecorder.ondataavailable = e => recordedChunks.push(e.data);

    mediaRecorder.onstop = () => {
      recordedBlob = new Blob(recordedChunks, { type: "audio/wav" });
      status.innerText = "Recording ready ✔️";
    };

    mediaRecorder.start();
    status.innerText = "Recording...";
  } catch {
    alert("Microphone access denied");
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
}

function showLoading(show = true) {
  document.getElementById("loading").style.display = show ? "block" : "none";
}

function createAudioPlayer(audioBlob, filename, alpha = null) {
  const audioUrl = URL.createObjectURL(audioBlob);
  const container = document.getElementById("audioContainer");
  
  const audioDiv = document.createElement("div");
  audioDiv.className = "audio-item";
  
  const title = document.createElement("h4");
  title.textContent = alpha !== null ? `Alpha: ${alpha}` : filename;
  audioDiv.appendChild(title);
  
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.src = audioUrl;
  audioDiv.appendChild(audio);
  
  const download = document.createElement("a");
  download.href = audioUrl;
  download.download = alpha !== null ? `neurovoice_alpha_${alpha}.wav` : filename;
  download.textContent = "Download";
  download.className = "download-btn";
  audioDiv.appendChild(download);
  
  container.appendChild(audioDiv);
}

async function generateSingle() {
  const text = document.getElementById("text").value;
  const language = document.getElementById("language").value;
  const alpha = parseFloat(document.getElementById("alpha").value);
  const uploadedAudio = document.getElementById("audio").files[0];

  if (text.trim() === "") {
    alert("Please enter text");
    return;
  }

  if (!uploadedAudio && !recordedBlob) {
    alert("Please upload or record a reference voice");
    return;
  }

  showLoading(true);
  document.getElementById("audioContainer").innerHTML = "";

  try {
    let audioBlob = uploadedAudio || recordedBlob;
    
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
    createAudioPlayer(responseAudioBlob, `neurovoice_${language}.wav`);

  } catch (err) {
    alert("Error generating speech: " + err.message);
    console.error(err);
  } finally {
    showLoading(false);
  }
}

async function generateMultiple() {
  const text = document.getElementById("text").value;
  const language = document.getElementById("language").value;
  const uploadedAudio = document.getElementById("audio").files[0];

  if (text.trim() === "") {
    alert("Please enter text");
    return;
  }

  if (!uploadedAudio && !recordedBlob) {
    alert("Please upload or record a reference voice");
    return;
  }

  showLoading(true);
  document.getElementById("audioContainer").innerHTML = "";

  const alphas = [0.1, 0.3, 0.5, 0.7, 0.9];
  
  try {
    let audioBlob = uploadedAudio || recordedBlob;
    
    // Convert MP3 to WAV if needed
    if (audioBlob.type.includes("mp3")) {
      audioBlob = await convertMp3ToWav(audioBlob);
    }

    for (const alpha of alphas) {
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
      createAudioPlayer(responseAudioBlob, `neurovoice_${language}.wav`, alpha);
    }

  } catch (err) {
    alert("Error generating speech: " + err.message);
    console.error(err);
  } finally {
    showLoading(false);
  }
}

// File selection handler
function handleFileSelect(input) {
  const file = input.files[0];
  const fileInfo = document.getElementById("fileInfo");
  
  if (file) {
    fileInfo.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
  } else {
    fileInfo.textContent = "";
  }
}

