# 🎭 EAMVCS - Emotion-Aware Multilingual Voice Cloning System

An advanced AI system that analyzes voice emotions and generates synthesized speech with emotional characteristics across multiple Indian languages .

## 🚀 **Quick Start (5 Minutes)**

### **1. Clone & Setup**
```bash
git clone https://github.com/TgeB4tMan/EAMVCS.git
cd EAMVCS
```

### **2. Create Virtual Environment**
```bash
# Create environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Start System**
```bash
# PowerShell (Windows)
.\start_neurovoice.ps1

# OR Manual Start:
# Terminal 1: Backend
cd Backend
python app.py

# Terminal 2: Frontend
cd Frontend/UI
python -m http.server 3000
```

### **5. Access Application**
- **🎨 Frontend**: http://localhost:3000
- **🔧 Backend API**: http://localhost:8000
- **📚 API Docs**: http://localhost:8000/docs

---

## 🎯 **Features**

### **Core Capabilities**
- 🎤 **Voice Recording/Upload**: Record or upload reference audio
- 🧠 **Emotion Detection**: Real-time 4-class emotion classification
- 🗣️ **Voice Synthesis**: Generate speech with emotion conditioning
- 🌍 **Multilingual Support**: 9 Indian languages + English
- 📊 **Quality Metrics**: Voice similarity, confidence scores, VAD profiles

### **Emotion Classes**
- 😐 **Neutral**: Baseline emotion state
- 😊 **Happy**: Positive valence, high arousal
- 😢 **Sad**: Negative valence, low arousal
- 😠 **Angry**: Negative valence, high arousal

### **Supported Languages**
- English, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi

---

## 🛠️ **System Requirements**

### **Minimum Requirements**
- **Python**: 3.8 or higher
- **RAM**: 8GB+ (for model loading)
- **Storage**: 5GB+ free space
- **Internet**: Required for first-time model downloads

### **Recommended**
- **GPU**: CUDA-compatible (NVIDIA recommended)
- **RAM**: 16GB+ (better performance)
- **Storage**: 10GB+ (for model cache)

---

## 📋 **Detailed Setup Guide**

### **Step 1: Environment Setup**
```bash
# Verify Python version
python --version  # Should be 3.8+

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Verify activation (should show (venv) prefix)
which python
```

### **Step 2: Install Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt

# Verify GPU support (optional but recommended)
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

### **Step 3: First-Time Model Downloads**
Models download automatically on first run:
```bash
# Start backend (models will download automatically)
cd Backend
python app.py

# Expected output:
# Loading NeuroVoice AI Models...
# Download Vocos from huggingface... (1.2GB)
# Loading Whisper model... (300MB)
# EmotionTTS initialized on cuda
# ✅ All models loaded successfully!
```

### **Step 4: Start Services**

#### **Option A: PowerShell Script (Recommended)**
```powershell
.\start_neurovoice.ps1
```

#### **Option B: Manual Start**
```bash
# Terminal 1: Backend Server
cd Backend
python app.py
# Wait for "✅ All models loaded successfully!"

# Terminal 2: Frontend Server  
cd Frontend/UI
python -m http.server 3000
```

---

## 🎮 **How to Use**

### **Basic Voice Cloning**
1. **Open Browser**: http://localhost:3000
2. **Record Voice**: Click "Start Recording" → Speak → "Stop Recording"
3. **OR Upload File**: Click "Choose File" → Select audio file
4. **Enter Text**: Type what you want the voice to say
5. **Select Language**: Choose from dropdown
6. **Generate**: Click "Generate Voice"
7. **Download**: Save the synthesized audio

### **Emotion Features**
1. **Automatic Detection**: System detects emotion from your voice
2. **Emotion Display**: See confidence scores and VAD profiles
3. **Feedback**: Correct emotion if wrong (helps improve model)
4. **Quality Metrics**: View voice similarity and confidence

### **Advanced Features**
- **Voice Similarity**: Compare generated voice with original
- **VAD Profiles**: View Valence/Arousal/Dominance scores
- **Active Learning**: System improves from your feedback
- **Progressive Accuracy**: Model gets smarter over time

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **Models Not Downloading**
```bash
# Check internet connection
ping huggingface.co

# Manual download (if auto fails)
python -c "import whisper; whisper.load_model('base')"
```

#### **GPU Not Detected**
```bash
# Check CUDA installation
python -c "import torch; print(torch.version.cuda)"

# Install PyTorch with CUDA (if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **Port Conflicts**
```bash
# Kill processes on ports 8000/3000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Mac/Linux:
lsof -ti:8000 | xargs kill -9
```

#### **Memory Issues**
```bash
# Reduce batch size in training
# Use CPU instead of GPU (slower but less memory)
```

### **Error Messages**
- **"Model not found"**: Wait for downloads to complete
- **"CUDA out of memory"**: Restart with CPU-only mode
- **"Port already in use"**: Kill conflicting processes

---

## 🧠 **Model Training (Advanced)**

### **Improve Emotion Detection**
```bash
cd emotion_training

# Train with user feedback
python train.py

# Train with archived data
python train_on_archived.py

# Evaluate model
python evaluate.py
```

### **Active Learning**
1. **Use the system normally**
2. **Provide emotion feedback** when wrong
3. **System auto-trains** after 5+ feedback files
4. **Progressive accuracy** improves over time

---

## 📁 **Project Structure**

```
EAMVCS/
├── Backend/                 # FastAPI server
│   ├── app.py              # Main server application
│   ├── emotion_detector.py # Emotion classification
│   ├── encoders/           # Voice embedding models
│   └── tts/                # Text-to-speech modules
├── Frontend/
│   └── UI/                 # Web interface
│       ├── index.html      # Main page
│       ├── script.js       # Frontend logic
│       └── styles.css      # UI styling
├── emotion_training/       # Model training scripts
│   ├── train.py           # Main training script
│   ├── model.py           # EmotionCNN architecture
│   └── evaluate.py        # Model evaluation
├── requirements.txt        # Python dependencies
├── start_neurovoice.ps1   # Startup script
└── README.md              # This file
```

---

## 🤝 **Contributing**

### **Development Setup**
```bash
# Clone repository
git clone https://github.com/TgeB4tMan/EAMVCS.git
cd EAMVCS

# Setup development environment
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Start development servers
.\start_neurovoice.ps1
```

### **Code Style**
- Python: Follow PEP 8
- JavaScript: Use ES6+ standards
- Comments: Explain complex logic
- Commits: Use conventional commit format

---

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🆘 **Support**

### **Get Help**
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Ask questions on GitHub Discussions
- **Documentation**: Check inline code comments

### **Common Questions**
- **Q: Can I use my own voice model?** A: Yes, replace models in models/ directory
- **Q: How accurate is emotion detection?** A: ~64% accuracy, improves with feedback
- **Q: Can I add new languages?** A: Yes, add to translation_service.py
- **Q: Is GPU required?** A: No, but highly recommended for performance

---

## 🎉 **Success!**

You now have a fully functional emotion-aware voice cloning system! 

**🚀 Start creating emotional voices in multiple languages today!**

---

**Built with ❤️ using PyTorch, FastAPI, and modern AI technologies**
