import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import EmotionCNN
from tqdm import tqdm
import json
from pathlib import Path
import librosa
import soundfile as sf

import random


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EMOTION_MAP = {
    '01': 0, # Neutral
    '02': 1, # Happy
    '03': 2, # Sad
    '04': 3, # Angry
}

def get_previous_accuracy():
    """Load previous accuracy from metadata file"""
    try:
        if os.path.exists("model_metadata.json"):
            with open("model_metadata.json", "r") as f:
                metadata = json.load(f)
                return metadata.get("best_accuracy", 55.0)
    except:
        return 55.0

def save_model_metadata(accuracy):
    """Save model accuracy to metadata file"""
    metadata = {
        "best_accuracy": accuracy,
        "last_updated": str(Path().resolve()),
        "model_type": "emotion_cnn"
    }
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

class RobustEmotionDataset(Dataset):
    """Dataset with heavy augmentation for real-world performance"""
    def __init__(self, data_dirs, target_sample_rate=16000, fixed_length=3.0, augment=False):
        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.target_sample_rate = target_sample_rate
        self.fixed_length_samples = int(target_sample_rate * fixed_length)
        self.augment = augment
        self.files = []
        self.labels = []
        self.actors = []
        
        # Folder to ID mapping
        self.folder_map = {
            'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
            'calm': 0 # Map calm to neutral for simplicity
        }
        
        # Audio Transforms
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate, n_fft=1024, hop_length=512, n_mels=64
        )
        
        # Augmentation Transforms
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)
        
        for data_dir in self.data_dirs:
            if not os.path.exists(data_dir):
                print(f"Warning: Data directory {data_dir} not found.")
                continue
            for root, _, files in os.walk(data_dir):
                # Identify if we are in an emotion folder (e.g., .../happy/...)
                folder_name = os.path.basename(root).lower()
                folder_label = self.folder_map.get(folder_name)
                
                for file in files:
                    if file.endswith('.wav'):
                        file_path = os.path.join(root, file)
                        
                        # PRIORITY 1: RAVDESS Format (03-01-0x-...)
                        parts = file.split('-')
                        if len(parts) >= 7 and parts[2] in EMOTION_MAP:
                            self.files.append(file_path)
                            self.labels.append(EMOTION_MAP[parts[2]])
                            self.actors.append(f"ravdess_{parts[6].split('.')[0]}")
                        
                        # PRIORITY 2: Folder-based Format (if not RAVDESS)
                        elif folder_label is not None:
                            self.files.append(file_path)
                            self.labels.append(folder_label)
                            # User feedback/training_data uses unique timestamps as actor IDs to prevent overfitting
                            self.actors.append(f"user_{file}")

    def __len__(self):
        return len(self.files)

    def _add_noise(self, waveform, noise_level=0.005):
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except Exception as e:
            # Fallback for browser-encoded WAVs missing standard RIFF headers
            try:
                y, sr = librosa.load(file_path, sr=None)
                waveform = torch.from_numpy(y).unsqueeze(0).to(torch.float32)
                sample_rate = sr
            except Exception as inner_e:
                print(f"Error loading {file_path}: {e} / {inner_e}")
                # Create a silent dummy waveform to prevent crashing the batch
                waveform = torch.zeros(1, self.target_sample_rate)
                sample_rate = self.target_sample_rate
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)
            
        # Robust Augmentation during Training
        # Skip noise injection for real mic recordings - they already have natural noise
        is_feedback = "user_feedback_data" in file_path or "feedback_" in os.path.basename(file_path)
        if self.augment:
            if not is_feedback:
                # 1. Random Noise Injection (only for clean RAVDESS studio recordings)
                if random.random() < 0.5:
                    waveform = self._add_noise(waveform, noise_level=random.uniform(0.001, 0.01))
            
            # 2. Light Pitch Shifting for all files (reduced range for feedback)
            pitch_range = 1 if is_feedback else 2
            if random.random() < 0.3:
                pitch_shift = random.randint(-pitch_range, pitch_range)
                waveform = torchaudio.functional.pitch_shift(waveform, self.target_sample_rate, pitch_shift)

        # Pad or truncate to fixed length
        if waveform.shape[1] < self.fixed_length_samples:
            padding = self.fixed_length_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            start = random.randint(0, waveform.shape[1] - self.fixed_length_samples) if self.augment else 0
            waveform = waveform[:, start:start+self.fixed_length_samples]
            
        # Convert to Mel Spectrogram
        mel = self.mel_spectrogram(waveform)
        mel = torch.log2(mel + 1e-9)
        
        # Spectral Augmentations
        if self.augment:
            mel = self.freq_mask(mel)
            mel = self.time_mask(mel)
            
        return mel, label

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Training Robust Emotion CNN on {device}")
    
    candidate_dirs = [
        os.path.join(PROJECT_ROOT, "Data", "archive"),
        os.path.join(PROJECT_ROOT, "training_data"),
        os.path.join(PROJECT_ROOT, "user_feedback_data"),
    ]
    data_dirs = [path for path in candidate_dirs if os.path.exists(path)]
    
    # Load base dataset to split by actors (prevent speaker leakage)
    full_dataset = RobustEmotionDataset(data_dirs, augment=False)
    if not full_dataset.files:
        print(f" No data found! Checked: {candidate_dirs}")
        return

    unique_actors = list(set(full_dataset.actors))
    random.shuffle(unique_actors)
    
    split_idx = int(0.85 * len(unique_actors))
    train_actors = unique_actors[:split_idx]
    
    train_indices = [i for i, a in enumerate(full_dataset.actors) if a in train_actors]
    val_indices = [i for i, a in enumerate(full_dataset.actors) if a not in train_actors]
    
    print(f"Loaded {len(full_dataset)} samples. Actors: {len(unique_actors)} (Split: {len(train_actors)} train / {len(unique_actors)-len(train_actors)} val)")
    
    train_ds = torch.utils.data.Subset(RobustEmotionDataset(data_dirs, augment=True), train_indices)
    val_ds = torch.utils.data.Subset(RobustEmotionDataset(data_dirs, augment=False), val_indices)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    model = EmotionCNN(num_classes=4).to(device)
    
    # FINETUNING LOGIC: Build on the existing 67% model
    model_path = "emotion_model.pth"
    is_finetuning = False
    if os.path.exists(model_path):
        try:
            print(f" Loading existing model weights from {model_path} for finetuning...")
            model.load_state_dict(torch.load(model_path, map_location=device))
            is_finetuning = True
            print(" Pre-existing model loaded successfully.")
        except Exception as e:
            print(f" Could not load existing weights (Architecture changed?): {e}")
            print("Starting training from scratch.")

    criterion = nn.CrossEntropyLoss()
    # Use a smaller learning rate if finetuning to avoid destroying learned patterns
    lr = 0.0001 if is_finetuning else 0.0005
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    
    num_epochs = 3 if is_finetuning else 50  # Fine-tuning: 3 precise passes; full training: 50 epochs
    
    # DYNAMIC ACCURACY GUARD: Start from previous best or reasonable baseline
    # This ensures continuous improvement from where we left off
    if is_finetuning and os.path.exists("emotion_model.pth"):
        # Load previous model's accuracy if available
        try:
            # Try to get previous accuracy from model metadata
            previous_acc = get_previous_accuracy()
            best_acc = max(previous_acc, 55.0)  # Min 55% to avoid regression
            print(f"📈 Starting from previous best: {previous_acc:.1f}% | Target: {best_acc:.1f}%")
        except:
            best_acc = 55.0  # Conservative baseline if no history
            print(f"🎯 Starting with conservative target: {best_acc}% Accuracy")
    else:
        best_acc = 64.0 if is_finetuning else 0.0
        print(f" Target to beat: {best_acc}% Accuracy")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            _, logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = torch.max(logits, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                _, logits = model(features)
                _, pred = torch.max(logits, 1)
                v_correct += (pred == labels).sum().item()
                v_total += labels.size(0)
        
        val_acc = 100 * v_correct / v_total
        scheduler.step()
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "emotion_model.pth")
            save_model_metadata(best_acc)  # Save accuracy to metadata
            print(f"✨ New Best Model Saved! (Acc: {best_acc:.2f}%)")
        else:
            print(f"⏸️ Model not improved (Current: {val_acc:.1f}% < Best: {best_acc:.1f}%)")

if __name__ == "__main__":
    try:
        train()
    finally:
        # Always remove the lock file
        if os.path.exists("training_in_progress.lock"):
            os.remove("training_in_progress.lock")
            print(" Training lock released.")
            
        # ARCHIVE FEEDBACK: Move files from user_feedback_data to an archive folder 
        # so they don't trigger training again immediately.
        import shutil
        from datetime import datetime
        
        feedback_dir = os.path.join(PROJECT_ROOT, "user_feedback_data")
        archive_root = os.path.join(PROJECT_ROOT, "feedback_archive")
        
        if os.path.exists(feedback_dir):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_archive = os.path.join(archive_root, f"session_{timestamp}")
            
            # Find all feedback files
            files_to_archive = []
            for root, _, files in os.walk(feedback_dir):
                for f in files:
                    if f.endswith('.wav'):
                        files_to_archive.append(os.path.join(root, f))
            
            if files_to_archive:
                os.makedirs(session_archive, exist_ok=True)
                print(f" Archiving {len(files_to_archive)} feedback files to prevent re-triggering...")
                for f_path in files_to_archive:
                    try:
                        # Keep the emotion subfolder structure in the archive
                        emotion_cat = os.path.basename(os.path.dirname(f_path))
                        target_dir = os.path.join(session_archive, emotion_cat)
                        os.makedirs(target_dir, exist_ok=True)
                        shutil.move(f_path, os.path.join(target_dir, os.path.basename(f_path)))
                    except Exception as e:
                        print(f"Error archiving {f_path}: {e}")
                print(" Feedback folder reset for next session.")
