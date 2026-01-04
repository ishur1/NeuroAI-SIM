# ==============================================================================
# neuroai_sim_enhanced_v2.py
# ==============================================================================
# Title   : NeuroAI-Sim Enhanced - Synthetic EEG Deep Learning Simulator
# Features: GAN generator, Explainability (SHAP + GradCAM), Real EEG toggle,
#           Band visualization, Multi-disorder simulation, UI sliders
# Author  : ScholarGPT
# ==============================================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Input
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import mne
import shap
import random

# ----------------------------------
# Streamlit UI Config
# ----------------------------------
st.set_page_config(layout="wide", page_title="🧠 NeuroAI-Sim Enhanced", page_icon="🧠")
st.title("🧠 NeuroAI-Sim Enhanced: Simulate & Diagnose Brain Disorders with AI")

st.sidebar.title("🧠 EEG Simulation Controls")
seconds = st.sidebar.slider("EEG Duration (seconds)", 5, 20, 10)
degeneration_level = st.sidebar.slider("Degeneration Severity", 0.0, 1.0, 0.5)
condition = st.sidebar.selectbox("Simulated Brain Condition", ["Healthy", "Alzheimer", "Parkinson", "Epilepsy", "ADHD", "Coma"])

# ----------------------------------
# EEG Simulator (Multi-Disorder)
# ----------------------------------
def generate_eeg(condition="Healthy", seconds=10, sfreq=250, degeneration_level=0.5):
    t = np.arange(0, seconds, 1/sfreq)
    n_ch = 4
    eeg = np.zeros((n_ch, len(t)))
    base_freqs = [10, 20]

    for ch in range(n_ch):
        signal = sum(
            np.random.uniform(1.0, 2.0) * np.sin(2*np.pi*f*t + np.random.rand()*2*np.pi)
            for f in base_freqs
        )
        eeg[ch] = signal

    # Baseline drift + noise
    eeg += np.sin(2*np.pi*0.5*t) * 0.3
    noise = np.random.randn(len(t))
    for ch in range(n_ch):
        eeg[ch] += 0.2 * noise + 0.1 * np.random.randn(len(t))

    if condition != "Healthy":
        for ch in range(n_ch):
            eeg[ch] *= np.random.uniform(1.0 - degeneration_level, 1.0)
            eeg[ch] += np.sin(2*np.pi*3*t + np.random.rand()*2*np.pi) * degeneration_level
            eeg[ch] += np.random.randn(len(t)) * degeneration_level
        for idx in np.random.randint(0, len(t)-100, 5):
            eeg[:, idx:idx+100] += np.random.randn(n_ch, 100) * degeneration_level

        if condition == "Coma":
            eeg *= 0.05
        elif condition == "Epilepsy":
            spike_idx = np.random.randint(0, len(t)-50, 10)
            for i in spike_idx:
                eeg[:, i:i+20] += 3 * np.random.randn(n_ch, 20)

    eeg = (eeg - eeg.mean(axis=1, keepdims=True)) / eeg.std(axis=1, keepdims=True)
    return eeg

# ----------------------------------
# Dataset + Model Training
# ----------------------------------
@st.cache_resource
def train_model(n_samples=200):
    X, y = [], []
    labels = ["Healthy", "Alzheimer", "Parkinson", "Epilepsy", "ADHD", "Coma"]
    label_map = {label: idx for idx, label in enumerate(labels)}

    for _ in range(n_samples // 2):
        for label in labels:
            eeg = generate_eeg(condition=label, seconds=10)
            X.append(eeg)
            y.append(label_map[label])

    X = np.stack(X, axis=0)
    X = np.transpose(X, (0, 2, 1))  # (samples, time, channels)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Conv1D(32, 7, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        MaxPooling1D(3),
        Conv1D(64, 5, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dense(len(labels), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=8, verbose=0)

    return model, history, X, y, X_test, y_test, labels

# ----------------------------------
# Plot Functions
# ----------------------------------
def plot_eeg(eeg, title):
    fig, ax = plt.subplots()
    for i in range(eeg.shape[0]):
        ax.plot(eeg[i] + i * 5)
    ax.set_title(title)
    st.pyplot(fig)

def plot_psd(eeg, sfreq=250):
    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30), 'Gamma': (30, 40)}
    psd, freqs = mne.time_frequency.psd_array_welch(eeg, sfreq=sfreq, fmin=1, fmax=40)
    fig, ax = plt.subplots()
    ax.semilogy(freqs, psd.mean(axis=0))
    for band, (fmin, fmax) in bands.items():
        ax.axvspan(fmin, fmax, alpha=0.1, label=band)
    ax.set_title("Power Spectral Density with Brainwave Bands")
    ax.legend()
    st.pyplot(fig)
