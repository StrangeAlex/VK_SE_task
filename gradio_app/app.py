MODEL_TYPE = "SMALL" # Или LARGE

import torch
import librosa
import os
import json
import soundfile as sf
import tempfile
import gradio as gr
from utils import mag_pha_stft, mag_pha_istft

import matplotlib.pyplot as plt
import numpy as np

if MODEL_TYPE == "LARGE":
    from models_large.model import MPNet
else:
    from models_small.model import MPNet
from utils import AttrDict

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load_model(checkpoint_file):
    config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        json_config = json.load(f)
    h = AttrDict(json_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MPNet(h).to(device)

    checkpoint_dict = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint_dict['generator'])
    model.eval()

    return model, h, device


def plot_spectrogram(mag, title):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(20 * np.log10(mag + 1e-8), origin='lower', aspect='auto', cmap='magma')
    ax.set_title(title)
    ax.set_xlabel("Время")
    ax.set_ylabel("Частота")
    ax.set_yticks([])
    ax.set_xticks([])
    fig.tight_layout()

    temp_img_path = tempfile.mktemp(suffix=".png")
    fig.savefig(temp_img_path)
    plt.close(fig)
    return temp_img_path


if MODEL_TYPE == "LARGE":
    checkpoint_file = 'checkpoints/large_model/g_00157500'
else:
    checkpoint_file = 'checkpoints/small_model/g_00057500'

model, h, device = load_model(checkpoint_file)

def enhance_audio(noisy_audio):
    noisy_wav, _ = librosa.load(noisy_audio, sr=h.sampling_rate)
    noisy_wav = torch.FloatTensor(noisy_wav).to(device)
    norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
    noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)

    noisy_amp, noisy_pha, _ = mag_pha_stft(noisy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    with torch.no_grad():
        amp_g, pha_g, _ = model(noisy_amp, noisy_pha)
        audio_g = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
        audio_g = audio_g / norm_factor

    temp_audio_path = tempfile.mktemp(suffix=".wav")
    sf.write(temp_audio_path, audio_g.squeeze().cpu().numpy(), h.sampling_rate, 'PCM_16')

    noisy_spec_img = plot_spectrogram(noisy_amp.squeeze().cpu().numpy(), "Зашумленная Спектрограмма")
    clean_spec_img = plot_spectrogram(amp_g.squeeze().cpu().numpy(), "Очищенная Спектрограмма")

    return noisy_spec_img, clean_spec_img, temp_audio_path


def gradio_ui():
    with gr.Blocks(title="Улучшение речи") as demo:
        gr.Markdown(f"# Демоверсия модели Speech Enhancement размера {MODEL_TYPE}")
        gr.Markdown("## Загрузите зашумлённую аудиозапись и нажмите \"**Очистить**\", чтобы удалить шум.")

        audio_input = gr.Audio(type="filepath", label="Зашумлённое аудио")

        submit_btn = gr.Button("Очистить", size="sm")

        audio_output = gr.Audio(type="filepath", label="Очищенное аудио")

        gr.Markdown("---")
        gr.Markdown("### Сравнение спектрограмм")

        with gr.Row():
            noisy_img = gr.Image(label="Спектрограмма шума")
            clean_img = gr.Image(label="Спектрограмма после очистки")

        submit_btn.click(fn=enhance_audio, inputs=audio_input, outputs=[noisy_img, clean_img, audio_output])

    demo.launch()



if __name__ == '__main__':
    gradio_ui()