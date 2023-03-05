import librosa
import numpy as np
import soundfile as sf
import torch.nn as nn
import os
import torch


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=0)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# load audio file
audio_file = "secondSense/res/classic.wav"
signal, sr = librosa.load(audio_file, sr=22050)

# compute MFCCs
mfccs = librosa.feature.mfcc(signal=signal, sr=sr, n_mfcc=20)
mfccs = (mfccs - mfccs.mean()) / mfccs.std()  # normalize MFCCs

# add random noise to MFCCs
noise = np.random.normal(scale=0.2, size=mfccs.shape)
input_data = mfccs + 0.5 * noise

# create autoencoder and load weights
autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load('autoencoder_weights.pth', map_location=torch.device('cpu')))

# encode and decode data
outputs = autoencoder(input_data)

# save decoded data as WAV file
decoded = outputs.squeeze().squeeze().T
sf.write("example_output.wav", decoded, sr, subtype='PCM_16')

# play the decoded file
os.system("aplay example_output.wav")
