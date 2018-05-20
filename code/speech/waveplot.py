import librosa
import librosa.display
from matplotlib import pyplot as plt


yes, sr = librosa.load("data_speech_commands_v0.01/yes/0a7c2a8d_nohash_0.wav",
                      sr=None)

librosa.display.waveplot(yes, sr)
plt.savefig("wave_yes")
plt.show()


#yes_mel = librosa.feature.melspectrogram(yes, sr=sr, n_fft=400, hop_length=160)
#yes_mel = librosa.power_to_db(yes_mel)
#librosa.display.specshow(yes_mel)
#plt.show()

yes_mel = librosa.stft(yes, n_fft=400, hop_length=160)
yes_mel = librosa.amplitude_to_db(yes_mel)
librosa.display.specshow(yes_mel, sr=sr, hop_length=160, x_axis="time",
                         y_axis="hz")
plt.savefig("stft_yes")
plt.show()
