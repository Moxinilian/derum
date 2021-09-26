import derum
import tensorflow as tf
import numpy as np
import scipy.io.wavfile

AUTOMATION_RATE = 800 # automation points/second
SAMPLE_RATE = 44000 # samples/second

NOTE_PER_SECOND = 2

# triangle wave
WAVE = [k/250 - 1 for k in range(500)] + [1 - k/250 for k in range(500)]

# uncomment for saw wave
# WAVE = [k/500 - 1 for k in range(1000)]

# uncomment for square wave
# WAVE = [-1 for _ in range(500)] + [1 for _ in range(500)]

# uncomment for sine wave
# WAVE = [np.sin(k/1000 * 2 * np.pi) for k in range(1000)]

FREQ_NOTES = np.array([440 * (2 ** (1/12)) ** n for n in range(-9, 4)])

notes = np.zeros((1, 8, 13), dtype=np.float32)
notes[0,0,0] = 1
notes[0,1,2] = 1
notes[0,2,4] = 1
notes[0,3,5] = 1
notes[0,4,7] = 1
notes[0,5,9] = 1
notes[0,6,11] = 1
notes[0,7,12] = 1

adsr_control = np.array([[0.1, 0.1, 0.8, 0.1, 8, 8]], dtype=np.float32)

automation_len = 8 * AUTOMATION_RATE // NOTE_PER_SECOND

detune = np.zeros((1, automation_len), dtype=np.float32)

wavetable = np.array([[WAVE] * automation_len], dtype=np.float32)

n_samples = 8 * SAMPLE_RATE // NOTE_PER_SECOND

@tf.function
def run():
    return derum.synth(notes, adsr_control, detune, wavetable, n_samples, SAMPLE_RATE, AUTOMATION_RATE, freq_notes=FREQ_NOTES)

audio = run()[0]

scipy.io.wavfile.write("output.wav", SAMPLE_RATE, audio.numpy())