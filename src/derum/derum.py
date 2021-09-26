import numpy as np
import tensorflow as tf

from note_decoder import global_envelope_v2, N_ADSR_CONTROL as N_CONTROLS
from unison import unison

FREQ_NOTES = np.array([440 * (2 ** (1/12)) ** n for n in range(-46, 40)])
N_NOTES = len(FREQ_NOTES)

N_ADSR_CONTROL = N_CONTROLS

def derum(notes, adsr_control, detune, wavetable, n_samples, sample_rate):
    """
    Generates an output audio signal from notes, ADSR controls,
    detune amplitude and wavetable.

    Shapes:
    - notes [batch, time (note), N_NOTES]
    - adsr_control [batch, N_ADSR_CONTROL]
    - detune [batch, time (auto)]
    - wavetable [batch, time (auto), time_wavetable (any)]
    """

    # assert proper shapes individually
    assert len(notes.shape) == 3 and notes.shape[2] == N_NOTES
    assert len(adsr_control.shape) == 2 and adsr_control.shape[1] == N_ADSR_CONTROL
    assert len(detune.shape) == 2
    assert len(wavetable.shape) == 3

    # assert batch consistency
    assert notes.shape[0] == adsr_control.shape[0] == detune.shape[0] == wavetable.shape[0]

    note_signals = []
    for k, f_0 in enumerate(FREQ_NOTES):
        with tf.name_scope(f"note-{k}"):
            note_track = tf.expand_dims(notes[:,:,k], 2)
            envelope = global_envelope_v2(note_track, adsr_control, detune.shape[1])
            f_0 = np.full(envelope.shape, f_0)
            note_signals.append(unison(f_0, detune, envelope, wavetable, n_samples, sample_rate))
    
    return tf.math.accumulate_n(note_signals)
