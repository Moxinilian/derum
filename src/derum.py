import numpy as np
import tensorflow as tf

from note_decoder import global_envelope_v1
from unison import unison

FREQ_NOTES = np.array([440 * (2 ** (1/12)) ** n for n in range(-46, 40)])

N_NOTES = len(FREQ_NOTES)

@tf.function
def derum(notes, note_envelope, detune, wavetable, n_samples, sample_rate):
    """
    Generates an output audio signal from notes, envelope filter,
    detune amplitude and wavetable.

    Shapes:
    - notes [batch, time (note), N_NOTES]
    - note_envelope [envelope_time (auto), 1, 1]
    - detune [batch, time (auto)]
    - wavetable [batch, time (auto), time_wavetable (any)]
    """

    # assert proper shapes individually
    assert len(notes.shape) == 3 and notes.shape[2] == N_NOTES
    assert len(note_envelope.shape) == 3 and note_envelope.shape[2] == 1
    assert len(detune.shape) == 2
    assert len(wavetable.shape) == 3

    note_signals = []
    for k, f_0 in enumerate(FREQ_NOTES):
        with tf.name_scope(f"note-{k}"):
            note_track = tf.expand_dims(notes[:,:,k], 2)
            envelope = global_envelope_v1(note_track, note_envelope, detune.shape[1])
            f_0 = np.full(envelope.shape, f_0)
            note_signals.append(unison(f_0, detune, envelope, wavetable, n_samples, sample_rate))
    
    return tf.math.accumulate_n(note_signals)
