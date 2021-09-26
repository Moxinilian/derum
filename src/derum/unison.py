import ddsp
import tensorflow as tf

UNISON_AMOUNT = 1

def detuner(k, f_0, d):
    if UNISON_AMOUNT > 1:
        # TODO: Fix this path, probably a silly shape mistake
        return f_0 + d * (2 * k / (UNISON_AMOUNT - 1) - 1)
    else:
        return f_0

def unison(f_0, detune, envelope, wavetable, n_samples, sample_rate):
    """
    Shapes:
    - f_0 [batch, time (note), 1]
    - detune [batch, time (auto)]
    - envelope [batch, time (auto), 1]
    - wavetable [batch, time (auto), time_wavetable (any)]
    """

    oscillators = []

    osc = ddsp.synths.Wavetable(n_samples, sample_rate, scale_fn=None)
    for k in range(UNISON_AMOUNT):
        freq = detuner(k, f_0, tf.expand_dims(detune, 2))
        oscillators.append(osc(envelope, wavetable, freq))

    return tf.math.accumulate_n(oscillators) / UNISON_AMOUNT
