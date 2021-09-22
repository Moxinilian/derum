import tensorflow as tf
import numpy as np
import ddsp

from utils import right_handed_conv

# ADSR controls: [a (secs), d (secs), s, r (secs), aslope, rslope]
N_ADSR_CONTROL = 6

DECAY_SLOPE = 8

# TODO: Support batches larger than 1

def build_attack_decay_filter(adsr_control, attack_decay_filter_len, automation_rate):
    a = adsr_control[:,0]
    d = adsr_control[:,1]
    s = adsr_control[:,2]
    aslope = a = adsr_control[:,4]
    ad_filter_concat = []
    for k in range(attack_decay_filter_len):
        t = k / automation_rate
        as_attack = (tf.exp(aslope/a * t) - 1) / (tf.exp(aslope) - 1)
        as_decay = (1 - s) * tf.exp(-DECAY_SLOPE * (t-a) / d) + s
        v = tf.where(t < a, as_attack, as_decay)
        ad_filter_concat.append(tf.reshape(v, (v.shape[0], 1)))
    return tf.concat(ad_filter_concat, axis=1)

def build_release_filter(adsr_control, release_filter_len, automation_rate):
    s = adsr_control[:,2]
    r = adsr_control[:,3]
    rslope = adsr_control[:,5]
    t = tf.repeat(tf.range(release_filter_len, dtype=tf.float32) / automation_rate, adsr_control.shape[0])
    t = tf.reshape(t, (adsr_control.shape[0], release_filter_len))
    return tf.nn.relu(s * (1 - (tf.exp(rslope/r * t) - 1) / (tf.exp(rslope) - 1)))

def global_envelope_v2(notes, adsr_control, automation_len, automation_rate):
    """
    Generates the envelope profile of a track of notes.
    This "version 2" method creates an ADSR profile differentiably
    and applies it to the note signal.

    Shapes:
    - notes [batch, time (note)]
    - adsr_control [batch, N_ADSR_CONTROL]
    Output:
    - global_envelope [batch, time (auto)]
    """

    assert len(notes.shape) == 3 and notes.shape[2] == 1
    assert len(adsr_control.shape) == 2 and adsr_control.shape[1] == N_ADSR_CONTROL

    assert notes.shape[0] == adsr_control.shape[0]

    notes = ddsp.core.resample(notes, automation_len, method="nearest", add_endpoint=False)
    
    notes_extended = tf.concat([notes, tf.zeros((notes.shape[0], 1))], axis=1)
    notes_diff = notes_extended[:,:-1] - notes_extended[:,1:]

    attack_decay_triggers = tf.nn.relu(notes_diff)
    release_triggers = tf.nn.relu(-notes_diff)

    # NOTE: We could pick a more sensible value, surely most a+d and r are shorter
    # than the whole output
    attack_decay_filter_len = automation_len
    release_filter_len = automation_len

    # TODO
    # Apply the attack+decay and release convolutions
    # Formula for release:
    # release = lambda x: s * (1 - (tf.exp(rslope/r * x) - 1) / (tf.exp(rslope) - 1))


def global_envelope_v1(notes, note_envelope, automation_len):
    """
    Generates the envelope profile of a track of notes.
    This "version 1" method only applies a convolution to
    the note signal.

    Shapes:
    - notes [batch, time (note), 1]
    - note_envelope [time (auto), 1, 1]
    Output:
    - global_envelope [batch, time (auto)]
    """

    assert len(notes.shape) == 3 and notes.shape[2] == 1
    assert len(note_envelope.shape) == 3 and note_envelope.shape[2] == 1

    notes = ddsp.core.resample(notes, automation_len, method="nearest", add_endpoint=False)
    env_auto = tf.nn.conv1d(notes, note_envelope, [1], "SAME")
    return env_auto
