from re import A
import tensorflow as tf
import numpy as np
import ddsp

from utils import right_handed_conv

# ADSR controls: [a (secs), d (secs), s, r (secs), aslope, rslope]
N_ADSR_CONTROL = 6

DECAY_SLOPE = 8

def build_attack_decay_filter(adsr_control, attack_decay_filter_len, automation_rate):
    a = tf.reshape(adsr_control[:,0], (adsr_control.shape[0], 1))
    d = tf.reshape(adsr_control[:,1], (adsr_control.shape[0], 1))
    s = tf.reshape(adsr_control[:,2], (adsr_control.shape[0], 1))
    aslope = tf.reshape(adsr_control[:,4], (adsr_control.shape[0], 1))
    t = tf.tile(tf.range(attack_decay_filter_len, dtype=tf.float32) / automation_rate, [adsr_control.shape[0]])
    t = tf.reshape(t, (adsr_control.shape[0], attack_decay_filter_len))
    as_attack = (tf.exp(aslope/a * t) - 1) / (tf.exp(aslope) - 1)
    as_decay = (1 - s) * tf.exp(-DECAY_SLOPE * (t-a) / d) + s
    return tf.where(t < a, as_attack, as_decay)

def build_release_filter(adsr_control, release_filter_len, automation_rate):
    s = tf.reshape(adsr_control[:,2], (adsr_control.shape[0], 1))
    r = tf.reshape(adsr_control[:,3], (adsr_control.shape[0], 1))
    rslope = tf.reshape(adsr_control[:,5], (adsr_control.shape[0], 1))
    t = tf.tile(tf.range(release_filter_len, dtype=tf.float32) / automation_rate, [adsr_control.shape[0]])
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

    assert len(notes.shape) == 2
    assert len(adsr_control.shape) == 2 and adsr_control.shape[1] == N_ADSR_CONTROL

    assert notes.shape[0] == adsr_control.shape[0]

    notes = ddsp.core.resample(notes, automation_len, method="nearest", add_endpoint=False)
    
    notes_extended = tf.concat([notes, tf.zeros((notes.shape[0], 1))], axis=1)
    notes_diff = notes_extended[:,1:] - notes_extended[:,:-1]

    attack_decay_triggers = tf.nn.relu(notes_diff)
    release_triggers = tf.nn.relu(-notes_diff)

    # FIXME (MEMORY): We could pick a more sensible value here, surely most a+d and r
    # are shorter than the whole output
    attack_decay_filter_len = automation_len
    release_filter_len = automation_len

    attack_decay_filter = build_attack_decay_filter(adsr_control, attack_decay_filter_len, automation_rate)
    release_filter = build_release_filter(adsr_control, release_filter_len, automation_rate)

    s = tf.reshape(adsr_control[:,2], (adsr_control.shape[0], 1))
    attack_decay_filter = attack_decay_filter - s

    attack_decays = right_handed_conv(attack_decay_triggers, attack_decay_filter)
    releases = right_handed_conv(release_triggers, release_filter)

    return attack_decays + notes * s + releases, attack_decay_triggers, release_triggers, attack_decays, releases


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
