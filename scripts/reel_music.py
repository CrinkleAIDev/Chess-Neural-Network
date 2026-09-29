"""Original, royalty-free synth track for the reel (generated from scratch with numpy).

Structure follows the video: a quiet intro, a build under the training scene,
the full beat under the rating graph and the final game, then an outro.
"""
import sys
import wave

import numpy as np

SR = 44100
BPM = 112
BEAT = 60 / BPM
BAR = 4 * BEAT
LENGTH = 47.0

# A minor progression, two bars per chord: Am - F - C - G
CHORDS = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]


def hz(note):
    return 440.0 * 2 ** ((note - 69) / 12)


def lowpass(x, cutoff):
    """One-pole low-pass, vectorised enough for a few seconds of audio at a time."""
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = (1 - a) * x[i] + a * acc
        y[i] = acc
    return y


def saw(freq, t):
    return 2 * ((freq * t) % 1.0) - 1


def section_gain(t, points):
    """Piecewise-linear automation: points = [(time, gain), ...]."""
    times, gains = zip(*points)
    return np.interp(t, times, gains)


def main(path):
    n = int(LENGTH * SR)
    t = np.arange(n) / SR
    left = np.zeros(n)
    right = np.zeros(n)

    def chord_at(time):
        return CHORDS[int(time // (2 * BAR)) % len(CHORDS)]

    # Pad: detuned saws, soft attack, darkened. Built per chord for smooth voicing.
    pad_l = np.zeros(n); pad_r = np.zeros(n)
    for start in np.arange(0, LENGTH, 2 * BAR):
        s, e = int(start * SR), min(n, int((start + 2 * BAR) * SR))
        seg = t[s:e] - start
        env = np.minimum(1, seg / 0.6) * np.minimum(1, (2 * BAR - seg) / 0.3)
        for note in chord_at(start):
            f = hz(note)
            pad_l[s:e] += env * (saw(f * 0.997, seg) + saw(f * 1.004, seg)) * 0.5
            pad_r[s:e] += env * (saw(f * 1.003, seg) + saw(f * 0.995, seg)) * 0.5
    pad_l = lowpass(pad_l, 900) * 0.09
    pad_r = lowpass(pad_r, 900) * 0.09

    # Arp: sixteenth-note plucks over the chord, an octave up.
    arp = np.zeros(n)
    step = BEAT / 4
    pattern = [0, 1, 2, 1, 0, 2, 1, 2]
    for k, start in enumerate(np.arange(0, LENGTH, step)):
        chord = chord_at(start)
        note = chord[pattern[k % len(pattern)]] + 12 + (12 if k % 16 >= 12 else 0)
        s = int(start * SR); e = min(n, s + int(0.22 * SR))
        seg = np.arange(e - s) / SR
        f = hz(note)
        tone = np.sign(np.sin(2 * np.pi * f * seg)) * 0.5 + np.sin(2 * np.pi * f * seg) * 0.5
        arp[s:e] += tone * np.exp(-seg * 18)
    arp = lowpass(arp, 2600) * 0.11

    # Bass: eighth notes on the chord root.
    bass = np.zeros(n)
    for start in np.arange(0, LENGTH, BEAT / 2):
        root = chord_at(start)[0] - 24
        s = int(start * SR); e = min(n, s + int(BEAT / 2 * 0.9 * SR))
        seg = np.arange(e - s) / SR
        f = hz(root)
        bass[s:e] += (np.sin(2 * np.pi * f * seg) + 0.3 * saw(f, seg)) * np.exp(-seg * 3)
    bass = lowpass(bass, 400) * 0.35

    # Drums.
    kick = np.zeros(n); hat = np.zeros(n); clap = np.zeros(n)
    rng = np.random.default_rng(7)
    for start in np.arange(0, LENGTH, BEAT):
        s = int(start * SR); e = min(n, s + int(0.35 * SR))
        seg = np.arange(e - s) / SR
        sweep = 45 + 90 * np.exp(-seg * 30)
        kick[s:e] += np.sin(2 * np.pi * np.cumsum(sweep) / SR) * np.exp(-seg * 9)
        beat_index = round(start / BEAT)
        if beat_index % 2 == 1:
            e2 = min(n, s + int(0.18 * SR))
            seg2 = np.arange(e2 - s) / SR
            clap[s:e2] += rng.standard_normal(e2 - s) * np.exp(-seg2 * 22)
        hs = int((start + BEAT / 2) * SR); he = min(n, hs + int(0.05 * SR))
        if hs < n:
            noise = rng.standard_normal(he - hs)
            hat[hs:he] += (noise - np.concatenate(([0], noise[:-1]))) * np.exp(-np.arange(he - hs) / SR * 90)
    clap = lowpass(clap, 3500)

    # Sidechain pump from the kick grid.
    phase = (t % BEAT) / BEAT
    pump = 1 - 0.55 * np.exp(-phase * 7)

    # Arrangement (seconds): intro 0-4.5, build 4.5-10, groove 10-30, peak 30-41, outro 41-47.
    g_pad = section_gain(t, [(0, 0.0), (1.5, 1.0), (30, 1.0), (41, 1.2), (47, 0.0)])
    g_arp = section_gain(t, [(0, 0.5), (4.5, 0.8), (10, 1.0), (41, 1.0), (46, 0.0)])
    g_bass = section_gain(t, [(0, 0), (4.4, 0), (4.6, 0.7), (10, 1.0), (41, 1.0), (45, 0)])
    g_kick = section_gain(t, [(0, 0), (9.9, 0), (10.0, 1.0), (41, 1.0), (41.1, 0)])
    g_hat = section_gain(t, [(0, 0), (4.5, 0), (4.6, 0.5), (10, 1.0), (41, 1.0), (43, 0)])
    g_clap = section_gain(t, [(0, 0), (18.9, 0), (19.0, 1.0), (41, 1.0), (41.1, 0)])
    duck = np.where(g_kick > 0, pump, 1.0)

    mid = (bass * g_bass + arp * g_arp) * duck + kick * g_kick * 0.8 + clap * g_clap * 0.18
    left = mid + pad_l * g_pad * duck + hat * g_hat * 0.10
    right = mid + pad_r * g_pad * duck + hat * g_hat * 0.08
    stereo = np.stack([left, right], axis=1)
    stereo = np.tanh(stereo * 1.4) / np.tanh(1.4)
    stereo *= 0.89 / np.max(np.abs(stereo))
    fade = np.clip((LENGTH - t) / 2.5, 0, 1)[:, None]
    stereo *= fade
    pcm = (stereo * 32767).astype("<i2")
    with wave.open(path, "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(SR)
        out.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/reel/music.wav")
