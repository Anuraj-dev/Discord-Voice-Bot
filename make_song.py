# Builds song.wav: the Team TEJAS entrance track, synthesized from scratch (no samples, no TTS).
# 140 BPM trap in F minor: intro -> build -> drop -> break -> drop -> outro, with distorted 808s.
# Run: .venv/bin/python make_song.py   (needs ffmpeg for loudness normalization)
import os, subprocess, wave
import numpy as np

SR = 48000
BPM = 140
STEP = 60 / BPM / 4          # 16th note
BAR = 16 * STEP
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)

# Fm  Db  Ab  Eb  (i VI III VII)
CHORDS = [(53, 56, 60), (49, 53, 56), (56, 60, 63), (51, 55, 58)]
ROOTS = [29, 25, 32, 27]                      # 808 notes, octave 1
HOOK = [72, None, 75, 72, 70, None, 68, 70, 72, None, 77, None, 75, 72, 70, 68]  # 16ths, F minor pentatonic
SECTIONS = [('intro', 4), ('build', 4), ('drop', 8), ('break', 2), ('build2', 2), ('drop', 8), ('outro', 2)]

def hz(m): return 440.0 * 2 ** ((m - 69) / 12)
def tt(sec): return np.arange(int(sec * SR)) / SR
def at(sec): return int(sec * SR)

total_bars = sum(n for _, n in SECTIONS)
N = at(total_bars * BAR + 3)
drums, bass, music, fx = (np.zeros(N, np.float32) for _ in range(4))
kick_times = []

def add(buf, clip, sec, gain=1.0):
    i = at(sec); j = min(len(buf), i + len(clip))
    if 0 <= i < len(buf): buf[i:j] += (clip[: j - i] * gain).astype(np.float32)

def kick():
    x = tt(0.35); f = 48 + 110 * np.exp(-x * 35)
    click = rng.standard_normal(len(x)) * np.exp(-x * 400) * 0.3
    return np.tanh(2.2 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x * 7)) + click

def clap():
    x = tt(0.3); n = rng.standard_normal(len(x))
    n = n - np.concatenate([[0], n[:-1]]) * 0.6          # tilt toward highs
    env = sum(np.exp(-np.clip(x - d, 0, None) * 45) * (x >= d) for d in (0, 0.011, 0.022))
    return n * env * 0.5

def hat(open_=False):
    x = tt(0.25 if open_ else 0.045); n = np.diff(rng.standard_normal(len(x) + 1))
    return n * np.exp(-x * (14 if open_ else 110)) * 0.22

def crash():
    x = tt(2.2); n = np.diff(rng.standard_normal(len(x) + 1))
    return n * np.exp(-x * 1.8) * 0.3

def e808(note, dur, glide_to=None):
    x = tt(dur + 0.08)
    f0, f1 = hz(note), hz(glide_to if glide_to else note)
    f = f0 + (f1 - f0) * np.clip((x - dur * 0.55) / (dur * 0.35), 0, 1) if glide_to else np.full(len(x), f0)
    env = np.minimum(1, x / 0.004) * np.exp(-x * 0.9) * np.clip((dur + 0.08 - x) / 0.08, 0, 1)
    return np.tanh(3.0 * np.sin(2 * np.pi * np.cumsum(f) / SR)) * env * 0.8   # drive adds audible harmonics

def pluck(note, dur=STEP * 1.8, bright=1.0):
    x = tt(dur); f = hz(note)
    tone = sum(np.sin(2 * np.pi * f * k * x * (1 + d)) / k ** (1.6 - 0.4 * bright)
               for k in range(1, 7) for d in (-0.003, 0.003))
    return tone * np.exp(-x * 9) * 0.07

def pad(chord, dur, bright=1.0):
    x = tt(dur); env = np.minimum(1, x / 0.3) * np.clip((dur - x) / 0.2, 0, 1)
    tone = sum(np.sin(2 * np.pi * hz(m) * k * x * (1 + d)) / k ** (2.2 - bright)
               for m in chord for k in (1, 2, 3) for d in (-0.004, 0.004))
    return tone * env * 0.018

def riser(dur):
    x = tt(dur); p = x / dur
    sweep = np.sin(2 * np.pi * np.cumsum(200 + 1800 * p ** 2) / SR) * 0.08
    noise = np.diff(rng.standard_normal(len(x) + 1)) * 0.12
    return (sweep + noise) * p ** 2

t = 0.0
for name, bars in SECTIONS:
    for b in range(bars):
        t0 = t + b * BAR
        ci = b % 4
        chord, root = CHORDS[ci], ROOTS[ci]
        last = b == bars - 1
        if name == 'intro':
            add(music, pad(chord, BAR, bright=0.4), t0)
            for s, n in enumerate(HOOK):
                if n and s % 2 == 0: add(music, pluck(n, bright=0.3), t0 + s * STEP, 0.7)
            if b >= 2:
                for s in range(0, 16, 2): add(drums, hat(), t0 + s * STEP, 0.6)
        elif name in ('build', 'build2'):
            add(music, pad(chord, BAR, bright=0.7), t0)
            for s, n in enumerate(HOOK):
                if n: add(music, pluck(n, bright=0.6), t0 + s * STEP, 0.8)
            # snare roll that doubles in speed each bar of the build
            rate = [4, 2, 1, 0.5][min(3, b + (4 - bars))]
            k = 0.0
            while k < 16:
                add(drums, clap(), t0 + k * STEP, 0.35 + 0.35 * (b + k / 16) / bars)
                k += rate
            if b == 0: add(fx, riser(bars * BAR), t0)
        elif name == 'drop':
            if b == 0:
                add(fx, crash(), t0, 1.0)
            add(music, pad(chord, BAR, bright=1.0), t0, 0.8)
            for s, n in enumerate(HOOK):
                if n: add(music, pluck(n + (12 if b >= 4 and s % 4 == 0 else 0), bright=1.0), t0 + s * STEP)
            hits = [0, 7, 10] + ([14] if b % 2 else [])
            for i, s in enumerate(hits):
                add(drums, kick(), t0 + s * STEP, 1.0); kick_times.append(t0 + s * STEP)
                nxt = hits[i + 1] if i + 1 < len(hits) else 16
                glide = root + 12 if (b % 2 and s == 14) else None
                add(bass, e808(root, (nxt - s) * STEP, glide), t0 + s * STEP)
            add(drums, clap(), t0 + 8 * STEP, 0.9)
            for s in range(16):  # 16th hats with a 32nd-note roll into every other bar
                add(drums, hat(), t0 + s * STEP, 1.0 if s % 2 == 0 else 0.6)
                if b % 2 and s >= 12: add(drums, hat(), t0 + (s + 0.5) * STEP, 0.5)
            if b % 4 == 3: add(drums, hat(True), t0 + 14 * STEP, 0.8)
        elif name == 'break':
            add(music, pad(chord, BAR, bright=0.5), t0)
            for s, n in enumerate(HOOK):
                if n and s % 4 == 0: add(music, pluck(n + 12, bright=0.5), t0 + s * STEP, 0.8)
            add(bass, e808(root, BAR * 0.9), t0, 0.7)
        elif name == 'outro':
            add(music, pad(chord, BAR, bright=0.4), t0)
            if b == 0:
                add(drums, kick(), t0); add(bass, e808(ROOTS[0], BAR * 1.6), t0)
                add(fx, crash(), t0, 0.7)
        # half-beat of silence right before each drop makes it hit harder
        if last and name in ('build', 'build2'):
            for buf in (drums, music, fx):
                buf[at(t0 + BAR - STEP * 2): at(t0 + BAR)] *= 0.0
    t += bars * BAR

# sidechain: duck the music under every kick for the classic pump
duck = np.ones(N, np.float32)
x = tt(0.25)
curve = 1 - 0.65 * np.exp(-x / 0.07)
for k in kick_times:
    i = at(k); j = min(N, i + len(curve)); duck[i:j] = np.minimum(duck[i:j], curve[: j - i])
mix = drums * 0.9 + bass * 0.95 + music * duck * 1.1 + fx * 0.8
end = at(total_bars * BAR + 2.5)
mix = mix[:end]
fade = at(2.0); mix[-fade:] *= np.linspace(1, 0, fade) ** 2
mix = np.tanh(mix * 1.4) / np.tanh(1.4)                       # gentle glue/saturation

raw = os.path.join(HERE, 'song.raw.wav'); out = os.path.join(HERE, 'song.wav')
with wave.open(raw, 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(mix / np.abs(mix).max() * 0.95, -1, 1) * 32767).astype(np.int16).tobytes())
# Consistent loudness so the entrance is punchy without blasting the VC.
subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', raw, '-af', 'loudnorm=I=-13:TP=-1.5:LRA=9', '-ar', str(SR), '-ac', '2', out], check=True)
os.remove(raw)
print(f'{out} {len(mix) / SR:.1f}s')
