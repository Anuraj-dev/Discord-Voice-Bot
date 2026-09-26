# Persistent speech-to-text worker: keeps the Whisper model loaded.
# Protocol: one 16 kHz mono WAV path per stdin line -> one JSON line {"text": ...} on stdout.
import json, os, sys, glob, ctypes

# Preload pip-installed CUDA libs so CTranslate2 finds them without LD_LIBRARY_PATH.
for lib in sorted(glob.glob(os.path.join(sys.prefix, 'lib/python*/site-packages/nvidia/*/lib/*.so*'))):
    try: ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
    except OSError: pass

from faster_whisper import WhisperModel, decode_audio

MODEL = os.environ.get('STT_MODEL', 'large-v3-turbo')
try:
    model = WhisperModel(MODEL, device='cuda', compute_type='float16')
    warmup = os.path.join(os.path.dirname(__file__), 'models/warmup.wav')
    if os.path.isfile(warmup):
        list(model.transcribe(warmup)[0])  # fail fast if CUDA kernels are unusable
    device = 'cuda'
except Exception as e:
    print(f'CUDA unavailable ({e}); falling back to CPU int8', file=sys.stderr)
    model = WhisperModel(MODEL, device='cpu', compute_type='int8', cpu_threads=12)
    device = 'cpu'
print(json.dumps({'ready': True, 'device': device}), flush=True)

for line in sys.stdin:
    path = line.strip()
    try:
        audio = decode_audio(path)
        # The crew only speaks English/Hinglish; free detection turns short clips into Spanish, Icelandic etc.
        _, _, probs = model.detect_language(audio=audio, vad_filter=True)
        p = dict(probs)
        lang = 'en' if p.get('en', 0) >= p.get('hi', 0) else 'hi'
        # hotwords only: an initial_prompt gets echoed back verbatim on noise ("Tejas Discord voice chat.").
        segs, _ = model.transcribe(audio, language=lang, beam_size=1, vad_filter=True,
                                   condition_on_previous_text=False, hotwords='Tejas')
        text = ' '.join(s.text.strip() for s in segs)
    except Exception as e:
        print(f'stt error: {e}', file=sys.stderr)
        text = ''
    print(json.dumps({'text': text}), flush=True)
