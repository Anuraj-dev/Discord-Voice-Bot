# Generates Team TEJAS anthem candidates with ACE-Step 1.5 (local, GPU).
# Run from anywhere: tools/ACE-Step-1.5/.venv/bin/python song/generate.py [count] [seed...]
# Outputs land in song/out/; pick one and copy it to song.wav (the bot's entrance track).
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
ACE = os.path.join(HERE, '..', 'tools', 'ACE-Step-1.5')
sys.path.insert(0, ACE)
os.chdir(ACE)

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

CAPTION = (
    "High-energy desi hip hop trap anthem with Hinglish male rap vocals and a crowd chant hook. "
    "Hard-hitting distorted 808 bass, punchy trap drums, rapid hi-hat rolls, dark synth lead. "
    "Big EDM riser build-up into a heavy bass drop. Hype, swagger, stadium energy."
)

count = int(sys.argv[1]) if len(sys.argv) > 1 else 2
seeds = [int(s) for s in sys.argv[2:]] or None

dit, llm = AceStepHandler(), LLMHandler()
# 8 GB card: keep every idle component (DiT included) on CPU, else the LM phase OOMs.
print(dit.initialize_service(project_root=ACE, config_path='acestep-v15-turbo', device='cuda', offload_to_cpu=True, offload_dit_to_cpu=True))
print(llm.initialize(checkpoint_dir=os.path.join(ACE, 'checkpoints'), lm_model_path='acestep-5Hz-lm-0.6B',
                     backend='pt', device='cuda', offload_to_cpu=True))

params = GenerationParams(
    caption=CAPTION,
    lyrics=open(os.path.join(HERE, 'lyrics.txt')).read(),
    vocal_language='hi',
    bpm=140,
    keyscale='F minor',
    duration=110,
    shift=3.0,
)
out = os.path.join(HERE, 'out')
os.makedirs(out, exist_ok=True)
for i in range(count):
    cfg = GenerationConfig(batch_size=1, audio_format='wav',
                           use_random_seed=seeds is None, seeds=[seeds[i]] if seeds and i < len(seeds) else None)
    res = generate_music(dit, llm, params, cfg, save_dir=out)
    if not res.success:
        print('FAILED', res.error); continue
    for a in res.audios:
        print(json.dumps({'path': a['path'], 'seed': a['params'].get('seed')}), flush=True)
