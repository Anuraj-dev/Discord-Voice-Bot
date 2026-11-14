<h1 align="center">Discord Voice Bot</h1>

<p align="center">
  A Discord voice bot for Team TEJAS that transcribes voice chat on your own machine.<br />
  Say "Tejas" in a voice channel and it answers out loud.
</p>

<p align="center">
  <a href="#running-it">Running it</a>
  ·
  <a href="#commands">Commands</a>
  ·
  <a href="#voice-and-privacy">Voice and privacy</a>
</p>

The bot joins a Discord voice channel and listens to everyone in it. faster-whisper transcribes each speaker on the machine that runs the bot. When someone says the name, the Claude Code CLI drafts a reply from the last 30 lines of chat, and Edge TTS speaks it back into the channel.

It runs as one Node.js process that starts a Python speech worker. You need a Discord bot token, `ffmpeg` and `uv` on your `PATH`, and the Claude Code CLI logged in on the same machine. A CUDA GPU is optional and used when it works.

## What you get

- **Speech stays on the host.** Each clip goes to a faster-whisper model on this machine. The model loads on CUDA in float16 and falls back to CPU int8 when CUDA fails to initialize.
- **English or Hindi, picked per clip.** `stt.py` compares the English and Hindi probabilities for each clip and transcribes in the higher one. The name "Tejas" is passed as a hotword to bias recognition toward it.
- **Wake word, then a short follow-up.** Saying "Tejas" or a close spelling such as "tejus" or "teja" starts a reply. The bot also answers to "Claude". After it replies to someone, that person's next line of three or more words gets a reply for 25 seconds without the wake word.
- **Short spoken replies.** The Claude Code CLI runs on the haiku model with no tools and no MCP servers. Its system prompt asks for one to three short sentences with no markdown, lists or URLs, because text to speech reads the reply aloud.
- **Clips end at silence.** A speaker's clip closes after 900 ms of silence. Clips shorter than 0.6 seconds are dropped before they reach Whisper.
- **One reply at a time.** The bot handles one clip at a time per server, in arrival order. It waits for the entrance song or the previous reply to finish before it speaks.
- **Noise filter.** Bracketed annotations such as `[Music]` are stripped from the transcript. Empty results and common Whisper hallucinations such as "thank you" or "bye" are discarded.

## Running it

Run the bot from a checkout of this repository on Linux or macOS.

### Requirements

- Node.js 22.12 or newer. `@discordjs/voice` requires it, as the lockfile records.
- Python 3 with `venv`.
- `ffmpeg` and `uv` on `PATH`. The bot runs `ffmpeg` to convert audio and `uvx edge-tts` to speak.
- The Claude Code CLI on `PATH`, installed and logged in as the user that runs the bot.
- A Discord application with a bot user.
- An NVIDIA GPU with CUDA, which is not required.

### Create the Discord bot

1. In the [Discord Developer Portal](https://discord.com/developers/applications), create an application and add a bot user. Copy the bot token.
2. Invite the bot with the `bot` and `applications.commands` scopes. Grant View Channel, Connect and Speak on the voice channels it should use.
3. No privileged intents are needed. The bot requests only the Guilds and Guild Voice States intents.

### Install and start

```sh
cp .env.example .env
npm install
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Set `DISCORD_TOKEN` in `.env` to the bot token. Git ignores `.env`, so keep it out of commits.

```sh
npm start
```

The bot starts the speech worker with `.venv/bin/python`. Create the virtual environment as `.venv` in the repository root. Windows puts the interpreter under `.venv\Scripts`, which the bot does not look for.

The first start downloads the Whisper model. Wait for `STT ready on cuda` or `STT ready on cpu` in the log before you test a channel.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DISCORD_TOKEN` | none, required | Bot token. The bot exits at startup without it. |
| `STT_MODEL` | `large-v3-turbo` | faster-whisper model for the speech worker. Downloaded on first use. |
| `TTS_VOICE` | `en-IN-PrabhatNeural` | Edge TTS voice for replies and the greeting. |

## Commands

- `/join` joins your current voice channel and starts listening. It plays `song.wav` as an entrance track, then Tejas greets the channel. Running `/join` again moves the bot to your channel.
- `/leave` disconnects the bot and clears its chat history.
- `/song` plays `song.wav`. Run `/join` first.
- `/fallback` plays `song-instrumental.wav`. Run `/join` first.

The bot registers these commands in each server it can see at startup. A server you invite it to later needs a restart before the commands appear.

## Voice and privacy

When `/join` succeeds, the bot posts a notice in the text channel where you ran the command. The notice says that speech is transcribed on the bot's machine.

- Incoming audio is transcribed on the bot's machine. The bot does not send audio to a speech service.
- The last 30 recognized lines go to the Claude Code CLI, which sends them to the model provider behind your login.
- Reply text goes to an online Edge TTS service to produce speech.
- Chat history lives in memory only. `/leave`, a disconnect or a restart clears it.
- `/leave` stops voice capture.

Do not run this bot in a channel without telling its members how it processes speech.

## Music

`song.wav` is the entrance and `/song` track. `song-instrumental.wav` is the `/fallback` track. Both ship with the repository.

`make_song.py` synthesizes a new instrumental `song.wav` with NumPy and `ffmpeg`. It uses no samples and no speech synthesis. It builds a 140 BPM track in F minor and overwrites `song.wav` when it runs.

```sh
.venv/bin/python make_song.py
```

`song/generate.py` generates sung candidates with ACE-Step 1.5, using the lyrics in `song/lyrics.txt`. It runs on a local CUDA GPU and is tuned for an 8 GB card. The ACE-Step checkout and its model weights are not in this repository. Place the checkout at `tools/ACE-Step-1.5` first.

```sh
tools/ACE-Step-1.5/.venv/bin/python song/generate.py 2
```

Candidates go to `song/out/`, which Git ignores. The script prints one JSON line per candidate with its path and seed. Copy the one you pick over `song.wav`.

## License

MIT. See [LICENSE](LICENSE).
