# Discord Voice Bot

A Discord voice companion for Team TEJAS. It joins a voice channel, transcribes speech with local Whisper, responds to “Tejas” (and short follow-ups), and speaks replies aloud.

## What it uses

- Discord.js and `@discordjs/voice` for Discord commands and voice audio.
- `faster-whisper` for speech recognition. The model runs on this machine, using CUDA when available and CPU otherwise. The default `large-v3-turbo` model downloads on first start.
- The Claude Code CLI (`claude -p --model haiku`) to draft replies from recent recognized speech. Install and authenticate the CLI on the machine running the bot.
- `uvx edge-tts` for speech output. It downloads the Edge TTS runner on first use and needs an internet connection.
- `ffmpeg` to convert incoming Discord audio for transcription.

## Requirements

- Node.js 22.12 or newer.
- Python 3.10 or newer.
- `ffmpeg`, `uv`, and the authenticated Claude Code CLI available on `PATH`.
- A Discord application with a bot user and a token.

CUDA is optional. The speech worker tries CUDA and falls back to CPU if it cannot initialize the GPU model.

## Setup

1. In the [Discord Developer Portal](https://discord.com/developers/applications), create a bot and invite it with the `bot` and `applications.commands` scopes. Grant it permission to view the voice channel, connect, and speak.
2. Create the local environment files and install dependencies:

   ```sh
   cp .env.example .env
   npm install
   python3 -m venv .venv
   . .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. Put the bot token in `.env` as `DISCORD_TOKEN=...`. Keep `.env` private; it is ignored by Git.
4. Start the bot:

   ```sh
   npm start
   ```

The first run downloads the Whisper model. Use `STT_MODEL` to select another faster-whisper model, or `TTS_VOICE` to change the Edge TTS voice. Defaults are `large-v3-turbo` and `en-IN-PrabhatNeural`.

## Commands

- `/join` joins your current voice channel and starts listening.
- `/leave` disconnects the bot.
- `/song` plays the vocal Team TEJAS anthem.
- `/fallback` plays the instrumental track.

The bot registers these commands in the guilds it can see when it starts. While it is connected, recognized speech is used to decide whether to reply. Say “Tejas” to start a conversation; the speaker's next line can omit the wake word for 25 seconds.

## Voice and privacy

When `/join` succeeds, the bot posts a notice in the text channel. Audio is transcribed on the machine running the bot. Recent recognized text is sent to the configured Claude Code CLI to produce replies, and Edge TTS uses an online service to synthesize those replies. Use `/leave` to stop voice capture. Do not run this bot in a channel without telling its members how it processes speech.

## Music tools

The two WAV files in the repository are the bot's default tracks. `make_song.py` can synthesize an instrumental `song.wav` with NumPy and `ffmpeg`; running it replaces the vocal track. `song/generate.py` generates vocal candidates with ACE-Step 1.5 and a local CUDA setup. That large external checkout and its model weights are intentionally not included; place the checkout at `tools/ACE-Step-1.5` before using that script. Generated candidates go to the ignored `song/out/` directory.
