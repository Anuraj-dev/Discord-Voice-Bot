// Tejas voice bot: listens in a Discord VC, transcribes locally with faster-whisper (stt.py),
// thinks with the host's authenticated `claude -p` CLI, speaks via edge-tts.
import { Client, GatewayIntentBits, Events, MessageFlags } from 'discord.js';
import {
  joinVoiceChannel, getVoiceConnection, createAudioPlayer, createAudioResource,
  EndBehaviorType, AudioPlayerStatus, VoiceConnectionStatus, entersState,
} from '@discordjs/voice';
import prism from 'prism-media';
import { spawn } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { createInterface } from 'node:readline';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const TOKEN = process.env.DISCORD_TOKEN;
const VOICE = process.env.TTS_VOICE || 'en-IN-PrabhatNeural';
const WAKE = /\b(te+\s*ja+s|tejus|tejaz|teja|claude)\b|[तटथ]े\s*ज़?स/i;
const FOLLOW_UP_MS = 25_000; // after Tejas replies to someone, their next line needs no wake word
const SONG = join(HERE, 'song.wav');
const FALLBACK = join(HERE, 'song-instrumental.wav');
const TMP = mkdtempSync(join(tmpdir(), 'tejas-'));

const SYSTEM = `You are Tejas, an AI voice buddy hanging out in the Team TEJAS Discord voice channel.
Team TEJAS is a crew of students and builders (IITM BS degree courses like Java, PDSA, MLT, MLP, MAD 2), who ship side projects, share AI news, memes, anime and games.
Everything you write is spoken aloud by text-to-speech, so:
- Reply in 1-3 short conversational sentences. Never use markdown, lists, emojis, code or URLs.
- Write your replies in English or romanized Hinglish (never Devanagari script), matching the speaker's vibe.
- Lines come from speech recognition: Hindi may appear in Devanagari and words may be garbled. That's the recognizer, not the speaker, so never comment on anyone's language, script or clarity; guess sensibly or briefly ask them to repeat.
- Be warm, witty and useful. Address people by name. You can't browse or run tools here.`;

if (!TOKEN) { console.error('Set DISCORD_TOKEN'); process.exit(1); }

const client = new Client({ intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates] });
const sessions = new Map(); // guildId -> { player, history, listening:Set, lastReplyTo, busy, queue }

function run(cmd, args, input) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { stdio: ['pipe', 'pipe', 'pipe'] });
    let out = '', err = '';
    p.stdout.on('data', d => (out += d));
    p.stderr.on('data', d => (err += d));
    p.on('error', reject);
    p.on('close', code => (code === 0 ? resolve(out) : reject(new Error(`${cmd} exited ${code}: ${err.slice(-400)}`))));
    if (input !== undefined) p.stdin.end(input); else p.stdin.end();
  });
}

// stt.py keeps Whisper loaded on the GPU; requests are serialized by drain(), so replies match FIFO.
const stt = spawn(join(HERE, '.venv/bin/python'), [join(HERE, 'stt.py')], { stdio: ['pipe', 'pipe', 'inherit'] });
const sttLines = createInterface({ input: stt.stdout });
const sttWaiters = [];
sttLines.on('line', l => {
  const msg = JSON.parse(l);
  if (msg.ready) return console.log(`STT ready on ${msg.device}`);
  sttWaiters.shift()?.(msg.text);
});
stt.on('exit', code => { console.error(`stt.py exited ${code}`); process.exit(1); });

async function transcribe(pcm) {
  const base = join(TMP, `u${Date.now()}${Math.random().toString(36).slice(2, 6)}`);
  writeFileSync(`${base}.pcm`, pcm);
  try {
    await run('ffmpeg', ['-loglevel', 'error', '-f', 's16le', '-ar', '48000', '-ac', '2', '-i', `${base}.pcm`, '-ar', '16000', '-ac', '1', '-y', `${base}.wav`]);
    const out = await new Promise(resolve => { sttWaiters.push(resolve); stt.stdin.write(`${base}.wav\n`); });
    const text = out.replace(/\[[^\]]*\]|\([^)]*\)/g, ' ').replace(/\s+/g, ' ').trim();
    // whisper hallucinates these on noise/breath
    if (!text || /^(thank you\.?|thanks for watching!?|you|bye\.?|\.+)$/i.test(text)) return '';
    return text;
  } finally {
    rmSync(`${base}.pcm`, { force: true }); rmSync(`${base}.wav`, { force: true });
  }
}

async function think(history) {
  const transcript = history.map(h => `${h.who}: ${h.text}`).join('\n');
  const out = await run('claude', ['-p', '--model', 'haiku', '--tools', '', '--strict-mcp-config', '--setting-sources', '',
    '--no-session-persistence', '--system-prompt', SYSTEM],
    `Recent voice chat (latest last):\n${transcript}\n\nReply as Tejas to the latest message.`);
  return out.replace(/[*_#`>]/g, '').trim();
}

async function speak(s, text) {
  const file = join(TMP, `say${Date.now()}.mp3`);
  await run('uvx', ['edge-tts', '--voice', VOICE, '--text', text, '--write-media', file]);
  // Don't cut off the entrance song (or anything else still playing).
  if (s.player.state.status !== AudioPlayerStatus.Idle) await new Promise(r => s.player.once(AudioPlayerStatus.Idle, r));
  s.player.play(createAudioResource(file));
  await entersState(s.player, AudioPlayerStatus.Playing, 5_000).catch(() => {});
  await new Promise(r => s.player.once(AudioPlayerStatus.Idle, r));
  rmSync(file, { force: true });
}

// One utterance at a time per guild so replies stay in order and the GPU isn't contended.
async function drain(guild, s) {
  if (s.busy) return;
  s.busy = true;
  while (s.queue.length) {
    const { userId, pcm } = s.queue.shift();
    try {
      const text = await transcribe(pcm);
      if (!text) continue;
      const member = await guild.members.fetch(userId).catch(() => null);
      const who = member?.displayName || 'someone';
      s.history.push({ who, text });
      s.history.splice(0, Math.max(0, s.history.length - 30));
      console.log(`[heard] ${who}: ${text}`);
      // Follow-ups skip the wake word, but one- or two-word lines ("Yeah.", "Hey!") are chatter, not a question.
      const followUp = s.lastReplyTo?.id === userId && Date.now() - s.lastReplyTo.at < FOLLOW_UP_MS && text.split(/\s+/).length >= 3;
      const addressed = WAKE.test(text) || followUp;
      if (!addressed) continue;
      const reply = await think(s.history);
      if (!reply) continue;
      s.history.push({ who: 'Tejas', text: reply });
      console.log(`[said] ${reply}`);
      await speak(s, reply);
      s.lastReplyTo = { id: userId, at: Date.now() };
    } catch (e) {
      console.error('[error]', e.message);
    }
  }
  s.busy = false;
}

function listen(guild, connection, s) {
  connection.receiver.speaking.on('start', userId => {
    if (s.listening.has(userId)) return;
    s.listening.add(userId);
    const opus = connection.receiver.subscribe(userId, { end: { behavior: EndBehaviorType.AfterSilence, duration: 900 } });
    const chunks = [];
    const pcm = opus.pipe(new prism.opus.Decoder({ rate: 48000, channels: 2, frameSize: 960 }));
    pcm.on('data', c => chunks.push(c));
    pcm.on('end', () => {
      s.listening.delete(userId);
      const buf = Buffer.concat(chunks);
      if (buf.length < 48000 * 2 * 2 * 0.6) return; // skip blips under 0.6s
      s.queue.push({ userId, pcm: buf });
      drain(guild, s);
    });
    pcm.on('error', () => s.listening.delete(userId));
  });
}

client.once(Events.ClientReady, async c => {
  console.log(`Logged in as ${c.user.tag}`);
  for (const g of c.guilds.cache.values()) {
    await g.commands.set([
      { name: 'join', description: 'Tejas joins your voice channel' },
      { name: 'leave', description: 'Tejas leaves the voice channel' },
      { name: 'song', description: 'Tejas plays the Team TEJAS anthem' },
      { name: 'fallback', description: 'Tejas plays the instrumental fallback music' },
    ]);
    console.log(`Commands registered in ${g.name}`);
  }
});

client.on(Events.InteractionCreate, async i => {
  if (!i.isChatInputCommand() || !i.guild) return;
  if (i.commandName === 'leave') {
    getVoiceConnection(i.guildId)?.destroy();
    sessions.delete(i.guildId);
    return i.reply('👋 Tejas left the VC.');
  }
  if (i.commandName === 'song') {
    const s = sessions.get(i.guildId);
    if (!s) return i.reply({ content: 'Use /join first.', flags: MessageFlags.Ephemeral });
    s.player.play(createAudioResource(SONG));
    return i.reply('🔥 Team TEJAS anthem, coming up.');
  }
  if (i.commandName === 'fallback') {
    const s = sessions.get(i.guildId);
    if (!s) return i.reply({ content: 'Use /join first.', flags: MessageFlags.Ephemeral });
    s.player.play(createAudioResource(FALLBACK));
    return i.reply('🎹 Instrumental fallback, coming up.');
  }
  const channel = i.member?.voice?.channel;
  if (!channel) return i.reply({ content: 'Join a voice channel first.', flags: MessageFlags.Ephemeral });
  await i.deferReply();
  console.log(`[join] ${i.user.username} -> ${channel.name}`);
  // A second /join would stack another speaking listener on the same connection and double every reply.
  getVoiceConnection(i.guildId)?.destroy();
  sessions.get(i.guildId)?.player.stop(true);
  sessions.delete(i.guildId);
  const connection = joinVoiceChannel({
    channelId: channel.id, guildId: i.guildId, adapterCreator: i.guild.voiceAdapterCreator, selfDeaf: false, selfMute: false,
  });
  try {
    await entersState(connection, VoiceConnectionStatus.Ready, 20_000);
  } catch {
    connection.destroy();
    return i.editReply('Could not connect to voice.');
  }
  const s = { player: createAudioPlayer(), history: [], listening: new Set(), lastReplyTo: null, busy: false, queue: [] };
  connection.subscribe(s.player);
  sessions.set(i.guildId, s);
  listen(i.guild, connection, s);
  connection.on(VoiceConnectionStatus.Disconnected, () => { connection.destroy(); sessions.delete(i.guildId); });
  await i.editReply(`🎙️ **Tejas joined ${channel.name}.** Heads up: while I'm here, speech in this VC is transcribed on the machine running the bot so I can reply. Say "Tejas" to talk to me; \`/leave\` to kick me.`);
  s.player.play(createAudioResource(SONG)); // entrance song (song/generate.py, fallback make_song.py)
  // speak() waits for the song to finish, then Tejas greets in its normal voice.
  speak(s, 'Arre hello Team Tejas! Main aa gaya. Just say Tejas when you want me.').catch(e => console.error('[error]', e.message));
});

process.on('SIGINT', () => { stt.kill(); rmSync(TMP, { recursive: true, force: true }); process.exit(0); });
client.login(TOKEN);
