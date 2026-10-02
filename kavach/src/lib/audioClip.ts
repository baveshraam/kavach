/**
 * Cut a segment out of a stored recording, in the browser, and return it as a
 * fresh 16 kHz mono WAV.
 *
 * Used by the demo panel to stage attacks from audio already in the corpus.
 * The re-encode matters: a byte-identical file is caught by the duplicate
 * detector before any model runs, which is the right outcome for a *replay*
 * but would hide what the voiceprint does with an impostor's voice.
 */
export async function trimToWav(url: string, startSec: number, durationSec: number): Promise<Blob> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not fetch the stored recording (${response.status}).`);
  const encoded = await response.arrayBuffer();

  const ctx = new AudioContext();
  const decoded = await ctx.decodeAudioData(encoded);
  await ctx.close();

  const start = Math.min(startSec, Math.max(0, decoded.duration - 1));
  const length = Math.min(durationSec, decoded.duration - start);
  const targetRate = 16000;
  const offline = new OfflineAudioContext(1, Math.ceil(length * targetRate), targetRate);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start(0, start, length);
  const rendered = await offline.startRendering();
  return encodeWav(rendered.getChannelData(0), targetRate);
}

/** Fetch a stored recording unchanged -- a literal replay. */
export async function fetchExact(url: string): Promise<Blob> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not fetch the stored recording (${response.status}).`);
  return response.blob();
}

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset: number, s: string) => { for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i)); };
  write(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  write(8, 'WAVE');
  write(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);            // PCM
  view.setUint16(22, 1, true);            // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}
