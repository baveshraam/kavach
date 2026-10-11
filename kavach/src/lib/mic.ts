/**
 * How the browser records the voice, shared by the Studio and the Unlock login so enrolment and login hear the same
 * thing. The browser's defaults (echo cancellation, noise suppression, automatic gain) are built for calls and can
 * make a voice sound robotic; the voiceprint wants the voice as it is. Opus at 96 kbps keeps the full band.
 */
export const MIC_CONSTRAINTS: MediaStreamConstraints = {
  audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
};

export const RECORDER_OPTIONS: MediaRecorderOptions = { mimeType: 'audio/webm;codecs=opus', audioBitsPerSecond: 96_000 };
