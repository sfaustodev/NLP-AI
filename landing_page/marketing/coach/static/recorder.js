/* MediaRecorder wrapper — Chrome / Firefox / Edge only.
 *
 * Exposes window.CoachRecorder.{isSupported, requestMic, startRecording,
 * stopRecording}. startRecording returns void; stopRecording returns a
 * Promise<Blob>. Auto-stop timeout configurable per call.
 */

(function () {
  "use strict";

  let stream = null;
  let recorder = null;
  let chunks = [];
  let stopTimeoutId = null;
  let stopResolve = null;
  let stopReject = null;

  function pickMime() {
    if (!window.MediaRecorder || !MediaRecorder.isTypeSupported) { return null; }
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/mp4",
    ];
    for (const c of candidates) {
      if (MediaRecorder.isTypeSupported(c)) { return c; }
    }
    return null;
  }

  function isSupported() {
    if (typeof navigator === "undefined" || !navigator.mediaDevices) { return false; }
    if (typeof navigator.mediaDevices.getUserMedia !== "function") { return false; }
    if (typeof MediaRecorder === "undefined") { return false; }
    if (!pickMime()) { return false; }
    // Quick Safari detection — SPEC §10.1 marks Safari out of scope.
    const ua = navigator.userAgent || "";
    const isSafari = /^((?!chrome|android|crios|fxios).)*safari/i.test(ua);
    if (isSafari) { return false; }
    return true;
  }

  async function requestMic() {
    if (stream) { return stream; }
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    return stream;
  }

  function releaseMic() {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
  }

  function startRecording(opts) {
    opts = opts || {};
    const timeoutMs = opts.timeoutMs || 30000;
    if (!stream) {
      throw new Error("Mic stream not ready; call requestMic() first.");
    }
    if (recorder && recorder.state !== "inactive") {
      throw new Error("Recorder is already running.");
    }
    const mime = pickMime();
    recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    chunks = [];
    recorder.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) { chunks.push(ev.data); }
    };
    recorder.onerror = (ev) => {
      if (stopReject) {
        stopReject(new Error("Recorder error: " + (ev.error && ev.error.message)));
        stopResolve = null; stopReject = null;
      }
    };
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: mime || "audio/webm" });
      if (stopResolve) {
        stopResolve(blob);
        stopResolve = null; stopReject = null;
      }
    };
    recorder.start();
    // Hard timeout — auto-stop in case user forgets.
    stopTimeoutId = setTimeout(() => {
      if (recorder && recorder.state === "recording") {
        recorder.stop();
      }
    }, timeoutMs);
  }

  function stopRecording() {
    return new Promise((resolve, reject) => {
      if (!recorder || recorder.state === "inactive") {
        reject(new Error("Recorder is not active."));
        return;
      }
      stopResolve = resolve;
      stopReject = reject;
      if (stopTimeoutId) { clearTimeout(stopTimeoutId); stopTimeoutId = null; }
      recorder.stop();
    });
  }

  window.CoachRecorder = {
    isSupported: isSupported,
    requestMic: requestMic,
    releaseMic: releaseMic,
    startRecording: startRecording,
    stopRecording: stopRecording,
    pickMime: pickMime,
  };
})();
