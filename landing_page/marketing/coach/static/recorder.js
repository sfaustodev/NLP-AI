/* MediaRecorder wrapper — Chrome / Firefox / Edge only.
 *
 * API:
 *   isSupported() / requestMic() / releaseMic()
 *   startRecording({timeoutMs, onAutoStop, onError}) → void
 *   stopRecording() → Promise<Blob>
 *   isActive() → bool
 *
 * Auto-stop semantics: when timeoutMs fires while still recording, the
 * onAutoStop callback receives the collected blob — the same data the user
 * would have gotten by clicking stop. This eliminates the "recorder not
 * active" race where the user clicks stop after the auto-stop already fired.
 */

(function () {
  "use strict";

  let stream = null;
  let recorder = null;
  let chunks = [];
  let stopTimeoutId = null;
  let stopResolve = null;
  let stopReject = null;
  let autoStopCb = null;
  let errorCb = null;

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
    // Safari detection — SPEC §10.1 marks Safari out of scope.
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

  function isActive() {
    return !!(recorder && recorder.state === "recording");
  }

  function startRecording(opts) {
    opts = opts || {};
    const timeoutMs = opts.timeoutMs || 30000;
    autoStopCb = typeof opts.onAutoStop === "function" ? opts.onAutoStop : null;
    errorCb    = typeof opts.onError    === "function" ? opts.onError    : null;
    if (!stream) {
      throw new Error("Mic stream not ready; call requestMic() first.");
    }
    if (recorder && recorder.state !== "inactive") {
      throw new Error("Recorder is already running.");
    }
    const mime = pickMime();
    recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    chunks = [];
    let stoppedByUser = false;

    recorder.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) { chunks.push(ev.data); }
    };

    recorder.onerror = (ev) => {
      const err = new Error("Recorder error: " + (ev.error && ev.error.message));
      if (stopReject) {
        stopReject(err);
        stopResolve = null; stopReject = null;
      } else if (errorCb) {
        errorCb(err);
      }
    };

    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: mime || "audio/webm" });
      if (stopResolve) {
        // User-initiated stop — resolve the awaiting promise.
        stopResolve(blob);
        stopResolve = null; stopReject = null;
      } else if (!stoppedByUser && autoStopCb) {
        // Auto-stop timer fired (or some other internal stop) — deliver the
        // blob via callback so the caller can upload it.
        try { autoStopCb(blob); } catch (e) { if (errorCb) { errorCb(e); } }
      }
      autoStopCb = null;
    };

    // Expose a hook so stopRecording can mark this stop as user-initiated.
    recorder._markStoppedByUser = () => { stoppedByUser = true; };

    recorder.start();
    stopTimeoutId = setTimeout(() => {
      if (recorder && recorder.state === "recording") {
        recorder.stop();
      }
    }, timeoutMs);
  }

  function stopRecording() {
    return new Promise((resolve, reject) => {
      if (!recorder || recorder.state === "inactive") {
        // Recorder may have already auto-stopped. The auto-stop callback
        // (if registered) has already delivered the blob — caller should
        // not be awaiting stopRecording in that case. Surface explicit.
        reject(new Error("Recorder is not active (auto-stop may have fired)."));
        return;
      }
      stopResolve = resolve;
      stopReject  = reject;
      if (stopTimeoutId) { clearTimeout(stopTimeoutId); stopTimeoutId = null; }
      if (recorder._markStoppedByUser) { recorder._markStoppedByUser(); }
      recorder.stop();
    });
  }

  window.CoachRecorder = {
    isSupported: isSupported,
    requestMic: requestMic,
    releaseMic: releaseMic,
    startRecording: startRecording,
    stopRecording: stopRecording,
    isActive: isActive,
    pickMime: pickMime,
  };
})();
