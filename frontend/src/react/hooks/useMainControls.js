import { useEffect, useState } from "react";

const DEFAULT_SOUND_LABEL = "소리 켜기";

export function useMainControls() {
  const [soundButtonLabel, setSoundButtonLabel] = useState(DEFAULT_SOUND_LABEL);

  useEffect(() => {
    const onAudioState = (event) => {
      const running = Boolean(event.detail?.running);
      setSoundButtonLabel(running ? "소리 끄기" : "소리 켜기");
    };

    window.addEventListener("jamjam:audio-state", onAudioState);
    return () => {
      window.removeEventListener("jamjam:audio-state", onAudioState);
    };
  }, []);

  const requestStart = () => {
    window.dispatchEvent(new CustomEvent("jamjam:start-request"));
  };

  const requestSoundToggle = () => {
    window.dispatchEvent(new CustomEvent("jamjam:sound-toggle-request"));
  };

  return {
    soundButtonLabel,
    requestStart,
    requestSoundToggle
  };
}
