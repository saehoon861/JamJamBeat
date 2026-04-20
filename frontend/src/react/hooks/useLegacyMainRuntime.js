import { useEffect, useRef } from "react";

export function useLegacyMainRuntime() {
  const bootedRef = useRef(false);

  useEffect(() => {
    if (bootedRef.current) return;
    bootedRef.current = true;

    let cancelled = false;

    import("../../js/main.js").catch((error) => {
      if (cancelled) return;
      console.error("Failed to boot legacy main runtime:", error);
    });

    return () => {
      cancelled = true;
    };
  }, []);
}
