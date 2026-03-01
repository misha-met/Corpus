"use client";

import { useCallback, useEffect, useRef } from "react";
import { MicIcon } from "lucide-react";
import { useSpeechToText } from "@/hooks/useSpeechToText";
import { cn } from "@/lib/utils";

interface SpeechToTextButtonProps {
  /** Ref to the input element — used to append transcript text at cursor pos */
  inputRef: React.RefObject<HTMLInputElement | null>;
  /** Current input value */
  value: string;
  /** Setter for the input value */
  onChange: (value: string) => void;
  /** Disable the button (e.g. while streaming) */
  disabled?: boolean;
}

/**
 * A microphone button that records audio and appends transcribed text into
 * the controlled input identified by `inputRef`.
 */
export function SpeechToTextButton({
  inputRef,
  onChange,
  disabled = false,
}: SpeechToTextButtonProps) {
  const lastInsertedRef = useRef("");

  const applyTranscript = useCallback(
    (chunk: string, _isFinal: boolean) => {
      const normalized = chunk.trim();
      if (!normalized) return;
      const el = inputRef.current;
      const prev = el?.value ?? "";
      const start = el?.selectionStart ?? prev.length;
      const end = el?.selectionEnd ?? prev.length;
      const effectiveStart = Math.max(
        0,
        start - lastInsertedRef.current.length,
      );
      const before = prev.slice(0, effectiveStart);
      const after = prev.slice(end);
      const sep =
        before.length > 0 && !before.endsWith(" ") ? " " : "";
      const inserted = sep + normalized;
      lastInsertedRef.current = inserted;
      const next = before + inserted + after;
      onChange(next);
      requestAnimationFrame(() => {
        el?.setSelectionRange(
          before.length + inserted.length,
          before.length + inserted.length,
        );
        el?.focus();
      });
    },
    [inputRef, onChange],
  );

  const { isListening, toggle, stop } = useSpeechToText({
    onTranscript: applyTranscript,
    onPermissionDenied: () => {},
    onNoSpeech: () => {},
    onError: () => {},
  });

  // Stop mic when the button becomes disabled (e.g. streaming starts)
  useEffect(() => {
    if (disabled && isListening) stop();
  }, [disabled, isListening, stop]);

  return (
    <button
      type="button"
      aria-label={isListening ? "Stop listening" : "Voice input"}
      disabled={disabled}
      onClick={() => {
        if (!disabled) toggle();
      }}
      className={cn(
        "w-10 h-10 flex items-center justify-center rounded-full transition-colors shrink-0",
        isListening
          ? "bg-red-600 hover:bg-red-700"
          : "bg-gray-700 hover:bg-gray-600",
        disabled && "opacity-50 cursor-not-allowed",
      )}
      title={isListening ? "Stop listening" : "Voice input"}
    >
      <MicIcon
        className={cn(
          "w-4 h-4 text-white",
          isListening && "animate-pulse",
        )}
      />
    </button>
  );
}
