"use client";

import { motion } from "motion/react";
import * as React from "react";
import { cn } from "@/lib/utils";

interface TypewriterTextProps {
  /** The text to type out once. */
  text: string;
  className?: string;
  cursorClassName?: string;
  /** Milliseconds between each character. Default 80 (slow). */
  typingSpeed?: number;
}

const TypewriterText = React.forwardRef<HTMLSpanElement, TypewriterTextProps>(
  ({ text, className, cursorClassName, typingSpeed = 80 }, ref) => {
    const [displayed, setDisplayed] = React.useState("");
    const [done, setDone] = React.useState(false);

    React.useEffect(() => {
      // Reset on remount (e.g. mode switch or new chat)
      setDisplayed("");
      setDone(false);

      let i = 0;
      function tick() {
        i += 1;
        setDisplayed(text.slice(0, i));
        if (i < text.length) {
          timeoutRef = setTimeout(tick, typingSpeed);
        } else {
          setDone(true);
        }
      }

      let timeoutRef: ReturnType<typeof setTimeout> = setTimeout(tick, typingSpeed);
      return () => clearTimeout(timeoutRef);
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [text, typingSpeed]);

    return (
      <span ref={ref} className={cn("inline-block", className)}>
        {displayed}
        <motion.span
          animate={done ? { opacity: 0 } : { opacity: [1, 0] }}
          transition={
            done
              ? { duration: 0.8, delay: 0.4, ease: "easeOut" }
              : { duration: 0.5, repeat: Infinity, repeatType: "reverse" }
          }
          className={cn(
            "ml-0.5 inline-block h-[1em] w-0.5 bg-current align-middle",
            cursorClassName,
          )}
        />
      </span>
    );
  },
);

TypewriterText.displayName = "TypewriterText";

export { TypewriterText };
