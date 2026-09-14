import { useState, useRef, useEffect } from "react";
import { Send, Sparkles, Loader2, CornerDownLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading: boolean;
  placeholder?: string;
  className?: string;
}

const QUICK_PROMPTS = [
  "Mind-bending sci-fi under 2 hours",
  "Feel-good 90s romantic comedy",
  "High-tension thriller like Inception",
  "Visually stunning animation with high rating",
  "Dark mystery with an unexpected twist",
];

export function ChatInput({
  onSend,
  isLoading,
  placeholder = "Ask Filmory... e.g. 'Show me a psychological thriller from the 90s without horror'",
  className,
}: ChatInputProps) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!isLoading && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [isLoading]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed || isLoading) return;
    onSend(trimmed);
    setText("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handlePromptClick = (prompt: string) => {
    if (isLoading) return;
    onSend(prompt);
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    // Auto-adjust height up to max 160px
    const target = e.target;
    target.style.height = "auto";
    target.style.height = `${Math.min(target.scrollHeight, 160)}px`;
  };

  return (
    <div className={cn("w-full space-y-3", className)}>
      {/* Quick Prompt Chips */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-none text-xs">
        <span className="flex items-center gap-1 text-muted-foreground whitespace-nowrap pl-1">
          <Sparkles className="h-3 w-3 text-gold" />
          <span>Try:</span>
        </span>
        {QUICK_PROMPTS.map((prompt) => (
          <button
            key={prompt}
            type="button"
            disabled={isLoading}
            onClick={() => handlePromptClick(prompt)}
            className="shrink-0 rounded-full border border-border/80 bg-surface/70 px-3 py-1 text-xs text-muted-foreground transition hover:border-gold/50 hover:bg-surface hover:text-foreground disabled:opacity-50"
          >
            {prompt}
          </button>
        ))}
      </div>

      {/* Input Box */}
      <form
        onSubmit={handleSubmit}
        className="relative flex items-end rounded-2xl border border-border bg-surface/90 p-2 shadow-lg backdrop-blur-xl focus-within:border-gold/60 focus-within:ring-1 focus-within:ring-gold/30 transition-all"
      >
        <textarea
          ref={textareaRef}
          value={text}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isLoading}
          rows={1}
          className="w-full resize-none bg-transparent px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none max-h-40 min-h-[44px]"
        />

        <div className="flex items-center gap-1.5 pb-1 pr-1">
          <Button
            type="submit"
            size="sm"
            disabled={!text.trim() || isLoading}
            className="h-9 w-9 rounded-xl p-0 bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-40"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
            <span className="sr-only">Send</span>
          </Button>
        </div>
      </form>
      <div className="flex items-center justify-between px-2 text-[11px] text-muted-foreground/60">
        <span>Powered by Gemini 2.0 & Filmory DAMR Recommender</span>
        <span className="hidden sm:inline-flex items-center gap-1">
          Press <CornerDownLeft className="h-2.5 w-2.5" /> to send, Shift + Enter for newline
        </span>
      </div>
    </div>
  );
}
