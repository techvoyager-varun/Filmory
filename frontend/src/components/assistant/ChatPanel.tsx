import { useEffect, useRef } from "react";
import { Sparkles, Bot, User as UserIcon, AlertCircle, HelpCircle, Info } from "lucide-react";
import { ChatInput } from "./ChatInput";
import { ActiveFilters } from "./ActiveFilters";
import { AssistantMovieCards } from "./AssistantMovieCards";
import { EvidencePanel } from "./EvidencePanel";
import type { MessageSchema, MovieIntent } from "@/types/assistant";
import { cn } from "@/lib/utils";

interface ChatPanelProps {
  messages: MessageSchema[];
  activeIntent: MovieIntent | null;
  isLoading: boolean;
  onSendMessage: (message: string) => void;
  onFilterRemoved?: (updatedIntent: MovieIntent) => void;
  sessionId: string;
  error?: string | null;
  className?: string;
}

export function ChatPanel({
  messages,
  activeIntent,
  isLoading,
  onSendMessage,
  onFilterRemoved,
  sessionId,
  error,
  className,
}: ChatPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className={cn("flex flex-col h-[calc(100vh-12rem)] min-h-[550px]", className)}>
      {/* Active Intent / Filters Bar */}
      {activeIntent && (
        <div className="mb-4">
          <ActiveFilters intent={activeIntent} sessionId={sessionId} onFilterRemoved={onFilterRemoved} />
        </div>
      )}

      {/* Message List */}
      <div className="flex-1 overflow-y-auto space-y-6 pr-2 scrollbar-thin">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center p-6 space-y-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-tr from-gold/20 via-primary/20 to-surface-raised border border-gold/30 shadow-inner">
              <Sparkles className="h-8 w-8 text-gold animate-pulse" />
            </div>
            <div className="max-w-md space-y-2">
              <h2 className="text-xl font-bold tracking-tight text-foreground">
                How can I help you find a movie?
              </h2>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Ask me in natural language. Describe a vibe, specific genres, plot twists,
                year constraints, or reference a movie you loved. Filmory's deep recommender
                engine and AI will find the perfect match.
              </p>
            </div>
          </div>
        ) : (
          messages.map((msg, index) => {
            const isUser = msg.role === "user";

            return (
              <div
                key={index}
                className={cn(
                  "flex gap-3 text-sm leading-relaxed",
                  isUser ? "justify-end" : "justify-start",
                )}
              >
                {!isUser && (
                  <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-xl bg-gold/10 border border-gold/30 text-gold shadow-sm mt-0.5">
                    <Bot className="h-4 w-4" />
                  </div>
                )}

                <div
                  className={cn(
                    "max-w-[85%] sm:max-w-[80%] rounded-2xl px-4 py-3 shadow-sm",
                    isUser
                      ? "bg-primary text-primary-foreground rounded-tr-sm"
                      : "bg-surface-raised border border-border/80 text-foreground rounded-tl-sm space-y-3",
                  )}
                >
                  {/* Message content */}
                  <div className="whitespace-pre-wrap">{msg.content}</div>

                  {/* Informational notice / disclaimer */}
                  {msg.notice && (
                    <div className="flex items-start gap-2 rounded-lg bg-sky-500/10 border border-sky-500/20 p-2.5 text-sky-300 text-xs">
                      <Info className="h-4 w-4 shrink-0 mt-0.5" />
                      <span>{msg.notice}</span>
                    </div>
                  )}

                  {/* Clarification prompt if any */}
                  {msg.intent?.needs_clarification && msg.intent.clarification_question && (
                    <div className="flex items-start gap-2 rounded-lg bg-amber-500/10 border border-amber-500/20 p-2.5 text-amber-300 text-xs">
                      <HelpCircle className="h-4 w-4 shrink-0 mt-0.5" />
                      <span>{msg.intent.clarification_question}</span>
                    </div>
                  )}

                  {/* Recommended Movies */}
                  {msg.movies && msg.movies.length > 0 && (
                    <AssistantMovieCards
                      movies={msg.movies}
                      sessionId={sessionId}
                    />
                  )}

                  {/* Grounded Evidence Panel */}
                  {msg.evidence && msg.evidence.length > 0 && (
                    <div className="pt-2">
                      <EvidencePanel evidence={msg.evidence} />
                    </div>
                  )}

                  <div
                    className={cn(
                      "text-[10px] select-none pt-1 opacity-60",
                      isUser ? "text-primary-foreground/80 text-right" : "text-muted-foreground",
                    )}
                  >
                    {msg.timestamp}
                  </div>
                </div>

                {isUser && (
                  <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-xl bg-surface-raised border border-border text-foreground shadow-sm mt-0.5">
                    <UserIcon className="h-4 w-4" />
                  </div>
                )}
              </div>
            );
          })
        )}

        {/* Loading Bubble */}
        {isLoading && (
          <div className="flex gap-3 justify-start text-sm">
            <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-xl bg-gold/10 border border-gold/30 text-gold shadow-sm animate-pulse">
              <Bot className="h-4 w-4" />
            </div>
            <div className="rounded-2xl rounded-tl-sm bg-surface-raised border border-border/80 px-4 py-3 text-muted-foreground flex items-center gap-2">
              <span className="inline-block h-2 w-2 rounded-full bg-gold animate-bounce" />
              <span
                className="inline-block h-2 w-2 rounded-full bg-gold animate-bounce"
                style={{ animationDelay: "150ms" }}
              />
              <span
                className="inline-block h-2 w-2 rounded-full bg-gold animate-bounce"
                style={{ animationDelay: "300ms" }}
              />
              <span className="text-xs ml-1 text-muted-foreground/80">
                Finding personalized films & assembling grounded evidence...
              </span>
            </div>
          </div>
        )}

        {/* Error Notification */}
        {error && (
          <div className="flex items-center gap-2 rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-400">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="pt-4 mt-auto">
        <ChatInput onSend={onSendMessage} isLoading={isLoading} />
      </div>
    </div>
  );
}
