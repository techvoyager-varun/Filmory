import { useEffect, useRef, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { PlusCircle, MessageSquare, Trash2, PanelLeft, Film, X } from "lucide-react";
import { RequireAuth } from "@/components/RequireAuth";
import { ChatPanel } from "@/components/assistant/ChatPanel";
import { Button } from "@/components/ui/button";
import { listSessions, getSession, deleteSession, sendMessage } from "@/api/assistant";
import type { MessageSchema, MovieIntent, SessionSummary } from "@/types/assistant";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/ask-filmory")({
  head: () => ({
    meta: [
      { title: "Ask Filmory — AI Movie Assistant" },
      {
        name: "description",
        content:
          "Find movies with conversational search, catalogue filters, and personalized recommendations.",
      },
      { property: "og:title", content: "Ask Filmory — AI Movie Assistant" },
      {
        property: "og:description",
        content:
          "Find any movie matching your exact vibe, mood, and constraints with conversational AI.",
      },
    ],
  }),
  component: () => (
    <RequireAuth>
      <AskFilmoryPage />
    </RequireAuth>
  ),
});

function AskFilmoryPage() {
  const [sessionId, setSessionId] = useState<string>("");
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [messages, setMessages] = useState<MessageSchema[]>([]);
  const [activeIntent, setActiveIntent] = useState<MovieIntent | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const requestRevision = useRef(0);
  const activeSession = useRef(sessionId);
  activeSession.current = sessionId;

  useEffect(() => {
    if (!sidebarOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSidebarOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [sidebarOpen]);

  // Load session list on mount
  useEffect(() => {
    fetchSessions();
    return () => {
      requestRevision.current += 1;
    };
  }, []);

  const fetchSessions = async () => {
    try {
      const list = await listSessions();
      setSessions(list);
    } catch (err) {
      console.warn("Could not fetch assistant sessions:", err);
    }
  };

  const handleStartNewChat = () => {
    requestRevision.current += 1;
    setIsLoading(false);
    setSessionId("");
    setMessages([]);
    setActiveIntent(null);
    setError(null);
    setSidebarOpen(false);
  };

  const handleSelectSession = async (id: string) => {
    const revision = ++requestRevision.current;
    try {
      setIsLoading(true);
      setError(null);
      const detail = await getSession(id);
      if (revision !== requestRevision.current) return;
      setSessionId(detail.session_id);
      setMessages(detail.messages || []);
      setActiveIntent(detail.active_intent || null);
      setSidebarOpen(false);
    } catch (err: unknown) {
      if (revision !== requestRevision.current) return;
      setError(err instanceof Error ? err.message : "Failed to load session");
    } finally {
      if (revision === requestRevision.current) setIsLoading(false);
    }
  };

  const handleDeleteSession = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      await deleteSession(id);
      setSessions((prev) => prev.filter((s) => s.session_id !== id));
      if (activeSession.current === id) {
        handleStartNewChat();
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to delete session");
    }
  };

  const handleSendMessage = async (text: string) => {
    if (!text.trim() || isLoading) return;
    const revision = ++requestRevision.current;

    const userMessage: MessageSchema = {
      role: "user",
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      const response = await sendMessage({
        message: text,
        session_id: sessionId || null,
      });
      if (revision !== requestRevision.current) {
        void fetchSessions();
        return;
      }

      setSessionId(response.session_id);
      setActiveIntent(response.intent);

      const assistantMessage: MessageSchema = {
        role: "assistant",
        content: response.message,
        movies: response.movies,
        intent: response.intent,
        evidence: response.evidence,
        structured_explanation: response.structured_explanation,
        notice: response.notice,
        request_mode: response.request_mode,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      // Refresh sessions in background to update previews
      fetchSessions();
    } catch (err: unknown) {
      if (revision !== requestRevision.current) return;
      setError(
        err instanceof Error
          ? err.message
          : "Failed to get response from AI assistant. Please try again.",
      );
    } finally {
      if (revision === requestRevision.current) setIsLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-[1600px] px-4 pb-12 pt-20 md:px-10 md:pt-24">
      {/* Top Header */}
      <div className="flex items-center justify-between border-b border-border/60 pb-4 mb-4">
        <div className="flex items-center gap-3">
          <Button
            size="icon"
            variant="outline"
            aria-label={sidebarOpen ? "Close conversations" : "Open conversations"}
            aria-expanded={sidebarOpen}
            onClick={() => setSidebarOpen((prev) => !prev)}
            className="md:hidden h-9 w-9 rounded-xl border-border"
          >
            <PanelLeft className="h-4 w-4" />
          </Button>
          <div className="flex items-center gap-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gold/10 border border-gold/30 text-gold shadow-sm">
              <Film className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight md:text-2xl text-foreground">
                  Ask Filmory
                </h1>
                <span className="inline-flex items-center gap-1.5 rounded-full bg-gold/10 border border-gold/30 px-2.5 py-0.5 text-[10px] font-semibold text-gold">
                  <span className="h-1.5 w-1.5 rounded-full bg-gold" />
                  Personalized discovery
                </span>
              </div>
              <p className="text-xs text-muted-foreground">
                Grounded conversational discovery with deep personalization
              </p>
            </div>
          </div>
        </div>

        <Button
          onClick={handleStartNewChat}
          aria-label="Start a new chat"
          size="sm"
          className="rounded-xl gap-1.5 bg-primary text-primary-foreground hover:bg-primary/90 text-xs shadow"
        >
          <PlusCircle className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">New Chat</span>
        </Button>
      </div>

      {/* Main Grid Layout: Sessions Sidebar + Chat Area */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 items-start">
        {sidebarOpen ? (
          <button
            type="button"
            aria-label="Dismiss conversation drawer"
            onClick={() => setSidebarOpen(false)}
            className="fixed inset-0 z-40 bg-black/60 md:hidden"
          />
        ) : null}
        {/* Sessions Sidebar (Desktop) */}
        <aside
          className={cn(
            "fixed inset-y-0 left-0 z-50 w-72 bg-background/95 p-4 backdrop-blur-xl border-r border-border transition-transform md:relative md:inset-auto md:z-0 md:w-full md:border-r-0 md:bg-transparent md:p-0 md:translate-x-0",
            sidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0",
          )}
        >
          <div className="rounded-2xl border border-border/80 bg-surface/60 p-3.5 backdrop-blur-md">
            <div className="flex items-center justify-between pb-3 border-b border-border/60">
              <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                <MessageSquare className="h-3.5 w-3.5 text-gold" />
                <span>Saved Conversations</span>
              </span>
              <div className="flex items-center gap-1">
                <button
                  onClick={handleStartNewChat}
                  className="text-xs text-gold hover:underline font-medium"
                >
                  + New
                </button>
                <button
                  type="button"
                  aria-label="Close conversation drawer"
                  onClick={() => setSidebarOpen(false)}
                  className="rounded p-1 text-muted-foreground hover:bg-surface-raised md:hidden"
                >
                  <X className="size-4" aria-hidden />
                </button>
              </div>
            </div>

            <div className="mt-3 space-y-1.5 max-h-[60vh] overflow-y-auto scrollbar-thin pr-1">
              {sessions.length === 0 ? (
                <div className="py-8 text-center text-xs text-muted-foreground/70">
                  No conversation history yet. Start asking questions!
                </div>
              ) : (
                sessions.map((s) => {
                  const isActive = s.session_id === sessionId;
                  return (
                    <div
                      key={s.session_id}
                      className={cn(
                        "group relative flex items-center justify-between rounded-xl px-3 py-2 text-xs transition cursor-pointer",
                        isActive
                          ? "bg-surface-raised border border-gold/40 text-foreground font-medium shadow-sm"
                          : "text-muted-foreground hover:bg-surface hover:text-foreground",
                      )}
                    >
                      <button
                        type="button"
                        onClick={() => handleSelectSession(s.session_id)}
                        aria-label={`Open conversation: ${s.preview || "Movie inquiry"}`}
                        className="min-w-0 flex-1 truncate pr-2 text-left"
                      >
                        <p className="truncate text-xs">{s.preview || "Movie inquiry"}</p>
                        <span className="text-[10px] text-muted-foreground/60">
                          {s.message_count} messages
                        </span>
                      </button>
                      <button
                        type="button"
                        onClick={(e) => handleDeleteSession(e, s.session_id)}
                        className="p-1 text-muted-foreground/60 hover:text-red-400 transition md:opacity-0 md:group-hover:opacity-100 focus-visible:opacity-100"
                        title="Delete session"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </aside>

        {/* Chat Area */}
        <div className="md:col-span-3 rounded-2xl border border-border/80 bg-surface/40 p-4 sm:p-6 backdrop-blur-xl shadow-lg">
          <ChatPanel
            messages={messages}
            activeIntent={activeIntent}
            isLoading={isLoading}
            onSendMessage={handleSendMessage}
            onFilterRemoved={(updated) => {
              if (activeSession.current === sessionId) setActiveIntent(updated);
            }}
            sessionId={sessionId}
            error={error}
          />
        </div>
      </div>
    </div>
  );
}
