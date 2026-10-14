import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import * as authApi from "@/api/auth";
import { AUTH_EXPIRED_EVENT, getAuthToken } from "@/api/client";
import type { AuthCredentials, RegisterPayload, User } from "@/types/movie";

interface AuthContextValue {
  user: User | null;
  ready: boolean;
  isAuthenticated: boolean;
  login: (credentials: AuthCredentials) => Promise<User>;
  register: (payload: RegisterPayload) => Promise<User>;
  loginAsDemo: () => Promise<User>;
  logout: () => void;
  savePreferences: (genres: string[], movieIds: number[]) => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let mounted = true;
    let restoreRevision = 0;
    const restore = async () => {
      const revision = ++restoreRevision;
      const token = getAuthToken();
      setReady(false);
      try {
        const next = await authApi.restoreSession();
        if (mounted && revision === restoreRevision && token === getAuthToken()) setUser(next);
      } catch {
        if (mounted && revision === restoreRevision && token === getAuthToken()) setUser(null);
      } finally {
        if (mounted && revision === restoreRevision) setReady(true);
      }
    };
    const expired = () => {
      restoreRevision += 1;
      authApi.logout();
      setUser(null);
      setReady(true);
    };
    const storageChanged = (event: StorageEvent) => {
      if (event.key === "filmory.token" || event.key === null) void restore();
    };
    void restore();
    window.addEventListener(AUTH_EXPIRED_EVENT, expired);
    window.addEventListener("storage", storageChanged);
    return () => {
      mounted = false;
      window.removeEventListener(AUTH_EXPIRED_EVENT, expired);
      window.removeEventListener("storage", storageChanged);
    };
  }, []);

  const login = useCallback(async (credentials: AuthCredentials) => {
    const next = await authApi.login(credentials);
    setUser(next);
    return next;
  }, []);

  const register = useCallback(async (payload: RegisterPayload) => {
    const next = await authApi.register(payload);
    setUser(next);
    return next;
  }, []);

  const loginAsDemo = useCallback(async () => {
    const next = await authApi.loginAsDemo();
    setUser(next);
    return next;
  }, []);

  const logout = useCallback(() => {
    authApi.logout();
    setUser(null);
  }, []);

  const savePreferences = useCallback(async (genres: string[], movieIds: number[]) => {
    const next = await authApi.updatePreferences(genres, movieIds);
    if (next) setUser(next);
  }, []);

  const value = useMemo(
    () => ({
      user,
      ready,
      isAuthenticated: Boolean(user),
      login,
      register,
      loginAsDemo,
      logout,
      savePreferences,
    }),
    [user, ready, login, register, loginAsDemo, logout, savePreferences],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
