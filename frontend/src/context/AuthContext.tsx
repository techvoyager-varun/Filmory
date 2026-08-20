import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import * as authApi from "@/api/auth";
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
    setUser(authApi.getCurrentUser());
    setReady(true);
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
