import { test, beforeEach, after } from "node:test";
import assert from "node:assert/strict";
import { readFile, writeFile, mkdir, mkdtemp, rm } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import ts from "typescript";

// Exercise actual TS API modules using Node's test runner, with only fetch/storage replaced.
await mkdir(new URL("../.test-tmp/", import.meta.url), { recursive: true });
const temp = await mkdtemp(new URL("../.test-tmp/api-", import.meta.url));
for (const name of ["client", "auth", "interactions", "movies", "recommendationStudy"]) {
  const source = await readFile(new URL(`../src/api/${name}.ts`, import.meta.url), "utf8");
  const js = ts
    .transpileModule(source, {
      compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
    })
    .outputText.replaceAll("import.meta.env", '({ VITE_API_BASE_URL: "http://filmory.test" })')
    .replaceAll('"./client"', '"./client.mjs"');
  await writeFile(`${temp}/${name}.mjs`, js);
}
const client = await import(pathToFileURL(`${temp}/client.mjs`));
const auth = await import(pathToFileURL(`${temp}/auth.mjs`));
const interactions = await import(pathToFileURL(`${temp}/interactions.mjs`));
const movies = await import(pathToFileURL(`${temp}/movies.mjs`));
const studies = await import(pathToFileURL(`${temp}/recommendationStudy.mjs`));
const user = { userId: "1", name: "Test", email: "test@example.com" };
beforeEach(() => {
  const values = new Map();
  globalThis.window = new EventTarget();
  globalThis.localStorage = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  };
});
after(async () => {
  await rm(temp, { recursive: true, force: true });
});

test("movie details distinguish absent titles from service failures", async () => {
  globalThis.fetch = async () => new Response('{"detail":"Not found"}', { status: 404 });
  assert.equal(await movies.getMovie(999999999), null);
  globalThis.fetch = async () => new Response('{"detail":"Unavailable"}', { status: 503 });
  await assert.rejects(movies.getMovie(1), (error) => error.status === 503);
  globalThis.fetch = async () => {
    throw new TypeError("Network unavailable");
  };
  await assert.rejects(movies.getMovie(1), /Network unavailable/);
});

test("204 responses succeed and Headers objects are preserved", async () => {
  client.setAuthToken("token");
  globalThis.fetch = async (_url, init) => {
    assert.equal(init.headers.get("X-Test"), "yes");
    assert.equal(init.headers.get("Authorization"), "Bearer token");
    return new Response(null, { status: 204 });
  };
  assert.equal(
    await client.request("/delete", { headers: new Headers({ "X-Test": "yes" }) }),
    undefined,
  );
});
test("current-session 401 clears the token and notifies the app", async () => {
  let expired = false;
  window.addEventListener(client.AUTH_EXPIRED_EVENT, () => (expired = true));
  client.setAuthToken("expired");
  globalThis.fetch = async () => new Response('{"detail":"Expired"}', { status: 401 });
  await assert.rejects(client.request("/private"), (error) => error.status === 401);
  assert.equal(client.getAuthToken(), null);
  assert.equal(expired, true);
});
test("late 401 from an old account cannot log out the new account", async () => {
  let complete;
  client.setAuthToken("old");
  globalThis.fetch = () => new Promise((resolve) => (complete = resolve));
  const pending = client.request("/private");
  client.setAuthToken("new");
  complete(new Response('{"detail":"Expired"}', { status: 401 }));
  await assert.rejects(pending);
  assert.equal(client.getAuthToken(), "new");
});
test("logout prevents an outstanding login from silently signing back in", async () => {
  let complete;
  globalThis.fetch = () => new Promise((resolve) => (complete = resolve));
  const pending = auth.login({ email: "test@example.com", password: "secret" });
  auth.logout();
  complete(new Response(JSON.stringify({ accessToken: "late", user }), { status: 200 }));
  await assert.rejects(pending, /session changed/);
  assert.equal(client.getAuthToken(), null);
});
test("saved list errors propagate instead of pretending the list is empty", async () => {
  globalThis.fetch = async () => new Response('{"detail":"Unavailable"}', { status: 503 });
  await assert.rejects(interactions.getMyList(), /Unavailable/);
  await assert.rejects(interactions.getHistory(), /Unavailable/);
  await assert.rejects(interactions.getLikes(), /Unavailable/);
  await assert.rejects(interactions.getHiddenMovies(), /Unavailable/);
});

test("movie feedback uses the authenticated account and supports undo", async () => {
  client.setAuthToken("test-session");
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url, method: init.method, body: init.body });
    assert.equal(init.headers.get("Authorization"), "Bearer test-session");
    return new Response(JSON.stringify(url.endsWith("hidden-movies") ? [1] : { movieId: 1 }));
  };
  assert.deepEqual(await interactions.getHiddenMovies(), [1]);
  for (const type of ["not_interested", "show_again"]) {
    await interactions.recordInteraction({ movieId: 1, type });
    const call = calls.at(-1);
    assert.equal(call.url, "http://filmory.test/api/interactions");
    assert.equal(call.method, "POST");
    assert.deepEqual(JSON.parse(call.body), { movieId: 1, type });
  }
});
test("session restoration validates the token with the server", async () => {
  client.setAuthToken("valid");
  globalThis.fetch = async (url) => {
    assert.equal(url, "http://filmory.test/api/auth/me");
    return new Response(JSON.stringify(user));
  };
  assert.deepEqual(await auth.restoreSession(), user);
});

test("local review records explicit consent and associates each vote with its slate", async () => {
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url, method: init.method, body: JSON.parse(init.body) });
    return new Response(JSON.stringify({ id: "review-1", movies: [], answers: {} }));
  };
  await studies.startRecommendationStudy();
  await studies.saveStudyAnswer("review-1", 10, "unsure");
  assert.deepEqual(calls, [
    {
      url: "http://filmory.test/api/recommendation-study",
      method: "POST",
      body: { consent: true },
    },
    {
      url: "http://filmory.test/api/recommendation-study/review-1/answers",
      method: "PUT",
      body: { movieId: 10, choice: "unsure" },
    },
  ]);
});
