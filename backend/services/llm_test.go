package services

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"arguehub/config"
)

// fakeProvider is an OpenAI-compatible endpoint that answers with scripted
// responses and records what it was sent.
type fakeProvider struct {
	t         *testing.T
	mu        sync.Mutex
	responses []fakeResponse
	requests  []map[string]any
	auth      []string
	server    *httptest.Server
}

type fakeResponse struct {
	status     int
	content    string
	body       string
	retryAfter string
}

func newFakeProvider(t *testing.T, responses ...fakeResponse) *fakeProvider {
	f := &fakeProvider{t: t, responses: responses}
	f.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		f.mu.Lock()
		f.requests = append(f.requests, body)
		f.auth = append(f.auth, r.Header.Get("Authorization"))
		resp := fakeResponse{status: 200, content: "ok"}
		if len(f.responses) > 0 {
			resp = f.responses[0]
			if len(f.responses) > 1 {
				f.responses = f.responses[1:]
			}
		}
		f.mu.Unlock()

		if resp.retryAfter != "" {
			w.Header().Set("Retry-After", resp.retryAfter)
		}
		if resp.status != 200 {
			w.WriteHeader(resp.status)
			_, _ = w.Write([]byte(resp.body))
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"content": resp.content}}},
			"usage":   map[string]any{"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": map[string]int{"cached_tokens": 4}},
		})
	}))
	t.Cleanup(f.server.Close)
	return f
}

func (f *fakeProvider) calls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.requests)
}

func (f *fakeProvider) request(i int) map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.requests[i]
}

func (f *fakeProvider) provider(name string) *llmProvider {
	return &llmProvider{name: name, baseURL: f.server.URL, apiKey: name + "-key", chatModel: "openai/gpt-oss-20b", judgeModel: "openai/gpt-oss-120b", reasoning: "groq"}
}

// testClient returns a client whose clock can be advanced and that never
// really sleeps between retries.
func testClient(providers ...*llmProvider) (*llmClient, *time.Time) {
	now := time.Date(2026, 10, 27, 12, 0, 0, 0, time.UTC)
	c := newLLMClient(providers)
	c.now = func() time.Time { return now }
	c.sleep = func(context.Context, time.Duration) error { return nil }
	return c, &now
}

func TestGenerateSendsOpenAICompatibleRequest(t *testing.T) {
	fake := newFakeProvider(t, fakeResponse{status: 200, content: "```json\n{\"winner\":\"For\"}\n```"})
	c, _ := testClient(fake.provider("groq"))

	got, err := c.Generate(context.Background(), llmRequest{Task: taskJudge, Prompt: "judge this", JSON: true, ReasoningEffort: "medium"})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got != `{"winner":"For"}` {
		t.Errorf("markdown fences should be stripped, got %q", got)
	}
	req := fake.request(0)
	if req["model"] != "openai/gpt-oss-120b" {
		t.Errorf("judge task should use the judge model, got %v", req["model"])
	}
	if req["max_tokens"] != float64(judgeMaxTokens) {
		t.Errorf("max_tokens = %v, want %d", req["max_tokens"], judgeMaxTokens)
	}
	if rf, _ := req["response_format"].(map[string]any); rf["type"] != "json_object" {
		t.Errorf("JSON request should set response_format json_object, got %v", req["response_format"])
	}
	if req["reasoning_effort"] != "medium" {
		t.Errorf("reasoning_effort = %v, want medium", req["reasoning_effort"])
	}
	msgs, _ := req["messages"].([]any)
	if len(msgs) != 1 || msgs[0].(map[string]any)["content"] != "judge this" {
		t.Errorf("unexpected messages: %v", req["messages"])
	}
	if fake.auth[0] != "Bearer groq-key" {
		t.Errorf("Authorization = %q", fake.auth[0])
	}
}

func TestChatTaskUsesChatModelWithoutJSONMode(t *testing.T) {
	fake := newFakeProvider(t)
	c, _ := testClient(fake.provider("groq"))
	if _, err := c.Generate(context.Background(), llmRequest{Prompt: "hi"}); err != nil {
		t.Fatal(err)
	}
	req := fake.request(0)
	if req["model"] != "openai/gpt-oss-20b" || req["max_tokens"] != float64(chatMaxTokens) {
		t.Errorf("chat defaults not applied: %v", req)
	}
	if _, ok := req["response_format"]; ok {
		t.Error("plain text request should not set response_format")
	}
}

func TestRateLimitFailsOverAndCoolsDown(t *testing.T) {
	limited := newFakeProvider(t, fakeResponse{status: 429, body: `{"error":"rate limit"}`, retryAfter: "20"})
	backup := newFakeProvider(t, fakeResponse{status: 200, content: "from backup"})
	c, now := testClient(limited.provider("groq"), backup.provider("cloudflare"))

	got, err := c.Generate(context.Background(), llmRequest{Prompt: "p"})
	if err != nil || got != "from backup" {
		t.Fatalf("got %q, %v; want the backup's answer", got, err)
	}
	if limited.calls() != 1 {
		t.Errorf("a 429 should not be retried on the same provider, got %d calls", limited.calls())
	}

	// Within the Retry-After window the limited provider is skipped.
	*now = now.Add(10 * time.Second)
	if _, err := c.Generate(context.Background(), llmRequest{Prompt: "p"}); err != nil {
		t.Fatal(err)
	}
	if limited.calls() != 1 {
		t.Errorf("provider should be skipped while cooling down, got %d calls", limited.calls())
	}

	// After it, the primary provider is used again.
	*now = now.Add(15 * time.Second)
	if _, err := c.Generate(context.Background(), llmRequest{Prompt: "p"}); err != nil {
		t.Fatal(err)
	}
	if limited.calls() != 2 {
		t.Errorf("provider should be retried after the cool-down, got %d calls", limited.calls())
	}
}

func TestCooldownIsPerModel(t *testing.T) {
	fake := newFakeProvider(t, fakeResponse{status: 429}, fakeResponse{status: 200, content: "judged"})
	c, _ := testClient(fake.provider("groq"))

	if _, err := c.Generate(context.Background(), llmRequest{Task: taskChat, Prompt: "p"}); err == nil {
		t.Fatal("expected an error when the only provider is rate-limited")
	}
	// The judge model has its own quota on Groq, so it is still tried.
	got, err := c.Generate(context.Background(), llmRequest{Task: taskJudge, Prompt: "p"})
	if err != nil || got != "judged" {
		t.Fatalf("judge call should not be blocked by the chat model's cool-down: %q, %v", got, err)
	}
}

func TestServerErrorIsRetriedOnSameProvider(t *testing.T) {
	fake := newFakeProvider(t, fakeResponse{status: 503, body: "overloaded"}, fakeResponse{status: 200, content: "second try"})
	c, _ := testClient(fake.provider("groq"))
	got, err := c.Generate(context.Background(), llmRequest{Prompt: "p"})
	if err != nil || got != "second try" {
		t.Fatalf("got %q, %v", got, err)
	}
	if fake.calls() != 2 {
		t.Errorf("want 2 calls, got %d", fake.calls())
	}
}

func TestJSONModeRejectedRetriesAsText(t *testing.T) {
	fake := newFakeProvider(t,
		fakeResponse{status: 400, body: `{"error":{"message":"response_format is not supported for this model"}}`},
		fakeResponse{status: 200, content: `{"ok":true}`},
	)
	c, _ := testClient(fake.provider("mistral"))
	got, err := c.Generate(context.Background(), llmRequest{Prompt: "p", JSON: true})
	if err != nil || got != `{"ok":true}` {
		t.Fatalf("got %q, %v", got, err)
	}
	if _, ok := fake.request(1)["response_format"]; ok {
		t.Error("retry should drop response_format")
	}
}

func TestAuthErrorMovesToNextProviderWithoutRetry(t *testing.T) {
	bad := newFakeProvider(t, fakeResponse{status: 401, body: "invalid key"})
	good := newFakeProvider(t)
	c, _ := testClient(bad.provider("groq"), good.provider("mistral"))
	if _, err := c.Generate(context.Background(), llmRequest{Prompt: "p"}); err != nil {
		t.Fatal(err)
	}
	if bad.calls() != 1 {
		t.Errorf("a 401 should not be retried, got %d calls", bad.calls())
	}
}

func TestEmptyResponseFallsThrough(t *testing.T) {
	empty := newFakeProvider(t, fakeResponse{status: 200, content: "  "})
	good := newFakeProvider(t, fakeResponse{status: 200, content: "answer"})
	c, _ := testClient(empty.provider("groq"), good.provider("cloudflare"))
	got, err := c.Generate(context.Background(), llmRequest{Prompt: "p"})
	if err != nil || got != "answer" {
		t.Fatalf("got %q, %v", got, err)
	}
}

func TestAllProvidersFailingReturnsCombinedError(t *testing.T) {
	a := newFakeProvider(t, fakeResponse{status: 429})
	b := newFakeProvider(t, fakeResponse{status: 403, body: "forbidden"})
	c, _ := testClient(a.provider("groq"), b.provider("mistral"))
	_, err := c.Generate(context.Background(), llmRequest{Prompt: "p"})
	if err == nil {
		t.Fatal("expected an error")
	}
	for _, want := range []string{"groq", "mistral", "429", "403"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q should mention %q", err, want)
		}
	}
}

func TestReasoningEffortOnlyForReasoningModels(t *testing.T) {
	fake := newFakeProvider(t)
	p := fake.provider("mistral")
	p.chatModel, p.reasoning = "mistral-small-latest", ""
	c, _ := testClient(p)
	if _, err := c.Generate(context.Background(), llmRequest{Prompt: "p", ReasoningEffort: "low"}); err != nil {
		t.Fatal(err)
	}
	if _, ok := fake.request(0)["reasoning_effort"]; ok {
		t.Error("reasoning_effort must not be sent to non-reasoning models")
	}

	or := newFakeProvider(t)
	op := or.provider("openrouter")
	op.reasoning = "openrouter"
	c2, _ := testClient(op)
	if _, err := c2.Generate(context.Background(), llmRequest{Prompt: "p", ReasoningEffort: "low"}); err != nil {
		t.Fatal(err)
	}
	if r, _ := or.request(0)["reasoning"].(map[string]any); r["effort"] != "low" {
		t.Errorf("OpenRouter should get reasoning.effort, got %v", or.request(0))
	}
}

func TestProvidersFromConfig(t *testing.T) {
	var cfg config.Config
	cfg.LLM.Groq.APIKey = "gk"
	cfg.LLM.Cloudflare.APIKey = "ck" // no account ID: skipped
	cfg.LLM.Mistral.APIKey = "<YOUR_MISTRAL_API_KEY>"
	cfg.LLM.OpenRouter.APIKey = "ok"
	cfg.LLM.OpenRouter.JudgeModel = "custom/judge"
	cfg.Gemini.ApiKey = "gem"

	names := func(ps []*llmProvider) string {
		var out []string
		for _, p := range ps {
			out = append(out, p.name)
		}
		return strings.Join(out, ",")
	}

	ps := providersFromConfig(&cfg)
	if got := names(ps); got != "groq,openrouter,gemini" {
		t.Errorf("default order = %q", got)
	}
	if ps[0].chatModel != "openai/gpt-oss-20b" || ps[0].judgeModel != "openai/gpt-oss-120b" {
		t.Errorf("groq defaults not applied: %+v", ps[0])
	}
	if ps[1].judgeModel != "custom/judge" || ps[1].chatModel != "openai/gpt-oss-20b:free" {
		t.Errorf("openrouter models wrong: %+v", ps[1])
	}

	cfg.LLM.Cloudflare.AccountID = "acc123"
	cfg.LLM.Providers = " gemini, cloudflare ,unknown,groq,gemini"
	ps = providersFromConfig(&cfg)
	if got := names(ps); got != "gemini,cloudflare,groq" {
		t.Errorf("custom order = %q", got)
	}
	if !strings.Contains(ps[1].baseURL, "/accounts/acc123/ai/v1") {
		t.Errorf("cloudflare base URL = %q", ps[1].baseURL)
	}
}

func TestGenerateWithoutProviders(t *testing.T) {
	var c *llmClient
	if _, err := c.Generate(context.Background(), llmRequest{Prompt: "p"}); err == nil {
		t.Error("expected an error with no client")
	}
	if _, err := newLLMClient(nil).Generate(context.Background(), llmRequest{Prompt: "p"}); err == nil {
		t.Error("expected an error with no providers")
	}
}

func TestParseRetryAfter(t *testing.T) {
	if got := parseRetryAfter("2.5"); got != 2500*time.Millisecond {
		t.Errorf("got %v", got)
	}
	if got := parseRetryAfter("soon"); got != 0 {
		t.Errorf("got %v", got)
	}
}
