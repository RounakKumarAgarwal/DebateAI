package services

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"arguehub/config"
)

// The backend talks to every LLM provider through the OpenAI-compatible
// chat completions API, so switching or adding a provider is a config change.
// Providers are tried in order: a rate-limited (429) provider is skipped for a
// cool-down period and the next one answers, so one provider's free-tier limit
// does not take the AI features down.

// llmTask picks which model of a provider handles a request.
type llmTask string

const (
	// taskChat: bot replies, weak statements, topics. A small, fast model.
	taskChat llmTask = "chat"
	// taskJudge: debate judging and argument evaluation. A stronger model.
	taskJudge llmTask = "judge"
)

const (
	chatMaxTokens  = 1200 // ~150-word reply plus low-effort reasoning
	judgeMaxTokens = 4000 // JSON verdict plus reasoning

	llmAttemptsPerProvider = 2
	llmRequestTimeout      = 60 * time.Second
	llmTotalTimeout        = 90 * time.Second
	llmDefaultCooldown     = 30 * time.Second
	llmMaxCooldown         = 10 * time.Minute
)

// llmProvider is one OpenAI-compatible endpoint with a model per task.
type llmProvider struct {
	name       string
	baseURL    string
	apiKey     string
	chatModel  string
	judgeModel string
	// reasoning says how this provider accepts a reasoning-effort hint for
	// reasoning models such as gpt-oss: "groq" (reasoning_effort),
	// "openrouter" (reasoning.effort) or "" (not sent).
	reasoning string
	headers   map[string]string
}

func (p *llmProvider) model(task llmTask) string {
	if task == taskJudge {
		return p.judgeModel
	}
	return p.chatModel
}

type llmClient struct {
	providers  []*llmProvider
	httpClient *http.Client
	now        func() time.Time
	sleep      func(context.Context, time.Duration) error

	mu        sync.Mutex
	cooldowns map[string]time.Time // provider/model -> rate-limited until
}

var defaultLLM *llmClient

// llmEnabled reports whether at least one AI provider is configured.
func llmEnabled() bool {
	return defaultLLM != nil && len(defaultLLM.providers) > 0
}

func newLLMClient(providers []*llmProvider) *llmClient {
	return &llmClient{
		providers:  providers,
		httpClient: &http.Client{Timeout: llmRequestTimeout},
		now:        time.Now,
		sleep:      sleepContext,
		cooldowns:  map[string]time.Time{},
	}
}

func sleepContext(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// Default models per provider. gpt-oss-20b handles chat and gpt-oss-120b
// judges; Cloudflare and OpenRouter serve the same models, so a failover
// keeps judging consistent.
const (
	defaultProviderOrder = "groq,cloudflare,mistral,openrouter,gemini"

	groqBaseURL       = "https://api.groq.com/openai/v1"
	cloudflareBaseURL = "https://api.cloudflare.com/client/v4/accounts/%s/ai/v1"
	mistralBaseURL    = "https://api.mistral.ai/v1"
	openRouterBaseURL = "https://openrouter.ai/api/v1"
	geminiBaseURL     = "https://generativelanguage.googleapis.com/v1beta/openai"
)

func valueOr(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return strings.TrimSpace(value)
}

// isPlaceholder treats sample-config values such as "<YOUR_GEMINI_API_KEY>"
// as unset.
func isPlaceholder(value string) bool {
	v := strings.TrimSpace(value)
	return v == "" || (strings.HasPrefix(v, "<") && strings.HasSuffix(v, ">"))
}

// providersFromConfig builds the ordered provider list, skipping providers
// without credentials and unknown names.
func providersFromConfig(cfg *config.Config) []*llmProvider {
	llm := cfg.LLM
	available := map[string]*llmProvider{}

	if !isPlaceholder(llm.Groq.APIKey) {
		available["groq"] = &llmProvider{
			name: "groq", baseURL: groqBaseURL, apiKey: llm.Groq.APIKey,
			chatModel:  valueOr(llm.Groq.ChatModel, "openai/gpt-oss-20b"),
			judgeModel: valueOr(llm.Groq.JudgeModel, "openai/gpt-oss-120b"),
			reasoning:  "groq",
		}
	}
	if !isPlaceholder(llm.Cloudflare.APIKey) && !isPlaceholder(llm.Cloudflare.AccountID) {
		available["cloudflare"] = &llmProvider{
			name: "cloudflare", baseURL: fmt.Sprintf(cloudflareBaseURL, llm.Cloudflare.AccountID), apiKey: llm.Cloudflare.APIKey,
			chatModel:  valueOr(llm.Cloudflare.ChatModel, "@cf/openai/gpt-oss-20b"),
			judgeModel: valueOr(llm.Cloudflare.JudgeModel, "@cf/openai/gpt-oss-120b"),
		}
	}
	if !isPlaceholder(llm.Mistral.APIKey) {
		available["mistral"] = &llmProvider{
			name: "mistral", baseURL: mistralBaseURL, apiKey: llm.Mistral.APIKey,
			chatModel:  valueOr(llm.Mistral.ChatModel, "mistral-small-latest"),
			judgeModel: valueOr(llm.Mistral.JudgeModel, "mistral-medium-latest"),
		}
	}
	if !isPlaceholder(llm.OpenRouter.APIKey) {
		available["openrouter"] = &llmProvider{
			name: "openrouter", baseURL: openRouterBaseURL, apiKey: llm.OpenRouter.APIKey,
			chatModel:  valueOr(llm.OpenRouter.ChatModel, "openai/gpt-oss-20b:free"),
			judgeModel: valueOr(llm.OpenRouter.JudgeModel, "openai/gpt-oss-120b:free"),
			reasoning:  "openrouter",
			headers:    map[string]string{"X-Title": "DebateAI"},
		}
	}
	if !isPlaceholder(cfg.Gemini.ApiKey) {
		available["gemini"] = &llmProvider{
			name: "gemini", baseURL: geminiBaseURL, apiKey: cfg.Gemini.ApiKey,
			chatModel:  valueOr(llm.Gemini.ChatModel, "gemini-3.6-flash"),
			judgeModel: valueOr(llm.Gemini.JudgeModel, "gemini-3.6-flash"),
		}
	}

	var ordered []*llmProvider
	seen := map[string]bool{}
	for _, name := range strings.Split(valueOr(llm.Providers, defaultProviderOrder), ",") {
		name = strings.ToLower(strings.TrimSpace(name))
		if p, ok := available[name]; ok && !seen[name] {
			ordered = append(ordered, p)
			seen[name] = true
		}
	}
	return ordered
}

// initLLM configures the shared client from config. It returns the provider
// names in the order they will be tried.
func initLLM(cfg *config.Config) []string {
	providers := providersFromConfig(cfg)
	defaultLLM = newLLMClient(providers)
	names := make([]string, 0, len(providers))
	for _, p := range providers {
		names = append(names, fmt.Sprintf("%s (chat %s, judge %s)", p.name, p.chatModel, p.judgeModel))
	}
	return names
}

type llmRequest struct {
	Task      llmTask
	Prompt    string
	JSON      bool
	MaxTokens int
	// ReasoningEffort for reasoning models: "low", "medium" or "high".
	ReasoningEffort string
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage llmUsage `json:"usage"`
}

type llmUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

// providerError classifies a failed call so the client knows whether to
// retry, skip the provider, or drop JSON mode.
type providerError struct {
	provider   string
	status     int
	body       string
	retryAfter time.Duration
	err        error
}

func (e *providerError) Error() string {
	if e.err != nil {
		return fmt.Sprintf("%s: %v", e.provider, e.err)
	}
	body := e.body
	if len(body) > 300 {
		body = body[:300] + "…"
	}
	return fmt.Sprintf("%s: HTTP %d: %s", e.provider, e.status, body)
}

func (e *providerError) rateLimited() bool { return e.status == http.StatusTooManyRequests }

// transient errors are worth retrying on the same provider.
func (e *providerError) transient() bool {
	return e.err != nil || e.status >= 500
}

// jsonModeRejected detects providers or models that do not support
// response_format, so the request can be retried as plain text.
func (e *providerError) jsonModeRejected() bool {
	if e.status != http.StatusBadRequest {
		return false
	}
	b := strings.ToLower(e.body)
	return strings.Contains(b, "response_format") || strings.Contains(b, "json_object") || strings.Contains(b, "json mode") || strings.Contains(b, "json_validate_failed")
}

func (c *llmClient) cooldownKey(p *llmProvider, model string) string { return p.name + "/" + model }

func (c *llmClient) coolingDown(p *llmProvider, model string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	until, ok := c.cooldowns[c.cooldownKey(p, model)]
	return ok && c.now().Before(until)
}

func (c *llmClient) setCooldown(p *llmProvider, model string, d time.Duration) {
	if d <= 0 {
		d = llmDefaultCooldown
	}
	if d > llmMaxCooldown {
		d = llmMaxCooldown
	}
	c.mu.Lock()
	c.cooldowns[c.cooldownKey(p, model)] = c.now().Add(d)
	c.mu.Unlock()
}

// Generate sends the request to the first provider that answers.
func (c *llmClient) Generate(ctx context.Context, req llmRequest) (string, error) {
	if c == nil || len(c.providers) == 0 {
		return "", errors.New("no AI provider configured")
	}
	if req.Task == "" {
		req.Task = taskChat
	}
	// Bound the whole failover chain so a request never hangs for minutes.
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, llmTotalTimeout)
		defer cancel()
	}
	if req.MaxTokens == 0 {
		req.MaxTokens = chatMaxTokens
		if req.Task == taskJudge {
			req.MaxTokens = judgeMaxTokens
		}
	}

	var errs []string
	for _, p := range c.providers {
		model := p.model(req.Task)
		if model == "" {
			continue
		}
		if c.coolingDown(p, model) {
			errs = append(errs, p.name+": rate-limited, cooling down")
			continue
		}
		text, err := c.tryProvider(ctx, p, model, req)
		if err == nil {
			return text, nil
		}
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		errs = append(errs, err.Error())
	}
	return "", fmt.Errorf("all AI providers failed: %s", strings.Join(errs, "; "))
}

func (c *llmClient) tryProvider(ctx context.Context, p *llmProvider, model string, req llmRequest) (string, error) {
	useJSON := req.JSON
	backoff := time.Second
	var lastErr error
	for attempt := 1; attempt <= llmAttemptsPerProvider; attempt++ {
		started := c.now()
		text, usage, err := c.call(ctx, p, model, req, useJSON)
		if err == nil {
			log.Printf("llm: provider=%s model=%s task=%s prompt_tokens=%d cached_tokens=%d completion_tokens=%d latency=%s",
				p.name, model, req.Task, usage.PromptTokens, usage.PromptTokensDetails.CachedTokens, usage.CompletionTokens, c.now().Sub(started).Round(time.Millisecond))
			return text, nil
		}
		lastErr = err

		var pe *providerError
		if !errors.As(err, &pe) {
			return "", err
		}
		switch {
		case pe.rateLimited():
			// Don't wait: the next provider can answer now.
			c.setCooldown(p, model, pe.retryAfter)
			log.Printf("llm: %s/%s rate-limited, trying the next provider", p.name, model)
			return "", err
		case useJSON && pe.jsonModeRejected():
			// Retry immediately without response_format; callers still
			// extract the JSON object from the text.
			log.Printf("llm: %s/%s rejected JSON mode, retrying as text", p.name, model)
			useJSON = false
			attempt--
			continue
		case !pe.transient():
			return "", err
		}
		if attempt < llmAttemptsPerProvider {
			if err := c.sleep(ctx, backoff); err != nil {
				return "", err
			}
			backoff *= 2
		}
	}
	return "", lastErr
}

func (c *llmClient) call(ctx context.Context, p *llmProvider, model string, req llmRequest, useJSON bool) (string, llmUsage, error) {
	body := map[string]any{
		"model":      model,
		"messages":   []chatMessage{{Role: "user", Content: req.Prompt}},
		"max_tokens": req.MaxTokens,
	}
	if useJSON {
		body["response_format"] = map[string]string{"type": "json_object"}
	}
	if req.ReasoningEffort != "" && isReasoningModel(model) {
		switch p.reasoning {
		case "groq":
			body["reasoning_effort"] = req.ReasoningEffort
		case "openrouter":
			body["reasoning"] = map[string]string{"effort": req.ReasoningEffort}
		}
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return "", llmUsage{}, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(p.baseURL, "/")+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return "", llmUsage{}, &providerError{provider: p.name, err: err}
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	for k, v := range p.headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return "", llmUsage{}, &providerError{provider: p.name, err: err}
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return "", llmUsage{}, &providerError{provider: p.name, err: err}
	}
	if resp.StatusCode != http.StatusOK {
		return "", llmUsage{}, &providerError{
			provider:   p.name,
			status:     resp.StatusCode,
			body:       string(raw),
			retryAfter: parseRetryAfter(resp.Header.Get("Retry-After")),
		}
	}

	var parsed chatResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return "", llmUsage{}, &providerError{provider: p.name, err: fmt.Errorf("invalid response: %w", err)}
	}
	if len(parsed.Choices) == 0 || strings.TrimSpace(parsed.Choices[0].Message.Content) == "" {
		// Treated as transient: e.g. a reasoning model that spent its whole
		// token budget thinking.
		return "", llmUsage{}, &providerError{provider: p.name, err: errors.New("empty response")}
	}
	return cleanModelOutput(parsed.Choices[0].Message.Content), parsed.Usage, nil
}

func isReasoningModel(model string) bool {
	return strings.Contains(model, "gpt-oss")
}

// parseRetryAfter reads a Retry-After header given in seconds.
func parseRetryAfter(value string) time.Duration {
	secs, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
	if err != nil || secs <= 0 {
		return 0
	}
	return time.Duration(secs * float64(time.Second))
}

func cleanModelOutput(text string) string {
	cleaned := strings.TrimSpace(text)
	cleaned = strings.TrimPrefix(cleaned, "```json")
	cleaned = strings.TrimPrefix(cleaned, "```JSON")
	cleaned = strings.TrimPrefix(cleaned, "```")
	cleaned = strings.TrimSuffix(cleaned, "```")
	return strings.TrimSpace(cleaned)
}

// Helpers used by the services. Chat calls use the small model with low
// reasoning effort; judge calls use the larger model.

func generateChatText(ctx context.Context, prompt string) (string, error) {
	return defaultLLM.Generate(ctx, llmRequest{Task: taskChat, Prompt: prompt, ReasoningEffort: "low"})
}

func generateChatJSON(ctx context.Context, prompt string) (string, error) {
	return defaultLLM.Generate(ctx, llmRequest{Task: taskChat, Prompt: prompt, JSON: true, ReasoningEffort: "low"})
}

func generateJudgeText(ctx context.Context, prompt string) (string, error) {
	return defaultLLM.Generate(ctx, llmRequest{Task: taskJudge, Prompt: prompt, ReasoningEffort: "medium"})
}

func generateJudgeJSON(ctx context.Context, prompt string) (string, error) {
	return defaultLLM.Generate(ctx, llmRequest{Task: taskJudge, Prompt: prompt, JSON: true, ReasoningEffort: "medium"})
}
