package config

import (
	"fmt"
	"os"
	"strconv"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server struct {
		Port int `yaml:"port"`
	} `yaml:"server"`

	Cognito struct {
		AppClientId     string `yaml:"appClientId"`
		AppClientSecret string `yaml:"appClientSecret"`
		UserPoolId      string `yaml:"userPoolId"`
		Region          string `yaml:"region"`
	} `yaml:"cognito"`

	Openai struct {
		GptApiKey string `yaml:"gptApiKey"`
	} `yaml:"openai"`

	Gemini struct {
		ApiKey string `yaml:"apiKey"`
	} `yaml:"gemini"`

	// LLM configures the AI providers. Every provider uses the
	// OpenAI-compatible API; providers without an API key are skipped and
	// the rest are tried in the order given by Providers.
	LLM struct {
		Providers  string            `yaml:"providers"` // e.g. "groq,cloudflare,mistral,openrouter,gemini"
		Groq       LLMProviderConfig `yaml:"groq"`
		Cloudflare struct {
			LLMProviderConfig `yaml:",inline"`
			AccountID         string `yaml:"accountId"`
		} `yaml:"cloudflare"`
		Mistral    LLMProviderConfig `yaml:"mistral"`
		OpenRouter LLMProviderConfig `yaml:"openrouter"`
		// Gemini uses Gemini.ApiKey; only the model names live here.
		Gemini LLMProviderConfig `yaml:"gemini"`
	} `yaml:"llm"`

	Database struct {
		URI string `yaml:"uri"`
	} `yaml:"database"`

	Redis struct {
		Addr     string `yaml:"addr"`
		Password string `yaml:"password"`
		DB       int    `yaml:"db"`
	} `yaml:"redis"`

	JWT struct {
		Secret string `yaml:"secret"`
		Expiry int    `yaml:"expiry"`
	} `yaml:"jwt"`

	SMTP struct {
		Host        string `yaml:"host"`
		Port        int    `yaml:"port"`
		Username    string `yaml:"username"`
		Password    string `yaml:"password"`
		SenderEmail string `yaml:"senderEmail"`
		SenderName  string `yaml:"senderName"`
	} `yaml:"smtp"`

	GoogleOAuth struct {
		ClientID string `yaml:"clientID"`
	} `yaml:"googleOAuth"`
}

// LLMProviderConfig holds one AI provider's key and the models used for chat
// (bot replies) and judging. Empty models fall back to built-in defaults.
type LLMProviderConfig struct {
	APIKey     string `yaml:"apiKey"`
	ChatModel  string `yaml:"chatModel"`
	JudgeModel string `yaml:"judgeModel"`
}

func setStringEnv(name string, target *string) {
	if value := os.Getenv(name); value != "" {
		*target = value
	}
}

func setIntEnv(name string, target *int) error {
	value := os.Getenv(name)
	if value == "" {
		return nil
	}

	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fmt.Errorf("invalid integer in %s: %w", name, err)
	}

	*target = parsed
	return nil
}

// LoadConfig reads YAML configuration and applies environment overrides.
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal yaml: %w", err)
	}

	stringOverrides := []struct {
		name   string
		target *string
	}{
		{"COGNITO_APP_CLIENT_ID", &cfg.Cognito.AppClientId},
		{"COGNITO_APP_CLIENT_SECRET", &cfg.Cognito.AppClientSecret},
		{"COGNITO_USER_POOL_ID", &cfg.Cognito.UserPoolId},
		{"COGNITO_REGION", &cfg.Cognito.Region},
		{"OPENAI_API_KEY", &cfg.Openai.GptApiKey},
		{"GEMINI_API_KEY", &cfg.Gemini.ApiKey},
		{"LLM_PROVIDERS", &cfg.LLM.Providers},
		{"GROQ_API_KEY", &cfg.LLM.Groq.APIKey},
		{"GROQ_CHAT_MODEL", &cfg.LLM.Groq.ChatModel},
		{"GROQ_JUDGE_MODEL", &cfg.LLM.Groq.JudgeModel},
		{"CLOUDFLARE_ACCOUNT_ID", &cfg.LLM.Cloudflare.AccountID},
		{"CLOUDFLARE_API_TOKEN", &cfg.LLM.Cloudflare.APIKey},
		{"CLOUDFLARE_CHAT_MODEL", &cfg.LLM.Cloudflare.ChatModel},
		{"CLOUDFLARE_JUDGE_MODEL", &cfg.LLM.Cloudflare.JudgeModel},
		{"MISTRAL_API_KEY", &cfg.LLM.Mistral.APIKey},
		{"MISTRAL_CHAT_MODEL", &cfg.LLM.Mistral.ChatModel},
		{"MISTRAL_JUDGE_MODEL", &cfg.LLM.Mistral.JudgeModel},
		{"OPENROUTER_API_KEY", &cfg.LLM.OpenRouter.APIKey},
		{"OPENROUTER_CHAT_MODEL", &cfg.LLM.OpenRouter.ChatModel},
		{"OPENROUTER_JUDGE_MODEL", &cfg.LLM.OpenRouter.JudgeModel},
		{"GEMINI_CHAT_MODEL", &cfg.LLM.Gemini.ChatModel},
		{"GEMINI_JUDGE_MODEL", &cfg.LLM.Gemini.JudgeModel},
		{"DATABASE_URI", &cfg.Database.URI},
		{"REDIS_ADDR", &cfg.Redis.Addr},
		{"REDIS_PASSWORD", &cfg.Redis.Password},
		{"JWT_SECRET", &cfg.JWT.Secret},
		{"SMTP_HOST", &cfg.SMTP.Host},
		{"SMTP_USERNAME", &cfg.SMTP.Username},
		{"SMTP_PASSWORD", &cfg.SMTP.Password},
		{"SMTP_SENDER_EMAIL", &cfg.SMTP.SenderEmail},
		{"SMTP_SENDER_NAME", &cfg.SMTP.SenderName},
		{"GOOGLE_CLIENT_ID", &cfg.GoogleOAuth.ClientID},
	}

	for _, override := range stringOverrides {
		setStringEnv(override.name, override.target)
	}

	intOverrides := []struct {
		name   string
		target *int
	}{
		{"PORT", &cfg.Server.Port},
		{"REDIS_DB", &cfg.Redis.DB},
		{"JWT_EXPIRY", &cfg.JWT.Expiry},
		{"SMTP_PORT", &cfg.SMTP.Port},
	}

	for _, override := range intOverrides {
		if err := setIntEnv(override.name, override.target); err != nil {
			return nil, err
		}
	}

	return &cfg, nil
}
