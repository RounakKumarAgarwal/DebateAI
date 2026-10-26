package services

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"strings"

	"arguehub/models"
)

// debateTopics are the Pros and Cons Challenge topics per skill level.
// Picking one locally saves an AI call per challenge and keeps topics
// well-formed; add topics here to widen the pool.
var debateTopics = map[string][]string{
	"beginner": {
		"Should school uniforms be mandatory?",
		"Is homework necessary for learning?",
		"Should students be allowed to use smartphones in class?",
		"Should junk food be banned in schools?",
		"Is it better to read books than watch movies?",
		"Should pets be allowed in classrooms?",
		"Should the school day start later?",
		"Are video games good for children?",
		"Should physical education be compulsory every day?",
		"Is it better to live in a city than in the countryside?",
	},
	"intermediate": {
		"Is social media beneficial for society?",
		"Should voting be mandatory?",
		"Is remote work better than office-based work?",
		"Should college education be free?",
		"Should plastic bags be banned?",
		"Is space exploration worth the cost?",
		"Should the voting age be lowered to 16?",
		"Should zoos be abolished?",
		"Is nuclear energy the best way to fight climate change?",
		"Should professional athletes be paid less?",
	},
	"advanced": {
		"Is globalization beneficial for developing countries?",
		"Should artificial intelligence be regulated?",
		"Should governments prioritize economic growth over environmental protection?",
		"Should a universal basic income replace existing welfare programs?",
		"Should social media platforms be held liable for user content?",
		"Are economic sanctions an effective foreign policy tool?",
		"Should genetic engineering of humans be permitted?",
		"Should central banks issue digital currencies?",
		"Do intellectual property laws hinder innovation?",
		"Should wealthy nations open their borders to climate refugees?",
	},
}

// GenerateDebateTopic returns a random topic for the user's skill level
// ("beginner", "intermediate" or "advanced").
func GenerateDebateTopic(skillLevel string) (string, error) {
	topics, ok := debateTopics[strings.ToLower(strings.TrimSpace(skillLevel))]
	if !ok || len(topics) == 0 {
		topics = debateTopics["beginner"]
	}
	return topics[rand.Intn(len(topics))], nil
}

// EvaluateProsCons evaluates the user's pros and cons
func EvaluateProsCons(topic string, pros, cons []string) (models.ProsConsEvaluation, error) {
	if !llmEnabled() {
		return models.ProsConsEvaluation{}, errors.New("AI provider not configured")
	}

	if len(pros) > 5 || len(cons) > 5 {
		return models.ProsConsEvaluation{}, errors.New("maximum of 5 pros and 5 cons allowed")
	}

	prompt := fmt.Sprintf(
		`Act as a debate coach and evaluate the following pros and cons for the topic "%s". For each argument:
- Score it out of 10 based on clarity (3), relevance (3), logic (2), and persuasiveness (2).
- Provide feedback explaining the score.
- Suggest a counterargument.

Pros:
%s

Cons:
%s

Required Output Format (JSON):
{
  "pros": [
    {"score": X, "feedback": "text", "counter": "text"},
    ...
  ],
  "cons": [
    {"score": X, "feedback": "text", "counter": "text"},
    ...
  ]
}`,
		topic,
		strings.Join(pros, "\n"),
		strings.Join(cons, "\n"),
	)

	ctx := context.Background()
	response, err := generateJudgeJSON(ctx, prompt)
	if err != nil {
		return models.ProsConsEvaluation{}, err
	}
	if response == "" {
		return models.ProsConsEvaluation{}, errors.New("no evaluation returned")
	}

	var eval models.ProsConsEvaluation
	if err := json.Unmarshal([]byte(response), &eval); err != nil {
		return models.ProsConsEvaluation{}, err
	}

	totalScore := 0
	for _, pro := range eval.Pros {
		totalScore += pro.Score
	}
	for _, con := range eval.Cons {
		totalScore += con.Score
	}
	eval.Score = totalScore

	return eval, nil
}
