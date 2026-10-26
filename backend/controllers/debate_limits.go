package controllers

import (
	"fmt"
	"unicode/utf8"

	"arguehub/models"
)

// Limits on what a client may send to the bot-debate AI endpoints. Every
// character ends up in an AI prompt, so these cap the tokens one request can
// use. They are generous for a real debate: 8 turns of a few hundred words.
const (
	maxDebateMessages     = 40
	maxDebateMessageChars = 4000 // ~700 words
	maxDebateTopicChars   = 300
	maxDebateContextChars = 1000
)

// validateDebateInput returns an error describing the first limit exceeded.
func validateDebateInput(topic, context string, history []models.Message) error {
	if utf8.RuneCountInString(topic) > maxDebateTopicChars {
		return fmt.Errorf("topic is too long (max %d characters)", maxDebateTopicChars)
	}
	if utf8.RuneCountInString(context) > maxDebateContextChars {
		return fmt.Errorf("context is too long (max %d characters)", maxDebateContextChars)
	}
	if len(history) > maxDebateMessages {
		return fmt.Errorf("debate history is too long (max %d messages)", maxDebateMessages)
	}
	for i, m := range history {
		if utf8.RuneCountInString(m.Text) > maxDebateMessageChars {
			return fmt.Errorf("message %d is too long (max %d characters)", i+1, maxDebateMessageChars)
		}
	}
	return nil
}
