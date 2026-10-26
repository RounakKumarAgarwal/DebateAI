package controllers

import (
	"strings"
	"testing"

	"arguehub/models"
)

func TestValidateDebateInput(t *testing.T) {
	ok := []models.Message{{Sender: "User", Text: "College should be free."}}
	if err := validateDebateInput("Should college be free?", "Make your statement", ok); err != nil {
		t.Fatalf("normal debate rejected: %v", err)
	}

	cases := []struct {
		name    string
		topic   string
		context string
		history []models.Message
		want    string
	}{
		{"long topic", strings.Repeat("a", maxDebateTopicChars+1), "", nil, "topic"},
		{"long context", "", strings.Repeat("a", maxDebateContextChars+1), nil, "context"},
		{"too many messages", "", "", make([]models.Message, maxDebateMessages+1), "history"},
		{"long message", "", "", []models.Message{{Text: "ok"}, {Text: strings.Repeat("a", maxDebateMessageChars+1)}}, "message 2"},
	}
	for _, tc := range cases {
		err := validateDebateInput(tc.topic, tc.context, tc.history)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: got %v, want an error about %q", tc.name, err, tc.want)
		}
	}

	// Limits count characters, not bytes, so non-English text isn't penalised.
	hindi := strings.Repeat("क", maxDebateMessageChars)
	if err := validateDebateInput("", "", []models.Message{{Text: hindi}}); err != nil {
		t.Errorf("%d Devanagari characters should be allowed: %v", maxDebateMessageChars, err)
	}
}
