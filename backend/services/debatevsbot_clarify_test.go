package services

import (
	"strings"
	"testing"

	"arguehub/models"
)

// useFakeLLM points the package's client at a fake provider for one test.
func useFakeLLM(t *testing.T, fake *fakeProvider) {
	t.Helper()
	prev := defaultLLM
	c, _ := testClient(fake.provider("groq"))
	defaultLLM = c
	t.Cleanup(func() { defaultLLM = prev })
}

func TestBotReplyMentioningClarifyIsKept(t *testing.T) {
	// Regression: "clarify" anywhere in a reply used to replace the whole
	// argument with a canned line. It is Expert Emma's own catchphrase.
	reply := "Allow me to clarify. Studies indicate free tuition raises enrolment."
	useFakeLLM(t, newFakeProvider(t, fakeResponse{status: 200, content: reply}))

	got := GenerateBotResponse("Expert Emma", "Hard", "Should college be free?",
		[]models.Message{{Sender: "User", Text: "College should be free.", Phase: "Opening Statement"}},
		"against", "Make your statement", 150)
	if got != reply {
		t.Errorf("reply was replaced:\n got %q\nwant %q", got, reply)
	}
}

func TestMarkedClarificationKeepsModelWording(t *testing.T) {
	useFakeLLM(t, newFakeProvider(t, fakeResponse{status: 200, content: "[CLARIFY] Clouded, your point is. Clarify, you must."}))
	got := GenerateBotResponse("Yoda", "Legends", "Topic", nil, "for", "", 150)
	if got != "Clouded, your point is. Clarify, you must." {
		t.Errorf("got %q", got)
	}
}

func TestBareMarkerUsesCannedLineWithoutModelNotes(t *testing.T) {
	useFakeLLM(t, newFakeProvider(t, fakeResponse{status: 200, content: "[clarify]"}))
	got := GenerateBotResponse("Expert Emma", "Hard", "Topic", nil, "for", "", 150)
	if !strings.Contains(got, "Please clarify") {
		t.Errorf("want the canned clarification, got %q", got)
	}
	if strings.Contains(got, "(") {
		t.Errorf("notes meant for the model leaked into the reply: %q", got)
	}
}

func TestPromptAsksForMarker(t *testing.T) {
	p := constructPrompt(GetBotPersonality("Yoda"), "Topic", nil, "for", "", 150)
	if !strings.Contains(p, "start your reply with [CLARIFY]") {
		t.Error("prompt should tell the model how to mark clarification requests")
	}
}

func TestWithoutNote(t *testing.T) {
	cases := map[string]string{
		"Academic conferences (her rhetorical arena)": "Academic conferences",
		"Dagobah":                     "Dagobah",
		"Stark Industries (his lab) ": "Stark Industries (his lab) ",
	}
	for in, want := range cases {
		if got := withoutNote(in); got != want {
			t.Errorf("withoutNote(%q) = %q, want %q", in, got, want)
		}
	}
}
