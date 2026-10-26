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

func TestLaterTurnPromptSendsUserMessageOnce(t *testing.T) {
	question := "Why should taxpayers fund degrees that some students never use?"
	history := []models.Message{
		{Sender: "User", Text: "Free college widens access.", Phase: "Opening Statement"},
		{Sender: "Bot", Text: "Access matters, but cost does too.", Phase: "Opening Statement"},
		{Sender: "User", Text: question, Phase: "Cross Examination"},
	}
	p := constructPrompt(GetBotPersonality("Expert Emma"), "Should college be free?", history, "against",
		"Answer this question: "+question, 150)
	if n := strings.Count(p, question); n != 1 {
		t.Errorf("the user's message should appear once (in the transcript), found %d times", n)
	}
	if !strings.Contains(p, "Answer this question: the user’s latest message in the transcript") {
		t.Error("context should point to the message instead of repeating it")
	}
}

func TestGenerateDebateTopicIsLocal(t *testing.T) {
	prev := defaultLLM
	defaultLLM = nil // no AI provider needed
	t.Cleanup(func() { defaultLLM = prev })
	for _, level := range []string{"beginner", "intermediate", "advanced", "unknown"} {
		topic, err := GenerateDebateTopic(level)
		if err != nil || !strings.HasSuffix(topic, "?") {
			t.Errorf("%s: got %q, %v", level, topic, err)
		}
	}
}
