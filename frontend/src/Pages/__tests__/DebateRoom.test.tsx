import React from 'react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import DebateRoom from '../DebateRoom';
import { sendDebateMessage, judgeDebate } from '@/services/vsbot';

vi.mock('@/services/vsbot', () => ({
  sendDebateMessage: vi.fn(),
  judgeDebate: vi.fn(),
  concedeDebate: vi.fn(),
}));

const debateData = {
  userId: 'user-1',
  botName: 'Rookie Rick',
  botLevel: 'Easy',
  topic: 'Should AI rule the world?',
  stance: 'For',
  debateId: 'debate-1',
  phaseTimings: [
    { name: 'Opening Statements', time: 60 },
    { name: 'Cross-Examination', time: 60 },
    { name: 'Closing Statements', time: 60 },
  ],
};

const scores = { user: { score: 5, reason: 'ok' }, bot: { score: 4, reason: 'ok' } };
const judgment = {
  opening_statement: scores,
  cross_examination: scores,
  answers: scores,
  closing: scores,
  total: { user: 20, bot: 16 },
  verdict: { winner: 'User', reason: 'Stronger closing', congratulations: '', opponent_analysis: '' },
};

const wait = (ms: number) => act(() => new Promise((resolve) => setTimeout(resolve, ms)));

// Plays a whole debate: answers every user turn until the judge is called.
async function playDebate(stance: 'For' | 'Against') {
  render(
    <React.StrictMode>
      <MemoryRouter initialEntries={[{ pathname: '/debate/debate-1', state: { ...debateData, stance } }]}>
        <Routes>
          <Route path="/debate/:roomId" element={<DebateRoom />} />
        </Routes>
      </MemoryRouter>
    </React.StrictMode>
  );
  for (let turn = 0; turn < 40 && vi.mocked(judgeDebate).mock.calls.length === 0; turn++) {
    const input = screen.queryByRole('textbox') as HTMLTextAreaElement | null;
    if (input && !input.disabled) {
      fireEvent.change(input, { target: { value: `Argument ${turn}` } });
      fireEvent.click(screen.getByRole('button', { name: 'Send' }));
    }
    await wait(250);
  }
  // Let any duplicate request that was scheduled fire.
  await wait(1000);
}

describe('DebateRoom judging', () => {
  beforeEach(() => {
    vi.mocked(sendDebateMessage).mockReset().mockResolvedValue({ response: 'Bot argument.' } as never);
    vi.mocked(judgeDebate).mockReset().mockResolvedValue({ result: JSON.stringify(judgment) } as never);
  });

  it('judges a full debate once in React StrictMode', async () => {
    // StrictMode runs state updaters twice in development; the turn logic sends
    // the judge request from inside one, which used to judge (and score) twice.
    await playDebate('For');
    expect(judgeDebate).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/Stronger closing/i)).toBeInTheDocument();
  }, 30_000);
});
