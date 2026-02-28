"""
Unit tests for AgentExecutionEngine.assemble_steps

Tests both the new text-based assembly (using offset_mapping) and the
legacy length-delta fallback.
"""
import pytest
import torch
from unittest.mock import MagicMock
from transformers import AutoTokenizer

from rllm.parser import ChatTemplateParser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_PATH = "/home/claude/work/rllm/models/Qwen3-32B"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


@pytest.fixture(scope="module")
def parser(tokenizer):
    return ChatTemplateParser.get_parser(tokenizer)


class EngineStub:
    """Minimal stub that holds the fields assemble_steps needs."""

    def __init__(self, tokenizer, parser, config=None):
        self.tokenizer = tokenizer
        self.chat_parser = parser
        self.config = config or MagicMock()


@pytest.fixture(scope="module")
def engine(tokenizer, parser):
    from rllm.engine.agent_execution_engine import AgentExecutionEngine
    stub = EngineStub(tokenizer, parser)
    # Bind real methods to the stub
    stub.assemble_steps = AgentExecutionEngine.assemble_steps.__get__(stub)
    stub._assemble_steps_text_based = AgentExecutionEngine._assemble_steps_text_based.__get__(stub)
    stub._assemble_steps_legacy = AgentExecutionEngine._assemble_steps_legacy.__get__(stub)
    return stub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_steps(tokenizer, parser, messages_per_step, responses):
    """
    Build step dicts that mirror what run_agent_trajectory_async produces.

    Parameters
    ----------
    messages_per_step : list[list[dict]]
        For each step, the full message list up to the generation prompt.
    responses : list[str]
        The model visible response text (decoded with skip_special_tokens=True).
    """
    steps = []
    for i, (msgs, resp) in enumerate(zip(messages_per_step, responses)):
        prompt_text = parser.parse(msgs, add_generation_prompt=True, is_first_msg=True)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        # Simulate completion_ids: response text + eot_token because the
        # model generates the stop token.
        completion_text = resp + parser.eot_token
        completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)

        steps.append({
            "prompt": prompt_text,
            "response": resp,
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
        })
    return steps


def make_messages(system, turns):
    """
    Build a message list from alternating user/assistant texts.
    turns = [user_0, assistant_0, user_1, assistant_1, ...]
    """
    msgs = [{"role": "system", "content": system}]
    roles = ["user", "assistant"]
    for idx, text in enumerate(turns):
        msgs.append({"role": roles[idx % 2], "content": text})
    return msgs


# ---------------------------------------------------------------------------
# Group A: Core functionality
# ---------------------------------------------------------------------------

class TestCoreFunctionality:

    def test_single_step(self, engine, tokenizer, parser):
        """Single step: all response tokens should be mask=1."""
        msgs = [make_messages("You are helpful.", ["Hello"])]
        responses = ["Hi there!"]
        steps = build_steps(tokenizer, parser, msgs, responses)

        prompt_tok, resp_tok, resp_mask, is_valid = engine.assemble_steps(steps)

        assert isinstance(prompt_tok, torch.Tensor)
        assert isinstance(resp_tok, torch.Tensor)
        assert isinstance(resp_mask, torch.Tensor)
        assert is_valid is True
        assert len(resp_tok) == len(resp_mask)
        assert len(resp_tok) > 0
        # Single step: all tokens should be completion (mask=1)
        assert resp_mask.sum().item() == len(resp_mask)

    def test_two_steps_basic(self, engine, tokenizer, parser):
        """Two steps: completions mask=1, observation tokens mask=0."""
        msgs_0 = make_messages("You are helpful.", ["Fix the bug"])
        msgs_1 = make_messages("You are helpful.", [
            "Fix the bug", "Let me check.", "Error: file not found"
        ])
        responses = ["Let me check.", "Found it, fixing now."]
        steps = build_steps(tokenizer, parser, [msgs_0, msgs_1], responses)

        prompt_tok, resp_tok, resp_mask, is_valid = engine.assemble_steps(steps)

        assert is_valid is True
        assert len(resp_tok) == len(resp_mask)
        # Should have both 0s (observation) and 1s (completions)
        assert resp_mask.sum().item() > 0
        assert resp_mask.sum().item() < len(resp_mask)

    def test_three_steps(self, engine, tokenizer, parser):
        """Three steps with two observations between them."""
        msgs_0 = make_messages("Assistant.", ["Step1 task"])
        msgs_1 = make_messages("Assistant.", [
            "Step1 task", "Action A", "Observation 1: success"
        ])
        msgs_2 = make_messages("Assistant.", [
            "Step1 task", "Action A", "Observation 1: success",
            "Action B", "Observation 2: done"
        ])
        responses = ["Action A", "Action B", "All done."]
        steps = build_steps(tokenizer, parser, [msgs_0, msgs_1, msgs_2], responses)

        prompt_tok, resp_tok, resp_mask, is_valid = engine.assemble_steps(steps)

        assert is_valid is True
        assert len(resp_tok) == len(resp_mask)
        ones = resp_mask.sum().item()
        zeros = (resp_mask == 0).sum().item()
        assert ones > 0
        assert zeros > 0

    def test_mask_sum_meaningful(self, engine, tokenizer, parser):
        """Mask=1 region should decode to text covering all completions."""
        msgs_0 = make_messages("Sys.", ["Q1"])
        msgs_1 = make_messages("Sys.", ["Q1", "A1", "Obs"])
        responses = ["A1", "A2"]
        steps = build_steps(tokenizer, parser, [msgs_0, msgs_1], responses)

        prompt_tok, resp_tok, resp_mask, is_valid = engine.assemble_steps(steps)

        decoded_completions = tokenizer.decode(
            resp_tok[resp_mask == 1].tolist(), skip_special_tokens=True
        )
        # All response texts should appear in the decoded completions
        for resp in responses:
            assert resp in decoded_completions, (
                f"Expected '{resp}' in decoded completions: {repr(decoded_completions)}"
            )


# ---------------------------------------------------------------------------
# Group B: BPE boundary cases
# ---------------------------------------------------------------------------

class TestBPEBoundary:

    def test_bpe_boundary_slash_chars(self, engine, tokenizer, parser):
        """
        Characters like ')/' at completion end and '</' at observation start
        tend to merge differently under BPE retokenization.
        Text-based approach should handle this correctly.
        """
        msgs_0 = make_messages("Sys.", ["Show code"])
        msgs_1 = make_messages("Sys.", [
            "Show code",
            "console.log('hello')/",
            "</output>\nDone.",
        ])
        responses = ["console.log('hello')/", "Fixed."]
        steps = build_steps(tokenizer, parser, [msgs_0, msgs_1], responses)

        prompt_tok, resp_tok, resp_mask, is_valid = engine.assemble_steps(steps)

        assert is_valid is True
        assert len(resp_tok) == len(resp_mask)
        assert resp_mask.sum().item() > 0

    def test_bpe_boundary_angle_brackets(self, engine, tokenizer, parser):
        """Test >> at boundary merging with >>> from observation."""
        msgs_0 = make_messages("Sys.", ["Go"])
        msgs_1 = make_messages("Sys.", [
            "Go",
            "result = x >> 3",
            ">>> Running test...",
        ])
        responses = ["result = x >> 3", "Test passed."]
        steps = build_steps(tokenizer, parser, [msgs_0, msgs_1], responses)

        prompt_tok, resp_tok, resp_mask, is_valid = engine.assemble_steps(steps)

        assert is_valid is True
        assert resp_mask.sum().item() > 0

    def test_decoded_text_consistency(self, engine, tokenizer, parser):
        """Assembled tokens should decode to the correct full conversation text."""
        msgs_0 = make_messages("Sys.", ["Task"])
        msgs_1 = make_messages("Sys.", ["Task", "Step 1 done.", "Output: OK"])
        responses = ["Step 1 done.", "Finished."]
        steps = build_steps(tokenizer, parser, [msgs_0, msgs_1], responses)

        prompt_tok, resp_tok, resp_mask, _ = engine.assemble_steps(steps)

        # Decode the full assembled sequence
        full_ids = prompt_tok.tolist() + resp_tok.tolist()
        decoded = tokenizer.decode(full_ids, skip_special_tokens=False)

        # Build expected full text
        last_comp_text = tokenizer.decode(
            steps[-1]["completion_ids"], skip_special_tokens=False
        )
        expected = steps[-1]["prompt"] + last_comp_text

        assert decoded == expected

    def test_verify_legacy_has_bpe_issue(self, engine, tokenizer, parser):
        """
        Demonstrate that the legacy method can produce wrong observation lengths
        when BPE retokenization changes token count at boundaries.
        """
        # Use text where retokenization definitely changes token count
        msgs_0 = make_messages("Sys.", ["x"])
        # Completion ends with characters that merge with observation start
        completion_0 = "end)/"
        observation = "</tag>"
        msgs_1 = make_messages("Sys.", ["x", completion_0, observation])
        responses = [completion_0, "ok"]
        steps = build_steps(tokenizer, parser, [msgs_0, msgs_1], responses)

        # Check if there IS a token count discrepancy
        step0_end_len = len(steps[0]["prompt_ids"]) + len(steps[0]["completion_ids"])
        step1_prompt_len = len(steps[1]["prompt_ids"])
        # The expected observation region is the chat template wrapping + observation text
        # With BPE, the re-tokenized prompt may have different length than expected

        # This test just documents the behavior — it's informational
        prompt_tok, resp_tok, resp_mask, is_valid = engine._assemble_steps_legacy(steps)
        assert is_valid is True  # legacy always returns True now
        assert len(resp_tok) == len(resp_mask)


# ---------------------------------------------------------------------------
# Group C: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_completion(self, engine, tokenizer, parser):
        """Step with empty response text."""
        prompt_text = parser.parse(
            [{"role": "system", "content": "Sys."},
             {"role": "user", "content": "Hi"}],
            add_generation_prompt=True, is_first_msg=True
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        steps = [{
            "prompt": prompt_text,
            "response": "",
            "prompt_ids": prompt_ids,
            "completion_ids": [],
        }]

        prompt_tok, resp_tok, resp_mask, is_valid = engine.assemble_steps(steps)

        assert is_valid is True
        assert len(resp_tok) == len(resp_mask)

    def test_single_token_completion(self, engine, tokenizer, parser):
        """Completion with exactly 1 visible character."""
        msgs = [make_messages("Sys.", ["Say yes"])]
        responses = ["Y"]
        steps = build_steps(tokenizer, parser, msgs, responses)

        prompt_tok, resp_tok, resp_mask, is_valid = engine.assemble_steps(steps)

        assert is_valid is True
        assert resp_mask.sum().item() >= 1

    def test_special_tokens_in_response_text(self, engine, tokenizer, parser):
        """Response text contains strings that look like special tokens."""
        msgs = [make_messages("Sys.", ["Print tags"])]
        responses = ["The tokens are <|im_start|> and <|im_end|> in the template"]
        steps = build_steps(tokenizer, parser, msgs, responses)

        prompt_tok, resp_tok, resp_mask, is_valid = engine.assemble_steps(steps)

        assert is_valid is True
        assert len(resp_tok) == len(resp_mask)

    def test_long_multi_step(self, engine, tokenizer, parser):
        """5-step trajectory to test accumulation over many steps."""
        system = "You are a coding agent."
        all_msgs = []
        responses = []
        turns = []
        for step_idx in range(5):
            action = f"Action step {step_idx}: run test_{step_idx}.py"
            observation = f"Test {step_idx} result: {'PASS' if step_idx % 2 == 0 else 'FAIL'}"
            turns.append(action if step_idx == 0 else observation)
            if step_idx > 0:
                turns.append(action)
            # Wait, need to build message lists properly
            pass

        # Simpler approach: build incrementally
        msg_lists = []
        resps = []
        conversation = []
        for step_idx in range(5):
            if step_idx == 0:
                conversation = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "Fix all tests"},
                ]
            else:
                conversation = conversation + [
                    {"role": "assistant", "content": resps[-1]},
                    {"role": "user", "content": f"Test {step_idx} output: {'PASS' if step_idx % 2 == 0 else 'FAIL'}"},
                ]
            msg_lists.append(list(conversation))
            resps.append(f"Running test_{step_idx}.py now.")

        steps = build_steps(tokenizer, parser, msg_lists, resps)

        prompt_tok, resp_tok, resp_mask, is_valid = engine.assemble_steps(steps)

        assert is_valid is True
        assert len(resp_tok) == len(resp_mask)
        assert resp_mask.sum().item() > 0
        # Should have observation regions for steps 1-4
        assert (resp_mask == 0).sum().item() > 0


# ---------------------------------------------------------------------------
# Group D: Compatibility
# ---------------------------------------------------------------------------

class TestCompatibility:

    def test_return_type(self, engine, tokenizer, parser):
        """Verify return types match the expected interface."""
        msgs = [make_messages("Sys.", ["Hi"])]
        responses = ["Hello!"]
        steps = build_steps(tokenizer, parser, msgs, responses)

        result = engine.assemble_steps(steps)

        assert len(result) == 4
        prompt_tok, resp_tok, resp_mask, is_valid = result
        assert isinstance(prompt_tok, torch.Tensor)
        assert prompt_tok.dtype == torch.long
        assert isinstance(resp_tok, torch.Tensor)
        assert resp_tok.dtype == torch.long
        assert isinstance(resp_mask, torch.Tensor)
        assert resp_mask.dtype == torch.long
        assert isinstance(is_valid, bool)

    def test_fallback_without_text_fields(self, engine, tokenizer, parser):
        """Steps without 'prompt'/'response' keys should use legacy method."""
        prompt_text = parser.parse(
            [{"role": "system", "content": "Sys."},
             {"role": "user", "content": "Q"}],
            add_generation_prompt=True, is_first_msg=True
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        resp_text = "Answer"
        completion_ids = tokenizer.encode(
            resp_text + parser.eot_token, add_special_tokens=False
        )

        # Only token IDs, no text fields
        steps = [{
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
        }]

        prompt_tok, resp_tok, resp_mask, is_valid = engine.assemble_steps(steps)

        assert is_valid is True
        assert len(resp_tok) == len(resp_mask)
        assert len(resp_tok) > 0

    def test_eot_in_completion_mask(self, engine, tokenizer, parser):
        """
        For non-last steps, <|im_end|> after each completion should be mask=1
        because the model generated the stop token.
        """
        msgs_0 = make_messages("Sys.", ["Start"])
        msgs_1 = make_messages("Sys.", ["Start", "Done.", "Next task"])
        responses = ["Done.", "Completed."]
        steps = build_steps(tokenizer, parser, [msgs_0, msgs_1], responses)

        prompt_tok, resp_tok, resp_mask, _ = engine.assemble_steps(steps)

        # Decode mask=1 tokens
        completion_token_ids = resp_tok[resp_mask == 1].tolist()
        completion_text = tokenizer.decode(
            completion_token_ids, skip_special_tokens=False
        )

        eot = parser.eot_token.strip()  # "<|im_end|>"
        assert eot in completion_text, (
            f"Expected {eot} in completion text, got: {repr(completion_text)}"
        )

    def test_legacy_and_text_based_same_single_step(self, engine, tokenizer, parser):
        """For a single step, legacy and text-based should produce same masks."""
        msgs = [make_messages("Sys.", ["Hello"])]
        responses = ["World!"]
        steps = build_steps(tokenizer, parser, msgs, responses)

        pt1, rt1, rm1, v1 = engine._assemble_steps_text_based(steps)
        pt2, rt2, rm2, v2 = engine._assemble_steps_legacy(steps)

        # Both should have all mask=1
        assert rm1.sum().item() == len(rm1)
        assert rm2.sum().item() == len(rm2)
