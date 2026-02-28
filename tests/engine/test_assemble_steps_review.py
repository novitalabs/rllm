"""
Additional tests for assemble_steps addressing GPT-5.2 review concerns:
- Prompt/response boundary token crossing
- EOT detection robustness
- (0,0) offset handling
- Unicode / non-ASCII text
- Span monotonicity
- Realistic completion_ids (no trailing \n)
"""
import pytest
import torch
from unittest.mock import MagicMock
from transformers import AutoTokenizer

from rllm.parser import ChatTemplateParser
from rllm.engine.agent_execution_engine import AgentExecutionEngine


MODEL_PATH = "/home/claude/work/rllm/models/Qwen3-32B"
IM_END_ID = 151645  # <|im_end|> token id for Qwen


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


@pytest.fixture(scope="module")
def parser(tokenizer):
    return ChatTemplateParser.get_parser(tokenizer)


@pytest.fixture(scope="module")
def engine(tokenizer, parser):
    stub = MagicMock()
    stub.tokenizer = tokenizer
    stub.chat_parser = parser
    stub.assemble_steps = AgentExecutionEngine.assemble_steps.__get__(stub)
    stub._assemble_steps_text_based = AgentExecutionEngine._assemble_steps_text_based.__get__(stub)
    stub._assemble_steps_legacy = AgentExecutionEngine._assemble_steps_legacy.__get__(stub)
    return stub


def make_messages(system, turns):
    msgs = [{"role": "system", "content": system}]
    roles = ["user", "assistant"]
    for idx, text in enumerate(turns):
        msgs.append({"role": roles[idx % 2], "content": text})
    return msgs


def build_steps_realistic(tokenizer, parser, messages_per_step, responses):
    """
    Build steps with REALISTIC completion_ids: response tokens + <|im_end|>.
    No trailing \\n in completion_ids (model stops at stop token).
    """
    steps = []
    for i, (msgs, resp) in enumerate(zip(messages_per_step, responses)):
        prompt_text = parser.parse(msgs, add_generation_prompt=True, is_first_msg=True)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        # Realistic: model generates response text + <|im_end|> only
        resp_token_ids = tokenizer.encode(resp, add_special_tokens=False)
        completion_ids = resp_token_ids + [IM_END_ID]

        steps.append({
            "prompt": prompt_text,
            "response": resp,
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
        })
    return steps


# ---------------------------------------------------------------------------
# GPT-5.2 Issue 1: Prompt/Response boundary token crossing
# ---------------------------------------------------------------------------

class TestBoundaryToken:

    def test_prompt_ends_on_token_boundary(self, engine, tokenizer, parser):
        """
        Verify that the Qwen chat template prompt ALWAYS ends on a clean
        token boundary (the \\n after 'assistant').
        No token should cross the prompt/response split.
        """
        msgs = make_messages("System prompt here.", ["User query text"])
        prompt_text = parser.parse(msgs, add_generation_prompt=True, is_first_msg=True)
        response = "Model response"

        full_text = prompt_text + response
        enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        boundary = len(prompt_text)

        # Check no token crosses the boundary
        for cs, ce in enc["offset_mapping"]:
            crossing = cs < boundary < ce
            assert not crossing, (
                f"Token crosses prompt/response boundary at char {boundary}: "
                f"token span ({cs}, {ce}), text={repr(full_text[cs:ce])}"
            )

    def test_prompt_boundary_various_content(self, engine, tokenizer, parser):
        """
        Test boundary alignment with various prompt content lengths to
        confirm the chat template always produces clean token boundaries.
        """
        test_cases = [
            ("A", ["x"]),
            ("Long system prompt " * 20, ["Short query"]),
            ("Sys", ["A very long user prompt with lots of text " * 10]),
            ("S", ["Unicode: \u4f60\u597d\u4e16\u754c"]),
        ]
        for system, turns in test_cases:
            msgs = make_messages(system, turns)
            prompt_text = parser.parse(msgs, add_generation_prompt=True, is_first_msg=True)
            full_text = prompt_text + "response text here"
            enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
            boundary = len(prompt_text)

            for cs, ce in enc["offset_mapping"]:
                assert not (cs < boundary < ce), (
                    f"Boundary crossing with system={repr(system[:20])}: "
                    f"token ({cs},{ce}) crosses {boundary}"
                )


# ---------------------------------------------------------------------------
# GPT-5.2 Issue 2: EOT detection robustness
# ---------------------------------------------------------------------------

class TestEOTDetection:

    def test_eot_present_after_response(self, engine, tokenizer, parser):
        """Verify eot_token is found right after each non-last response."""
        msgs_0 = make_messages("Sys.", ["Q"])
        msgs_1 = make_messages("Sys.", ["Q", "A1", "Obs1"])
        msgs_2 = make_messages("Sys.", ["Q", "A1", "Obs1", "A2", "Obs2"])
        responses = ["A1", "A2", "A3"]
        steps = build_steps_realistic(tokenizer, parser,
                                       [msgs_0, msgs_1, msgs_2], responses)

        # Reconstruct full_text as the implementation does
        last_comp = tokenizer.decode(steps[-1]["completion_ids"], skip_special_tokens=False)
        full_text = steps[-1]["prompt"] + last_comp
        eot = parser.eot_token

        for i in range(len(steps) - 1):
            comp_end_text = len(steps[i]["prompt"]) + len(steps[i]["response"])
            actual = full_text[comp_end_text:comp_end_text + len(eot)]
            assert actual == eot, (
                f"Step {i}: expected eot at char {comp_end_text}, "
                f"got {repr(actual)} instead of {repr(eot)}"
            )

    def test_eot_missing_graceful_fallback(self, engine, tokenizer, parser):
        """
        If eot_token is not found (e.g., different template), the span
        should still be valid (just without eot).
        """
        msgs_0 = make_messages("Sys.", ["Q"])
        msgs_1 = make_messages("Sys.", ["Q", "A1", "Obs"])
        responses = ["A1", "A2"]
        steps = build_steps_realistic(tokenizer, parser, [msgs_0, msgs_1], responses)

        # Temporarily override eot_token to something that won't match
        original_eot = parser.eot_token
        parser.eot_token = "<FAKE_EOT>"
        try:
            pt, rt, rm, v = engine.assemble_steps(steps)
            assert v is True
            assert len(rt) == len(rm)
            # Mask should still have completion regions
            assert rm.sum().item() > 0
        finally:
            parser.eot_token = original_eot


# ---------------------------------------------------------------------------
# GPT-5.2 Issue 3: (0,0) offset handling
# ---------------------------------------------------------------------------

class TestZeroOffsets:

    def test_no_zero_offsets_with_add_special_false(self, engine, tokenizer, parser):
        """
        Verify that add_special_tokens=False never produces (0,0) offsets
        on the Qwen2TokenizerFast tokenizer.
        """
        # Various text patterns including special token strings
        texts = [
            "<|im_start|>system\nHello<|im_end|>\n<|im_start|>user\nWorld<|im_end|>\n",
            "Normal text without any special tokens.",
            "<|im_start|>assistant\n\u4f60\u597d<|im_end|>\n",
            "",  # empty text
        ]
        for text in texts:
            if not text:
                continue
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            for i, (cs, ce) in enumerate(enc["offset_mapping"]):
                assert not (cs == 0 and ce == 0 and i > 0), (
                    f"Unexpected (0,0) offset at token {i} for text: {repr(text[:50])}"
                )


# ---------------------------------------------------------------------------
# GPT-5.2 Issue 4: Unicode / non-ASCII
# ---------------------------------------------------------------------------

class TestUnicode:

    def test_chinese_response(self, engine, tokenizer, parser):
        """Multi-step with Chinese text in both response and observation."""
        msgs_0 = make_messages("\u7cfb\u7edf", ["\u4f60\u597d"])
        msgs_1 = make_messages("\u7cfb\u7edf", [
            "\u4f60\u597d", "\u6211\u6765\u68c0\u67e5",
            "\u9519\u8bef\uff1a\u6587\u4ef6\u672a\u627e\u5230"
        ])
        responses = ["\u6211\u6765\u68c0\u67e5", "\u5df2\u4fee\u590d"]
        steps = build_steps_realistic(tokenizer, parser, [msgs_0, msgs_1], responses)

        pt, rt, rm, v = engine.assemble_steps(steps)

        assert v is True
        assert len(rt) == len(rm)
        # Verify completion text contains the Chinese responses
        comp_text = tokenizer.decode(rt[rm == 1].tolist(), skip_special_tokens=True)
        for resp in responses:
            assert resp in comp_text, (
                f"Expected '{resp}' in completion text: {repr(comp_text)}"
            )

    def test_mixed_unicode_and_ascii(self, engine, tokenizer, parser):
        """Response with mixed English, Chinese, and emoji-like chars."""
        msgs = [make_messages("Sys.", ["Describe"])]
        responses = ["Result: \u6210\u529f (100%) path=/tmp/\u6d4b\u8bd5.txt"]
        steps = build_steps_realistic(tokenizer, parser, msgs, responses)

        pt, rt, rm, v = engine.assemble_steps(steps)

        assert v is True
        decoded = tokenizer.decode(rt[rm == 1].tolist(), skip_special_tokens=True)
        assert "\u6210\u529f" in decoded
        assert "100%" in decoded

    def test_unicode_offset_alignment(self, tokenizer):
        """
        Verify that Python string char offsets align with tokenizer
        offset_mapping for multi-byte Unicode characters.
        """
        text = "abc\u4f60\u597d\u4e16\u754cxyz"
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)

        for cs, ce in enc["offset_mapping"]:
            substr = text[cs:ce]
            decoded = tokenizer.decode(
                [enc["input_ids"][enc["offset_mapping"].index((cs, ce))]],
                skip_special_tokens=False
            )
            # The substring from text should match the decoded token
            assert substr == decoded, (
                f"Offset mismatch: text[{cs}:{ce}]={repr(substr)} "
                f"vs decoded={repr(decoded)}"
            )


# ---------------------------------------------------------------------------
# GPT-5.2 Issue 5: Span monotonicity and ordering
# ---------------------------------------------------------------------------

class TestSpanMonotonicity:

    def test_completion_spans_ordered_and_non_overlapping(self, engine, tokenizer, parser):
        """Completion char spans should be strictly ordered and non-overlapping."""
        msgs_0 = make_messages("S.", ["Q1"])
        msgs_1 = make_messages("S.", ["Q1", "A1", "O1"])
        msgs_2 = make_messages("S.", ["Q1", "A1", "O1", "A2", "O2"])
        msgs_3 = make_messages("S.", ["Q1", "A1", "O1", "A2", "O2", "A3", "O3"])
        responses = ["A1", "A2", "A3", "A4"]
        steps = build_steps_realistic(tokenizer, parser,
                                       [msgs_0, msgs_1, msgs_2, msgs_3], responses)

        # Reconstruct spans as the implementation does
        last_comp = tokenizer.decode(steps[-1]["completion_ids"], skip_special_tokens=False)
        full_text = steps[-1]["prompt"] + last_comp
        eot = parser.eot_token

        spans = []
        for i, step in enumerate(steps):
            cs = len(step["prompt"])
            ce_text = cs + len(step["response"])
            if i < len(steps) - 1 and full_text[ce_text:ce_text + len(eot)] == eot:
                ce = ce_text + len(eot)
            else:
                ce = len(full_text)
            spans.append((cs, ce))

        # Verify ordered and non-overlapping
        for j in range(1, len(spans)):
            prev_end = spans[j - 1][1]
            curr_start = spans[j][0]
            assert prev_end <= curr_start, (
                f"Spans overlap: span[{j-1}]={spans[j-1]}, span[{j}]={spans[j]}"
            )
            # Spans should be strictly after each other (with observation gap)
            assert curr_start > prev_end, (
                f"No observation gap between span[{j-1}] and span[{j}]"
            )

    def test_mask_transitions_are_clean(self, engine, tokenizer, parser):
        """
        In the response mask, transitions between 0 and 1 regions should
        correspond to clean completion/observation boundaries.
        """
        msgs_0 = make_messages("S.", ["Q"])
        msgs_1 = make_messages("S.", ["Q", "A1", "Obs"])
        responses = ["A1", "A2"]
        steps = build_steps_realistic(tokenizer, parser, [msgs_0, msgs_1], responses)

        pt, rt, rm, v = engine.assemble_steps(steps)

        # Count transitions
        transitions = 0
        for i in range(1, len(rm)):
            if rm[i] != rm[i - 1]:
                transitions += 1

        # 2-step trajectory: [comp1=1s][obs=0s][comp2=1s]
        # Expected transitions: 1->0 and 0->1 = 2 transitions
        assert transitions == 2, (
            f"Expected 2 mask transitions, got {transitions}. "
            f"Mask: {rm.tolist()}"
        )


# ---------------------------------------------------------------------------
# GPT-5.2 Suggested: Realistic completion_ids
# ---------------------------------------------------------------------------

class TestRealisticCompletionIds:

    def test_realistic_two_step(self, engine, tokenizer, parser):
        """
        Two-step trajectory with realistic completion_ids
        (response tokens + <|im_end|>, no trailing \\n).
        """
        msgs_0 = make_messages("Sys.", ["Hi"])
        msgs_1 = make_messages("Sys.", ["Hi", "Hello", "Obs"])
        responses = ["Hello", "Bye"]
        steps = build_steps_realistic(tokenizer, parser, [msgs_0, msgs_1], responses)

        # Verify completion_ids end with <|im_end|> but not \n
        for s in steps:
            assert s["completion_ids"][-1] == IM_END_ID
            assert len(s["completion_ids"]) >= 2  # at least 1 content token + eot

        pt, rt, rm, v = engine.assemble_steps(steps)

        assert v is True
        assert len(rt) == len(rm)

        # Full text should be consistent
        decoded = tokenizer.decode(pt.tolist() + rt.tolist(), skip_special_tokens=False)
        expected = steps[-1]["prompt"] + tokenizer.decode(
            steps[-1]["completion_ids"], skip_special_tokens=False
        )
        assert decoded == expected

    def test_realistic_eot_in_mask(self, engine, tokenizer, parser):
        """
        With realistic completion_ids, <|im_end|> should still be mask=1
        for non-last steps (included via eot_token span extension).
        """
        msgs_0 = make_messages("Sys.", ["Start"])
        msgs_1 = make_messages("Sys.", ["Start", "Done.", "Next"])
        responses = ["Done.", "End."]
        steps = build_steps_realistic(tokenizer, parser, [msgs_0, msgs_1], responses)

        pt, rt, rm, _ = engine.assemble_steps(steps)

        comp_text = tokenizer.decode(rt[rm == 1].tolist(), skip_special_tokens=False)
        # First completion should have <|im_end|> in mask
        eot_str = "<" + "|im_end|" + ">"
        assert eot_str in comp_text, (
            f"Expected {eot_str} in mask=1 text: {repr(comp_text)}"
        )

    def test_realistic_text_consistency(self, engine, tokenizer, parser):
        """
        Verify decoded mask=1 text contains all response texts,
        using realistic completion_ids.
        """
        msgs_0 = make_messages("Sys.", ["Task"])
        msgs_1 = make_messages("Sys.", ["Task", "Step 1", "Result 1"])
        msgs_2 = make_messages("Sys.", ["Task", "Step 1", "Result 1", "Step 2", "Result 2"])
        responses = ["Step 1", "Step 2", "Final step"]
        steps = build_steps_realistic(tokenizer, parser,
                                       [msgs_0, msgs_1, msgs_2], responses)

        pt, rt, rm, v = engine.assemble_steps(steps)

        comp_text = tokenizer.decode(rt[rm == 1].tolist(), skip_special_tokens=True)
        for resp in responses:
            assert resp in comp_text, (
                f"Missing '{resp}' in completion text: {repr(comp_text)}"
            )
