from unittest import TestCase

from serialization.ai_response_tokenizer import AIResponseTypedToken
from src.backend.src.serialization.ai_response_tokenizer import (
    AIResponseTypedTokenizer,
    AIResponseToken,
    UnspecifiedAIResponseToken,
    UnspecifiedAIResponseTypedToken,
    TypedTokenizationError,
    TypedAIResponse, TypedToken,
)


class TestAIResponseTokenizer(TestCase):
    def setUp(self):
        self.tokenizer = AIResponseTypedTokenizer()

    def test_get_prompt_includes_all_token_specs(self):
        prompt = self.tokenizer.get_prompt()

        self.assertIn("PLAN: ... ", prompt)
        self.assertIn("STATE: ... ", prompt)
        self.assertIn("NOTE: ... (if any)", prompt)

    def test_find_spec_recognizes_correct_spec(self):
        # Testing recognized spec lines
        self.assertEqual(
            self.tokenizer.find_spec("PLAN: do something").name, "plan"
        )
        self.assertEqual(
            self.tokenizer.find_spec("STATE: some state info").name, "state"
        )
        self.assertEqual(
            self.tokenizer.find_spec("NOTE: optional note").name, "note"
        )

    def test_find_spec_returns_unspecified_for_unmatched_line(self):
        spec = self.tokenizer.find_spec("This line doesn't start with a known token:")
        self.assertIsInstance(spec, UnspecifiedAIResponseToken)

    def test_find_spec_raises_tokenization_error_for_unknown_and_no_unspecified(self):
        # Create a tokenizer without an unspecified token spec
        tokenizer = AIResponseTypedTokenizer()
        tokenizer.token_specs = [
            AIResponseToken("plan", False),
        ]
        with self.assertRaises(TypedTokenizationError):
            tokenizer.find_spec("unknownline: data")

    def test_tokenize_line_returns_token_with_correct_name_and_value(self):
        line = "PLAN: Execute step 1"
        token = self.tokenizer.tokenize_line(line)
        self.assertEqual(token.get_name(), "plan")
        self.assertTrue(token.get_value().startswith("PLAN: Execute step 1"))

    def test_tokenize_splits_lines_and_tokenizes_each(self):
        raw_response = "PLAN: step 1\nSTATE: ready\nNOTE: optional info\nMISC: extra"
        tokenized = self.tokenizer.tokenize(raw_response)
        self.assertIsInstance(tokenized, TypedAIResponse)
        self.assertEqual(len(tokenized.typed_tokens), 4)
        names = [token.get_name() for token in tokenized.typed_tokens]
        self.assertIn("plan", names)
        self.assertIn("state", names)
        self.assertIn("note", names)
        self.assertIn(None, names)  # for unspecified token, name is None

    def test_AIResponseTokenSpecification_accepts_method(self):
        spec = AIResponseToken("test", False)
        self.assertTrue(spec.accepts("TEST: some value"))
        self.assertFalse(spec.accepts("NOTTEST: some value"))

    def test_AIResponseTokenSpecification_as_partial_prompt_and_readable_line(self):
        spec = AIResponseToken("test", optional=True)
        # partial prompt includes (if any)
        partial_prompt = spec.as_partial_prompt()
        self.assertIn("(if any)", partial_prompt)
        # readable line includes token's value plus " + \n"
        class DummyToken(TypedToken):
            def get_value(self):
                return "value"

        line = spec.as_readable_line(DummyToken())
        self.assertIn("TEST:", line)
        self.assertIn("value", line)
        self.assertIn("+ \n", line)

    def test_UnspecifiedAIResponseTokenSpecification_singleton_and_methods(self):
        spec1 = UnspecifiedAIResponseToken()
        spec2 = UnspecifiedAIResponseToken()
        self.assertIs(spec1, spec2)

        partial_prompt = spec1.as_partial_prompt()
        self.assertEqual(partial_prompt, "")

        class DummyToken(TypedToken):
            def get_value(self):
                return "some value"

        token = DummyToken()
        readable = spec1.as_readable_line(token)
        self.assertEqual(readable, "some value\n")

        token_obj = spec1.tokenize_line("random line")
        self.assertIsInstance(token_obj, UnspecifiedAIResponseTypedToken)

        self.assertTrue(spec1.accepts("anything"))

    def test_AIResponseToken_getters_and_value_formatting(self):
        spec = AIResponseToken("example")
        token = AIResponseTypedToken(spec, "line content")
        self.assertEqual(token.get_name(), "example")
        self.assertEqual(token.get_value(), "line content\n")

    def test_UnspecifiedAIResponseToken_str_and_get_value(self):
        spec = UnspecifiedAIResponseToken()
        token = UnspecifiedAIResponseTypedToken(spec, "value line")
        self.assertEqual(token.get_value(), "value line\n")
        self.assertIn("value line\n", str(token))

    def test_TokenizedAIResponse_add_token_and_tokens_list(self):
        tokenized = TypedAIResponse()
        spec = AIResponseToken("test")
        token = AIResponseTypedToken(spec, "val")
        tokenized.add_typed_token(token)
        self.assertIn(token, tokenized.typed_tokens)