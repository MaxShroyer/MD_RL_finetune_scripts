from __future__ import annotations

import os
import unittest
from unittest import mock

from tictaktoe_QA import check_moondream_staging_api as mod


class CheckMoondreamStagingApiTests(unittest.TestCase):
    def test_parse_args_defaults_to_staging(self) -> None:
        args = mod._parse_args([])
        self.assertEqual(args.base_url, mod.DEFAULT_STAGING_BASE_URL)
        self.assertEqual(args.api_key_env_var, mod.DEFAULT_API_KEY_ENV_VAR)
        self.assertEqual(args.model, mod.DEFAULT_MODEL)
        self.assertIsNone(args.reasoning)

    def test_build_query_payload_omits_reasoning_when_none(self) -> None:
        payload = mod._build_query_payload(
            model="moondream3-preview",
            question="q",
            image_url="data:image/jpeg;base64,abc",
            temperature=0.0,
            top_p=1.0,
            max_tokens=64,
            reasoning=None,
        )
        self.assertNotIn("reasoning", payload)

    def test_build_query_payload_includes_reasoning_when_false(self) -> None:
        payload = mod._build_query_payload(
            model="moondream3-preview",
            question="q",
            image_url="data:image/jpeg;base64,abc",
            temperature=0.0,
            top_p=1.0,
            max_tokens=64,
            reasoning=False,
        )
        self.assertIn("reasoning", payload)
        self.assertFalse(payload["reasoning"])

    def test_resolve_api_key_prefers_explicit_value(self) -> None:
        with mock.patch.dict(os.environ, {"CICID_GPUB_MOONDREAM_API_KEY_1": "env-key"}, clear=False):
            api_key, source = mod._resolve_api_key("cli-key", mod.DEFAULT_API_KEY_ENV_VAR)
        self.assertEqual(api_key, "cli-key")
        self.assertEqual(source, "cli")

    def test_resolve_api_key_uses_named_env_var_then_fallbacks(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "CICID_GPUB_MOONDREAM_API_KEY_1": "",
                "MOONDREAM_API_KEY": "fallback-key",
            },
            clear=False,
        ):
            api_key, source = mod._resolve_api_key("", mod.DEFAULT_API_KEY_ENV_VAR)
        self.assertEqual(api_key, "fallback-key")
        self.assertEqual(source, "MOONDREAM_API_KEY")

    def test_build_auth_headers_wraps_authorization_with_bearer(self) -> None:
        headers = mod._build_auth_headers("secret", auth_header="Authorization")
        self.assertEqual(headers["Authorization"], "Bearer secret")

    def test_extract_answer_text_supports_nested_output_shape(self) -> None:
        answer = mod._extract_answer_text({"output": {"answer": "nested"}})
        self.assertEqual(answer, "nested")


if __name__ == "__main__":
    unittest.main()
