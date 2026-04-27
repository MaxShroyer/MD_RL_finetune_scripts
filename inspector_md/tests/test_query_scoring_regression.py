from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspector_md import common
from inspector_md import moondream_client
from inspector_md import train_inspector_query as train_query_mod
from tuna_sdk import QueryRequest, QuerySettings


class QueryScoringRegressionTests(unittest.TestCase):
    def test_finding_issue_match_and_task_correct_are_scored_separately(self) -> None:
        example = train_query_mod.QueryExample(
            row_id="row-1",
            split="validation",
            task_type="finding",
            image_path=Path("unused.jpg"),
            inspection_request="Inspect the highlighted region.",
            asset_context="Synthetic exterior image.",
            spatial_refs=[[0.0, 0.0, 1.0, 1.0]],
            question="For the highlighted region only: Inspect this image for visible building or site issues.",
            target_text="crack_defect | Crack Defect | moderate | false | Visible cracks in the annotated area. | Inspect and repair cracked surfaces as needed.",
            target_format="compact_text",
            final_answer_json=json.dumps(
                {
                    "finding": {
                        "issue_code": "crack_defect",
                        "title": "Crack Defect",
                        "evidence": ["Visible cracks in the annotated area."],
                        "severity": "moderate",
                        "recommended_action": "Inspect and repair cracked surfaces as needed.",
                        "insufficient_evidence": False,
                    }
                }
            ),
            reasoning_text="",
            hard_example=False,
            query_text_refresh_mode="openrouter",
            issue_count=1,
            is_multi_issue=False,
            source_dataset="synthetic",
            source_annotation_type="finding",
            crop_derived=False,
        )

        score, parse = train_query_mod._score_answer_text(
            example,
            "crack_defect | Crack Defect | moderate | false | Visible evidence consistent with crack defect is present in the annotated region. | Inspect the crack pattern and repair the cracked surface where appropriate.",
            grader=None,
        )

        self.assertEqual(parse.method, "compact_text")
        self.assertEqual(score.issue_f1, 1.0)
        self.assertLess(score.reasoning_f1, 0.999)
        self.assertFalse(score.task_correct)

    def test_build_auth_headers_uses_api_style_user_agent_by_default(self) -> None:
        headers = common.build_auth_headers("secret")
        self.assertEqual(headers["X-Moondream-Auth"], "secret")
        self.assertEqual(headers["User-Agent"], "inspector-md-client/0.1")

    def test_query_raw_omits_base_model_parameter(self) -> None:
        captured: dict[str, object] = {}

        def fake_post_json(**kwargs):
            captured["payloads"] = kwargs["payloads"]
            return {"answer": "ok"}, 5.0

        pool = common.ApiKeyPool([common.ApiKeySlot(index=0, env_var="<test>", api_key="secret")])
        client = moondream_client.MoondreamInspectorClient(
            api_key_pool=pool,
            base_url="https://api-staging.moondream.ai/v1",
            timeout=5.0,
            max_retries=0,
            post_json=fake_post_json,
        )
        request = QueryRequest(
            question="What is in this image?",
            image_url="data:image/jpeg;base64,abc",
            settings=QuerySettings(temperature=0.0, top_p=1.0, max_tokens=16),
        )

        client.query_raw(model=common.DEFAULT_BASE_MODEL, request=request)

        payloads = captured["payloads"]
        self.assertIsInstance(payloads, list)
        self.assertEqual(len(payloads), 1)
        self.assertNotIn("model", payloads[0])


if __name__ == "__main__":
    unittest.main()
