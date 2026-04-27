from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock

from tuna_sdk import (
    AsyncTunaClient,
    DetectAnnotation,
    DetectRequest,
    DetectSFTTarget,
    PointRequest,
    PointSFTTarget,
    QueryOutput,
    QueryRequest,
    QuerySFTTarget,
    Rollout,
    RolloutsRequest,
    RolloutsResult,
    TunaClient,
    TrainStepGroup,
    TrainStepResponse,
)
from tuna_sdk.client import _RequestOptions as SyncRequestOptions


class TunaSdkSFTTests(unittest.TestCase):
    def test_tuna_client_preserves_default_timeout_when_no_override_is_set(self) -> None:
        client = TunaClient(api_key="test", base_url="https://example.com", timeout=12.5)

        class _Response:
            status_code = 200
            is_success = True

            def json(self) -> dict[str, bool]:
                return {"ok": True}

        request_mock = Mock(return_value=_Response())
        client._client.request = request_mock  # type: ignore[method-assign]
        payload = client._request_json("GET", "/health")

        self.assertEqual(payload, {"ok": True})
        self.assertNotIn("timeout", request_mock.call_args.kwargs)
        client.close()

    def test_tuna_client_passes_explicit_timeout_override(self) -> None:
        client = TunaClient(api_key="test", base_url="https://example.com", timeout=12.5)

        class _Response:
            status_code = 200
            is_success = True

            def json(self) -> dict[str, bool]:
                return {"ok": True}

        request_mock = Mock(return_value=_Response())
        client._client.request = request_mock  # type: ignore[method-assign]
        payload = client._request_json(
            "GET",
            "/health",
            options=SyncRequestOptions(timeout=3.0),
        )

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(request_mock.call_args.kwargs.get("timeout"), 3.0)
        client.close()

    def test_async_tuna_client_preserves_default_timeout_when_no_override_is_set(self) -> None:
        async def run() -> None:
            client = AsyncTunaClient(api_key="test", base_url="https://example.com", timeout=12.5)

            class _Response:
                status_code = 200
                is_success = True

                def json(self) -> dict[str, bool]:
                    return {"ok": True}

            request_mock = AsyncMock(return_value=_Response())
            client._client.request = request_mock  # type: ignore[method-assign]
            payload = await client._request_json("GET", "/health")

            self.assertEqual(payload, {"ok": True})
            self.assertNotIn("timeout", request_mock.call_args.kwargs)
            await client.close()

        asyncio.run(run())

    def test_rl_train_step_group_round_trip_preserves_mode(self) -> None:
        request = QueryRequest(question="What is the best move?", reasoning=False)
        rollouts_request = RolloutsRequest(
            finetune_id="ft_test",
            num_rollouts=1,
            request=request,
        )
        rollout = Rollout(
            skill="query",
            finish_reason="stop",
            output=QueryOutput(answer='{"row":1,"col":1}'),
        )
        group = TrainStepGroup.from_rl(
            request=rollouts_request,
            rollouts=[rollout],
            rewards=[0.75],
        )

        payload = group.to_payload()
        round_trip = TrainStepGroup.from_payload(payload)

        self.assertEqual(payload["mode"], "rl")
        self.assertEqual(round_trip.mode, "rl")
        self.assertIsInstance(round_trip.request, RolloutsRequest)
        self.assertEqual(round_trip.rewards, [0.75])

    def test_rollouts_result_to_group_builds_rl_group(self) -> None:
        request = QueryRequest(question="Count moves")
        rollouts_request = RolloutsRequest(
            finetune_id="ft_test",
            num_rollouts=1,
            request=request,
        )
        result = RolloutsResult(
            request=rollouts_request,
            rollouts=[
                Rollout(
                    skill="query",
                    finish_reason="stop",
                    output=QueryOutput(answer='{"available_move_count":1}'),
                )
            ],
        )

        group = result.to_group(rewards=[1.0])

        self.assertEqual(group.mode, "rl")
        self.assertIsInstance(group.request, RolloutsRequest)
        self.assertEqual(group.rewards, [1.0])

    def test_query_sft_group_round_trip_preserves_reasoning(self) -> None:
        request = QueryRequest(question="What is the best move?", reasoning=True)
        group = TrainStepGroup.from_sft(
            request=request,
            targets=[
                QuerySFTTarget(
                    answer='{"col":1,"row":1}',
                    reasoning="Center creates the fork.",
                )
            ],
        )

        payload = group.to_payload()
        round_trip = TrainStepGroup.from_payload(payload)

        self.assertEqual(payload["mode"], "sft")
        self.assertIn("target", payload)
        self.assertNotIn("targets", payload)
        self.assertEqual(round_trip.mode, "sft")
        self.assertIsInstance(round_trip.request, QueryRequest)
        self.assertEqual(len(round_trip.targets), 1)
        self.assertIsInstance(round_trip.targets[0], QuerySFTTarget)
        self.assertEqual(round_trip.targets[0].reasoning, "Center creates the fork.")

    def test_point_and_detect_sft_groups_round_trip_boxes(self) -> None:
        box = DetectAnnotation(x_min=0.1, y_min=0.2, x_max=0.3, y_max=0.4)

        point_group = TrainStepGroup.from_sft(
            request=PointRequest(object_name="valve", image_url="data:image/jpeg;base64,abc"),
            targets=[PointSFTTarget(boxes=[box])],
        )
        detect_group = TrainStepGroup.from_sft(
            request=DetectRequest(object_name="valve", image_url="data:image/jpeg;base64,abc"),
            targets=[DetectSFTTarget(boxes=[box])],
        )

        point_round_trip = TrainStepGroup.from_payload(point_group.to_payload())
        detect_round_trip = TrainStepGroup.from_payload(detect_group.to_payload())

        self.assertIn("target", point_group.to_payload())
        self.assertIn("target", detect_group.to_payload())
        self.assertIsInstance(point_round_trip.targets[0], PointSFTTarget)
        self.assertEqual(point_round_trip.targets[0].boxes, [box])
        self.assertIsInstance(detect_round_trip.targets[0], DetectSFTTarget)
        self.assertEqual(detect_round_trip.targets[0].boxes, [box])

    def test_point_and_detect_sft_groups_parse_legacy_targets_payload(self) -> None:
        box = DetectAnnotation(x_min=0.1, y_min=0.2, x_max=0.3, y_max=0.4)

        point_round_trip = TrainStepGroup.from_payload(
            {
                "mode": "sft",
                "request": PointRequest(object_name="valve", image_url="data:image/jpeg;base64,abc").to_payload(),
                "targets": [PointSFTTarget(boxes=[box]).to_payload()],
            }
        )
        detect_round_trip = TrainStepGroup.from_payload(
            {
                "mode": "sft",
                "request": DetectRequest(object_name="valve", image_url="data:image/jpeg;base64,abc").to_payload(),
                "targets": [DetectSFTTarget(boxes=[box]).to_payload()],
            }
        )

        self.assertIsInstance(point_round_trip.targets[0], PointSFTTarget)
        self.assertEqual(point_round_trip.targets[0].boxes, [box])
        self.assertIsInstance(detect_round_trip.targets[0], DetectSFTTarget)
        self.assertEqual(detect_round_trip.targets[0].boxes, [box])

    def test_train_step_response_parses_extended_fields(self) -> None:
        response = TrainStepResponse.from_payload(
            {
                "step": {"value": "7"},
                "applied": {"success": "true"},
                "kl": 0.12,
                "router_kl": 0.03,
                "grad_norm": 1.5,
                "reward_mean": 0.81,
                "reward_std": 0.09,
                "sft_loss": 0.44,
            }
        )

        self.assertEqual(response.step, 7)
        self.assertTrue(response.applied)
        self.assertAlmostEqual(float(response.reward_mean), 0.81, places=6)
        self.assertAlmostEqual(float(response.reward_std), 0.09, places=6)
        self.assertAlmostEqual(float(response.sft_loss), 0.44, places=6)


if __name__ == "__main__":
    unittest.main()
