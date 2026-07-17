"""Run and rank multiple fabrication strategy candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from fabrication.context import FabricationContext, build_fabrication_context
from fabrication.contracts import FabricationConfig, FabricationPlan
from fabrication.scoring import add_basic_score
from fabrication.strategies.planar_skin import PlanarSkinStrategy


def available_strategies() -> Dict[str, object]:
    strategies: Dict[str, object] = {
        PlanarSkinStrategy.strategy_id: PlanarSkinStrategy(),
    }

    try:
        from fabrication.strategies.contour_stack import ContourStackStrategy

        strategies[ContourStackStrategy.strategy_id] = ContourStackStrategy()
    except ImportError:
        pass

    try:
        from fabrication.strategies.waffle_ribs import WaffleRibsStrategy

        strategies[WaffleRibsStrategy.strategy_id] = WaffleRibsStrategy()
    except ImportError:
        pass

    try:
        from fabrication.strategies.voxel_blocks import VoxelBlocksStrategy

        strategies[VoxelBlocksStrategy.strategy_id] = VoxelBlocksStrategy()
    except ImportError:
        pass

    return strategies


def run_tournament(
    config: FabricationConfig,
    artifacts_dir: Path | None = None,
    strategy_ids: Iterable[str] | None = None,
) -> Dict[str, object]:
    context = build_fabrication_context(config)
    registry = available_strategies()
    requested = tuple(strategy_ids or config.strategies)
    plans: List[FabricationPlan] = []

    for strategy_id in requested:
        strategy = registry.get(strategy_id)
        if strategy is None:
            plans.append(
                FabricationPlan(
                    strategy_id=strategy_id,
                    status="error",
                    warnings=[f"Strategy is not registered: {strategy_id}"],
                    scores={"overall": 0.0},
                )
            )
            continue

        try:
            plan = strategy.generate(context, artifacts_dir=artifacts_dir)
            add_basic_score(plan, context)
        except Exception as exc:  # pragma: no cover - exercised by CLI failures
            plan = FabricationPlan(
                strategy_id=strategy_id,
                status="error",
                warnings=[f"{type(exc).__name__}: {exc}"],
                scores={"overall": 0.0},
                debug={"exception_type": type(exc).__name__},
            )
        plans.append(plan)

    ranked = sorted(
        plans,
        key=lambda plan: (
            plan.status == "error",
            -plan.overall_score,
            len(plan.parts),
            plan.strategy_id,
        ),
    )
    return {
        "schema_version": "fabrication.tournament.v0",
        "context": context.summary_payload(),
        "requested_strategies": list(requested),
        "available_strategies": sorted(registry.keys()),
        "ranking": [
            {
                "rank": index + 1,
                "strategy_id": plan.strategy_id,
                "status": plan.status,
                "overall_score": plan.overall_score,
                "part_count": len(plan.parts),
                "warning_count": len(plan.warnings),
                "scores": plan.scores,
            }
            for index, plan in enumerate(ranked)
        ],
        "plans": {plan.strategy_id: plan for plan in plans},
    }


def write_tournament_artifacts(result: Dict[str, object], artifacts_dir: Path) -> None:
    tournament_dir = Path(artifacts_dir)
    tournament_dir.mkdir(parents=True, exist_ok=True)
    plans = result["plans"]
    assert isinstance(plans, dict)

    for strategy_id, plan in plans.items():
        assert isinstance(plan, FabricationPlan)
        strategy_dir = tournament_dir / strategy_id
        strategy_dir.mkdir(parents=True, exist_ok=True)
        _write_json(strategy_dir / "plan.json", plan.to_payload())

    ranking_payload = {key: value for key, value in result.items() if key != "plans"}
    _write_json(tournament_dir / "ranking.json", ranking_payload)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
