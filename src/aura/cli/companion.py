"""Aura CLI Companion — terminal-based conversational interface.

Phase 1 implementation per PRD v2.2 §19 Demo Spec:
"Aura companion running as a CLI/terminal interface."

This is the user-facing front end. It:
1. Accepts natural language input from the trader
2. Processes messages through ConversationProcessor for emotional signals
3. Updates the readiness score via ReadinessComputer
4. Writes the readiness signal for Buddy to read via the bridge
5. Shows bridge status (Buddy's latest outcomes, override history)

Phase 2 adds: Phi-4 14B responses, self-model graph queries, pattern insights.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.aura.core.conversation_processor import ConversationProcessor, ConversationSignals
from src.aura.core.readiness import ReadinessComputer, ReadinessSignal
from src.aura.core.self_model import (
    SelfModelGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
)
from src.aura.bridge.signals import FeedbackBridge, OutcomeSignal
from src.aura.patterns.engine import PatternEngine

# Phase 3 prediction models (lazy import — may not be available)
try:
    from src.aura.prediction.override_predictor import OverridePredictor
    from src.aura.prediction.readiness_v2 import ReadinessModelV2
    _HAS_PREDICTION = True
except ImportError:
    _HAS_PREDICTION = False

logger = logging.getLogger(__name__)

# ANSI colors for terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


class AuraCompanion:
    """The Aura CLI companion — conversational interface + readiness engine.

    Args:
        db_path: Path to self-model database
        bridge_dir: Path to bridge signal directory
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        bridge_dir: Optional[Path] = None,
    ):
        self.processor = ConversationProcessor()
        self.readiness = ReadinessComputer(
            signal_path=(bridge_dir or Path(".aura/bridge")) / "readiness_signal.json"
        )
        self.graph = SelfModelGraph(db_path=db_path)
        self.bridge = FeedbackBridge(bridge_dir=bridge_dir)
        self.pattern_engine = PatternEngine(
            patterns_dir=(bridge_dir or Path(".aura/bridge")).parent / "patterns",
            bridge_dir=bridge_dir,
        )
        self._conversation_id = f"conv_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self._active_stressors: List[str] = []
        self._latest_signals: Optional[ConversationSignals] = None
        self._latest_readiness: Optional[ReadinessSignal] = None
        self._message_history: List[Dict[str, str]] = []

        # Phase 3: Prediction models
        self._override_predictor: Optional[Any] = None
        self._readiness_v2: Optional[Any] = None
        if _HAS_PREDICTION:
            try:
                models_dir = (bridge_dir or Path(".aura/bridge")).parent / "models"
                self._override_predictor = OverridePredictor(
                    model_path=models_dir / "override_predictor.json"
                )
                self._readiness_v2 = ReadinessModelV2(
                    model_path=models_dir / "readiness_v2.json"
                )
            except Exception as e:
                logger.debug(f"Prediction models not available: {e}")

    def _needs_onboarding(self) -> bool:
        """Check if self-model is empty and needs onboarding."""
        stats = self.graph.get_stats()
        # If no Person or Goal nodes exist, we haven't onboarded
        nodes_by_type = stats.get("nodes_by_type", {})
        # Keys are lowercase enum values (e.g., "person", "goal")
        has_person = nodes_by_type.get("person", 0) > 0
        has_goals = nodes_by_type.get("goal", 0) > 0
        return not (has_person and has_goals)

    def run_onboarding(self) -> None:
        """Run first-time onboarding to seed the self-model graph.

        PRD v2.2 §13 Phase 2: "Companion onboarding flow that seeds self-model"

        Asks a structured set of questions to populate:
        - Person node (trader identity)
        - Goal nodes (trading goals, life goals)
        - Value nodes (what matters to the trader)
        - TradingState node (current experience level, style)

        All responses are processed through ConversationProcessor for
        emotional signal extraction even during onboarding.
        """
        print(f"\n{CYAN}{BOLD}╔══════════════════════════════════════════╗{RESET}")
        print(f"{CYAN}{BOLD}║       Welcome to Aura — First Setup      ║{RESET}")
        print(f"{CYAN}{BOLD}╚══════════════════════════════════════════╝{RESET}")
        print()
        print(f"I'm Aura, your self-awareness companion. I work alongside Buddy")
        print(f"to help you trade better by understanding yourself better.")
        print()
        print(f"Let me get to know you a bit. This takes about 2 minutes.")
        print(f"{DIM}(You can skip any question by pressing Enter){RESET}")
        print()

        onboarding_data: Dict[str, Any] = {}

        # --- Q1: Name / identity ---
        name = input(f"{CYAN}What should I call you? {RESET}").strip()
        if name:
            onboarding_data["name"] = name
            self.graph.add_node(GraphNode(
                id="person_trader",
                node_type=NodeType.PERSON,
                label=name,
                properties={"role": "trader", "onboarded": True},
                confidence=1.0,
            ))
            print(f"  Nice to meet you, {name}.\n")

        # --- Q2: Trading experience ---
        experience = input(
            f"{CYAN}How would you describe your trading experience? "
            f"{DIM}(beginner / intermediate / experienced){RESET} "
        ).strip().lower()
        if experience:
            onboarding_data["experience"] = experience
            self.graph.add_node(GraphNode(
                id="trading_state_baseline",
                node_type=NodeType.TRADING_STATE,
                label=f"Trader experience: {experience}",
                properties={
                    "experience_level": experience,
                    "onboarding": True,
                },
            ))

        # --- Q3: Trading goals ---
        print(f"\n{CYAN}What are your main trading goals?{RESET}")
        goals_raw = input(
            f"{DIM}(e.g., consistent income, grow capital, learn the craft, "
            f"financial independence){RESET}\n> "
        ).strip()
        if goals_raw:
            # Process for emotional signals
            signals = self.processor.process_message(goals_raw, role="user")
            goals = [g.strip() for g in goals_raw.replace(",", "\n").split("\n") if g.strip()]
            for i, goal in enumerate(goals):
                goal_id = f"goal_{i}_{goal.lower().replace(' ', '_')[:30]}"
                self.graph.add_node(GraphNode(
                    id=goal_id,
                    node_type=NodeType.GOAL,
                    label=goal,
                    properties={"priority": i + 1, "source": "onboarding"},
                    confidence=0.8,
                ))
                # Link person → goal
                if name:
                    self.graph.add_edge(GraphEdge(
                        source_id="person_trader",
                        target_id=goal_id,
                        edge_type=EdgeType.INFLUENCES,
                        properties={"relationship": "pursues"},
                    ))
            onboarding_data["goals"] = goals
            print(f"  Noted — {len(goals)} goal{'s' if len(goals) != 1 else ''} recorded.\n")

        # --- Q4: What matters most (values) ---
        values_raw = input(
            f"{CYAN}What matters most to you beyond trading? "
            f"{DIM}(family, health, career, relationships, hobbies?){RESET}\n> "
        ).strip()
        if values_raw:
            signals = self.processor.process_message(values_raw, role="user")
            values = [v.strip() for v in values_raw.replace(",", "\n").split("\n") if v.strip()]
            for i, value in enumerate(values):
                value_id = f"value_{value.lower().replace(' ', '_')[:30]}"
                self.graph.add_node(GraphNode(
                    id=value_id,
                    node_type=NodeType.VALUE,
                    label=value,
                    properties={"priority": i + 1, "source": "onboarding"},
                    confidence=0.7,
                ))
                if name:
                    self.graph.add_edge(GraphEdge(
                        source_id="person_trader",
                        target_id=value_id,
                        edge_type=EdgeType.INFLUENCES,
                        properties={"relationship": "values"},
                    ))
            onboarding_data["values"] = values
            print(f"  Got it — these help me understand what else is on your plate.\n")

        # --- Q5: Known stressors ---
        stressors_raw = input(
            f"{CYAN}Any current stressors I should be aware of? "
            f"{DIM}(work deadlines, health, financial pressure, relationship?){RESET}\n> "
        ).strip()
        if stressors_raw:
            signals = self.processor.process_message(stressors_raw, role="user")
            stressors = [s.strip() for s in stressors_raw.replace(",", "\n").split("\n") if s.strip()]
            for stressor in stressors:
                stressor_id = f"stressor_{stressor.lower().replace(' ', '_')[:30]}"
                self.graph.add_node(GraphNode(
                    id=stressor_id,
                    node_type=NodeType.LIFE_EVENT,
                    label=stressor,
                    properties={"active": True, "source": "onboarding"},
                ))
                self._active_stressors.append(stressor)
            onboarding_data["stressors"] = stressors
            print(f"  I'll keep these in mind — they affect readiness even subtly.\n")

        # --- Q6: Trading style (helps Buddy calibrate) ---
        style = input(
            f"{CYAN}How would you describe your trading style? "
            f"{DIM}(scalper / swing / position / day-trader / mixed){RESET} "
        ).strip().lower()
        if style:
            onboarding_data["trading_style"] = style
            # Update trading state node
            self.graph.add_node(GraphNode(
                id="trading_state_style",
                node_type=NodeType.TRADING_STATE,
                label=f"Trading style: {style}",
                properties={"style": style, "source": "onboarding"},
            ))

        # --- Wrap up ---
        # Write onboarding data to bridge for Buddy to see
        onboarding_path = (self.bridge.bridge_dir / "onboarding_profile.json")
        try:
            onboarding_path.write_text(
                json.dumps(onboarding_data, indent=2, default=str)
            )
        except Exception as e:
            logger.warning(f"Failed to write onboarding profile: {e}")

        # Compute initial readiness baseline
        self._update_readiness()

        print(f"\n{GREEN}{BOLD}Onboarding complete!{RESET}")
        print(f"Your self-model has been seeded with {self.graph.get_stats()['total_nodes']} nodes.")
        print(f"The more we talk, the better I understand your patterns.\n")

    def start_session(self) -> None:
        """Initialize a new conversation session."""
        # Check if first-time onboarding is needed
        if self._needs_onboarding():
            self.run_onboarding()

        # US-209: Validate onboarding actually completed (user may have Ctrl-C'd)
        if self._needs_onboarding():
            logger.warning(
                "US-209: Onboarding incomplete — self-model missing Person or Goal nodes. "
                "Readiness will default to 50 (unknown state)."
            )
            print(f"\n{YELLOW}Note: Onboarding wasn't completed. "
                  f"Readiness will default to neutral (50/100) until onboarding finishes.{RESET}")
            print(f"{DIM}Run /onboard to restart onboarding.{RESET}\n")

        print(f"\n{CYAN}{BOLD}╔══════════════════════════════════════════╗{RESET}")
        print(f"{CYAN}{BOLD}║          AURA — Human Engine             ║{RESET}")
        print(f"{CYAN}{BOLD}║   Recursive Self-Improving Companion     ║{RESET}")
        print(f"{CYAN}{BOLD}╚══════════════════════════════════════════╝{RESET}")
        print()
        print(f"{DIM}Session: {self._conversation_id}{RESET}")
        print(f"{DIM}Commands: /status /bridge /readiness /graph /patterns /validate /rules /predict /quit{RESET}")
        print()

        # Show bridge status if Buddy is running
        bridge_status = self.bridge.get_bridge_status()
        if bridge_status["outcome_signal"]["available"]:
            outcome = self.bridge.read_outcome()
            if outcome:
                streak_color = GREEN if outcome.streak == "winning" else (RED if outcome.streak == "losing" else YELLOW)
                print(f"{DIM}Buddy is active — PnL today: {outcome.pnl_today:+.2f}, "
                      f"regime: {outcome.regime}, streak: {streak_color}{outcome.streak}{RESET}{RESET}")
        else:
            print(f"{DIM}Buddy not detected — readiness signals will be cached for when it starts{RESET}")

        # Compute initial readiness
        self._update_readiness()
        print()

    def process_input(self, user_input: str) -> str:
        """Process user input and return Aura's response.

        Args:
            user_input: The trader's message

        Returns:
            Aura's response text
        """
        # Handle commands
        if user_input.startswith("/"):
            return self._handle_command(user_input)

        # Process through conversation processor
        signals = self.processor.process_message(user_input, role="user")
        self._latest_signals = signals

        # Update active stressors
        for stressor in signals.detected_stressors:
            if stressor not in self._active_stressors:
                self._active_stressors.append(stressor)

        # Log to self-model graph
        self._log_to_graph(user_input, signals)

        # Update readiness score
        readiness = self._update_readiness(signals)

        # Run T1 pattern detection after each message
        self._run_pattern_detection(signals)

        # Generate response (Phase 1: rule-based; Phase 2: Phi-4 14B)
        response = self._generate_response(user_input, signals, readiness)

        # Log assistant response
        self.processor.process_message(response, role="assistant")
        self._message_history.append({"role": "user", "content": user_input})
        self._message_history.append({"role": "assistant", "content": response})

        return response

    def _generate_response(
        self,
        user_input: str,
        signals: ConversationSignals,
        readiness: ReadinessSignal,
    ) -> str:
        """Generate Aura's response based on conversation signals.

        Phase 1: Rule-based responses that demonstrate the value.
        Phase 2: Phi-4 14B via MLX for deep, contextual responses.
        """
        parts: List[str] = []

        # Acknowledge emotional state if notable
        if signals.emotional_state == "stressed":
            parts.append(
                "I can hear that you're dealing with a lot right now. "
                "Let's talk through it."
            )
        elif signals.emotional_state == "fatigued":
            parts.append(
                "It sounds like you're running low on energy. "
                "That matters for your decision-making today."
            )
        elif signals.emotional_state == "anxious":
            parts.append(
                "I'm picking up some tension in what you're sharing. "
                "That's worth being aware of."
            )
        elif signals.emotional_state == "energized":
            parts.append(
                "You sound like you're in a good headspace. That's great."
            )

        # Override detection
        if signals.override_mentioned:
            recent_overrides = self.bridge.get_recent_overrides(limit=10)
            losing_overrides = [o for o in recent_overrides if o.outcome == "loss"]
            if len(losing_overrides) >= 2:
                parts.append(
                    f"I notice you're talking about overriding Buddy's signals. "
                    f"Over the last few sessions, {len(losing_overrides)} of your "
                    f"overrides resulted in losses. Let's think about what's driving "
                    f"that impulse right now."
                )
            else:
                parts.append(
                    "You mentioned overriding Buddy. I'll log that — understanding "
                    "when and why you override helps us both learn."
                )

            # Phase 3: Override prediction warning
            if self._override_predictor:
                try:
                    context = {
                        "emotional_state": signals.emotional_state,
                        "cognitive_load": "high" if len(signals.stress_keywords_found) > 2 else "normal",
                        "confidence_at_time": 0.5,  # Unknown — use neutral
                        "weighted_vote_at_time": 0.5,
                        "override_type": "took_rejected",
                        "regime": "NORMAL",
                    }
                    prediction = self._override_predictor.predict_loss_probability(context)
                    if prediction.loss_probability >= 0.60:
                        parts.append(
                            f"\n{YELLOW}⚠ Override risk: {prediction.loss_probability:.0%} "
                            f"predicted loss probability. {prediction.recommendation}{RESET}"
                        )
                except Exception:
                    pass

        # Stressor acknowledgment
        if signals.detected_stressors:
            stressor_text = ", ".join(signals.detected_stressors)
            parts.append(
                f"I'm noting {stressor_text} as active stressors. "
                f"These can affect your trading readiness even when you don't feel it."
            )

        # Readiness impact
        if readiness.readiness_score < 40:
            parts.append(
                f"\n{YELLOW}⚠ Your readiness score is {readiness.readiness_score:.0f}/100. "
                f"Buddy will reduce position sizes and may block new trades "
                f"until your cognitive state improves.{RESET}"
            )
        elif readiness.readiness_score < 60:
            parts.append(
                f"\n{DIM}Readiness: {readiness.readiness_score:.0f}/100 — "
                f"Buddy is using reduced position sizes.{RESET}"
            )

        # Default conversational response
        if not parts:
            parts.append(
                "I'm listening. Tell me what's on your mind — "
                "it all helps me understand how you're doing."
            )

        # Buddy outcome context
        outcome = self.bridge.read_outcome()
        if outcome and "trading" in signals.topics:
            if outcome.streak == "losing":
                parts.append(
                    f"\n{DIM}Buddy context: Currently on a losing streak. "
                    f"PnL today: {outcome.pnl_today:+.2f}. "
                    f"7-day win rate: {outcome.win_rate_7d:.0%}.{RESET}"
                )

        return "\n".join(parts)

    def _update_readiness(self, signals: Optional[ConversationSignals] = None) -> ReadinessSignal:
        """Recompute readiness score and write to bridge."""
        recent_overrides = self.bridge.get_recent_overrides(limit=20)
        override_events = [o.to_dict() for o in recent_overrides]

        # Get conversation count from graph
        recent_convs = self.graph.get_recent_conversations(limit=50)
        # Count conversations from last 7 days
        from datetime import timedelta
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        conv_count_7d = sum(1 for c in recent_convs if c.get("timestamp", "") >= week_ago)

        readiness = self.readiness.compute(
            emotional_state=signals.emotional_state if signals else "neutral",
            stress_keywords=signals.stress_keywords_found if signals else [],
            active_stressors=self._active_stressors,
            recent_override_events=override_events,
            conversation_count_7d=conv_count_7d + 1,  # +1 for current conversation
            confidence_trend=signals.confidence_trend if signals else "stable",
        )
        self._latest_readiness = readiness

        # Log to graph
        self.graph.log_readiness(
            score=readiness.readiness_score,
            components=readiness.components.to_dict(),
            trigger="conversation_update" if signals else "session_start",
        )

        return readiness

    def _log_to_graph(self, message: str, signals: ConversationSignals) -> None:
        """Log conversation data to the self-model graph."""
        # Add conversation node
        conv_node = GraphNode(
            id=f"{self._conversation_id}_{signals.message_count}",
            node_type=NodeType.CONVERSATION,
            label=message[:100],
            properties=signals.to_dict(),
            confidence=0.8,
        )
        self.graph.add_node(conv_node)

        # Add emotion node if notable
        if signals.emotional_state not in ("neutral",):
            emotion_node = GraphNode(
                id=f"emotion_{signals.emotional_state}_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                node_type=NodeType.EMOTION,
                label=signals.emotional_state,
                properties={
                    "sentiment_score": signals.sentiment_score,
                    "stress_keywords": signals.stress_keywords_found,
                },
            )
            self.graph.add_node(emotion_node)

            # Link conversation → emotion
            self.graph.add_edge(GraphEdge(
                source_id=conv_node.id,
                target_id=emotion_node.id,
                edge_type=EdgeType.TRIGGERS,
            ))

        # Add stressor nodes
        for stressor in signals.detected_stressors:
            stressor_node = GraphNode(
                id=f"stressor_{stressor.replace(' ', '_')}",
                node_type=NodeType.LIFE_EVENT,
                label=stressor,
                properties={"active": True},
            )
            self.graph.add_node(stressor_node)

    def _run_pattern_detection(self, signals: ConversationSignals) -> None:
        """Run T1 pattern detection after each conversation message.

        T2 cross-domain runs less frequently — only every 5th message or via /patterns command.
        """
        try:
            conversations = self.graph.get_recent_conversations(limit=20)
            readiness_history = self.graph.get_readiness_history(limit=20)

            # T1 runs every message (lightweight)
            t1_patterns = self.pattern_engine.run_t1(conversations, readiness_history)

            if t1_patterns:
                logger.info(
                    f"T1 detected {len(t1_patterns)} patterns after message "
                    f"#{signals.message_count}"
                )

            # T2 runs every 5th message (moderate cost)
            if signals.message_count % 5 == 0 and signals.message_count > 0:
                t2_patterns = self.pattern_engine.run_t2(conversations, readiness_history)
                if t2_patterns:
                    logger.info(f"T2 detected {len(t2_patterns)} cross-domain patterns")

        except Exception as e:
            logger.warning(f"Pattern detection failed (non-fatal): {e}")

    def _handle_command(self, command: str) -> str:
        """Handle CLI commands."""
        cmd = command.strip().lower()

        if cmd == "/status":
            return self._cmd_status()
        elif cmd == "/bridge":
            return self._cmd_bridge()
        elif cmd == "/readiness":
            return self._cmd_readiness()
        elif cmd == "/graph":
            return self._cmd_graph()
        elif cmd.startswith("/patterns"):
            return self._cmd_patterns(cmd)
        elif cmd.startswith("/predict"):
            return self._cmd_predict()
        elif cmd.startswith("/validate"):
            return self._cmd_validate()
        elif cmd.startswith("/rules"):
            return self._cmd_rules()
        elif cmd == "/quit":
            return "__QUIT__"
        else:
            return f"Unknown command: {cmd}. Available: /status /bridge /readiness /graph /patterns /validate /rules /predict /quit"

    def _cmd_status(self) -> str:
        """Show current Aura status."""
        readiness = self._latest_readiness
        summary = self.processor.get_session_summary()
        graph_stats = self.graph.get_stats()

        lines = [
            f"{BOLD}═══ Aura Status ═══{RESET}",
            f"Session: {self._conversation_id}",
            f"Messages this session: {summary['message_count']}",
            f"Net sentiment: {summary['net_sentiment']:+.3f}",
            f"Active stressors: {', '.join(self._active_stressors) or 'none'}",
            "",
            f"{BOLD}Self-Model Graph:{RESET}",
            f"  Nodes: {graph_stats['total_nodes']}",
            f"  Edges: {graph_stats['total_edges']}",
            f"  Conversations logged: {graph_stats['total_conversations']}",
        ]

        if readiness:
            lines.extend([
                "",
                f"{BOLD}Readiness: {readiness.readiness_score:.0f}/100{RESET}",
                f"  Emotional: {readiness.emotional_state}",
                f"  Cognitive load: {readiness.cognitive_load}",
                f"  Override loss rate (7d): {readiness.override_loss_rate_7d:.0%}",
                f"  Confidence trend: {readiness.confidence_trend}",
            ])

        return "\n".join(lines)

    def _cmd_bridge(self) -> str:
        """Show bridge status."""
        status = self.bridge.get_bridge_status()
        lines = [f"{BOLD}═══ Bridge Status ═══{RESET}"]

        r = status["readiness_signal"]
        lines.append(f"Readiness → Buddy: {'✓' if r['available'] else '✗'} "
                     f"(score: {r['score']:.0f}/100)" if r['available'] else
                     f"Readiness → Buddy: not yet computed")

        o = status["outcome_signal"]
        if o["available"]:
            lines.append(f"Outcomes ← Buddy: ✓ (PnL: {o['pnl_today']:+.2f})")
        else:
            lines.append(f"Outcomes ← Buddy: ✗ (Buddy not active)")

        ov = status["override_events"]
        lines.append(f"Override events: {ov['total_recent']} recent")

        return "\n".join(lines)

    def _cmd_readiness(self) -> str:
        """Show detailed readiness breakdown."""
        if not self._latest_readiness:
            return "No readiness score computed yet. Start a conversation first."

        r = self._latest_readiness
        c = r.components

        score_color = GREEN if r.readiness_score >= 70 else (YELLOW if r.readiness_score >= 40 else RED)

        lines = [
            f"{BOLD}═══ Readiness Score: {score_color}{r.readiness_score:.0f}/100{RESET} ═══",
            "",
            f"  Emotional state:    {c.emotional_state_score:.2f}  (weight: 25%)",
            f"  Cognitive load:     {c.cognitive_load_score:.2f}  (weight: 20%)",
            f"  Override discipline: {c.override_discipline_score:.2f}  (weight: 25%)",
            f"  Stress level:       {c.stress_level_score:.2f}  (weight: 15%)",
            f"  Confidence trend:   {c.confidence_trend_score:.2f}  (weight: 10%)",
            f"  Engagement:         {c.engagement_score:.2f}  (weight: 5%)",
            "",
            f"  Buddy impact: ",
        ]

        if r.readiness_score >= 80:
            lines.append(f"    {GREEN}Full trading capacity{RESET}")
        elif r.readiness_score >= 60:
            lines.append(f"    {YELLOW}Position sizes reduced 20%{RESET}")
        elif r.readiness_score >= 40:
            lines.append(f"    {YELLOW}Position sizes reduced 40%, wider SL{RESET}")
        elif r.readiness_score >= 20:
            lines.append(f"    {RED}Minimum positions only{RESET}")
        else:
            lines.append(f"    {RED}TRADE BLOCK — readiness too low{RESET}")

        return "\n".join(lines)

    def _cmd_graph(self) -> str:
        """Show self-model graph summary."""
        stats = self.graph.get_stats()
        lines = [
            f"{BOLD}═══ Self-Model Graph ═══{RESET}",
            f"Total nodes: {stats['total_nodes']}",
            f"Total edges: {stats['total_edges']}",
            f"Conversations: {stats['total_conversations']}",
            "",
            "Nodes by type:",
        ]
        for node_type, count in stats["nodes_by_type"].items():
            lines.append(f"  {node_type}: {count}")

        # Show recent readiness history
        history = self.graph.get_readiness_history(limit=5)
        if history:
            lines.extend(["", "Recent readiness scores:"])
            for entry in history:
                ts = entry.get("timestamp", "")[:19]
                score = entry.get("score", 0)
                trigger = entry.get("trigger", "")
                lines.append(f"  {ts} — {score:.0f}/100 ({trigger})")

        return "\n".join(lines)

    def _cmd_patterns(self, cmd: str) -> str:
        """Show pattern engine status or force a run.

        Sub-commands:
            /patterns         — Show all active patterns
            /patterns run     — Force full T1+T2+T3 analysis
            /patterns arcs    — Show T3 narrative arcs
            /patterns cloud   — Show cloud synthesis status
        """
        parts = cmd.strip().split()

        # /patterns arcs → show T3 narrative arcs specifically
        if len(parts) > 1 and parts[1] == "arcs":
            arcs = self.pattern_engine.get_narrative_arcs()
            if not arcs:
                return f"{DIM}No narrative arcs detected yet. Need 4+ weeks of data.{RESET}"
            lines = [f"{BOLD}═══ Narrative Arcs (T3) ═══{RESET}"]
            for arc in arcs:
                phase_icon = {
                    "building": "📈", "peak": "🔺", "resolving": "📉",
                    "resolved": "✅", "stable": "➡️",
                }.get(arc.get("phase", ""), "❓")
                lines.append(f"\n{phase_icon} {BOLD}{arc['arc_type']}{RESET} ({arc['phase']})")
                lines.append(f"  {arc['description']}")
                lines.append(f"  Duration: {arc['duration_weeks']} weeks | Confidence: {arc['confidence']:.0%}")
                if arc.get("insight"):
                    lines.append(f"  {GREEN}→ {arc['insight']}{RESET}")
            return "\n".join(lines)

        # /patterns cloud → show cloud synthesis status
        if len(parts) > 1 and parts[1] == "cloud":
            cloud_status = self.pattern_engine.cloud.get_status()
            lines = [f"{BOLD}═══ Cloud Synthesis Status ═══{RESET}"]
            lines.append(f"  Configured: {cloud_status['configured']}")
            lines.append(f"  Provider: {cloud_status['provider']}")
            lines.append(f"  Model: {cloud_status['model']}")
            lines.append(f"  Available: {cloud_status['available']}")
            lines.append(f"  Daily calls: {cloud_status['daily_calls_used']}/{cloud_status['daily_calls_max']}")
            if not cloud_status['configured']:
                lines.append(f"\n{DIM}Set AURA_LLM_PROVIDER and AURA_LLM_API_KEY to enable.{RESET}")
            return "\n".join(lines)

        # /patterns run → force full T1+T2+T3 analysis
        if len(parts) > 1 and parts[1] == "run":
            conversations = self.graph.get_recent_conversations(limit=50)
            readiness_history = self.graph.get_readiness_history(limit=50)
            results = self.pattern_engine.run_all(conversations, readiness_history)

            t1_count = len(results.get("t1", []))
            t2_count = len(results.get("t2", []))
            t3_count = len(results.get("t3", []))
            report = self.pattern_engine.format_patterns_report()

            return (
                f"{DIM}Pattern engine ran: {t1_count} T1 + {t2_count} T2 + {t3_count} T3 patterns{RESET}\n\n"
                + report
            )

        # /patterns → show current status
        return self.pattern_engine.format_patterns_report()

    def _cmd_validate(self) -> str:
        """Run self-model graph validation and show health report."""
        try:
            from src.aura.core.self_model_validator import SelfModelValidator
            validator = SelfModelValidator(
                graph=self.graph,
                auto_remediate=True,
            )
            report = validator.validate()
            return report.format_report()
        except Exception as e:
            return f"{DIM}Self-model validation failed: {e}{RESET}"

    def _cmd_rules(self) -> str:
        """Show active bridge rules."""
        try:
            from src.aura.bridge.rules_engine import BridgeRulesEngine
            engine = BridgeRulesEngine()
            summary = engine.get_rules_summary()

            if summary["active_rules"] == 0:
                return f"{DIM}No active bridge rules. Rules are created when patterns are promoted.{RESET}"

            lines = [f"{BOLD}═══ Active Bridge Rules ═══{RESET}"]
            lines.append(
                f"  Active: {summary['active_rules']} | "
                f"Expired: {summary['expired_rules']} | "
                f"Aura→Buddy: {summary['aura_to_buddy']} | "
                f"Buddy→Aura: {summary['buddy_to_aura']}"
            )

            for rule in summary["rules"]:
                direction_icon = "🧠→📊" if rule["direction"] == "aura_to_buddy" else "📊→🧠"
                lines.append(f"\n{direction_icon} {BOLD}{rule['rule_type']}{RESET}")
                lines.append(f"  {rule['description']}")
                lines.append(
                    f"  Confidence: {rule['confidence']:.0%} | "
                    f"Triggered: {rule['triggered_count']}x | "
                    f"Expires: {rule['expires_at'][:10]}"
                )
                adj = rule.get("adjustment", {})
                if adj:
                    lines.append(f"  Adjustment: {adj.get('parameter')} = {adj.get('value')}")

            return "\n".join(lines)
        except Exception as e:
            return f"{DIM}Bridge rules engine not available: {e}{RESET}"

    def _cmd_predict(self) -> str:
        """Show prediction model status and current override risk."""
        lines = [f"{BOLD}═══ Prediction Models (Phase 3) ═══{RESET}"]

        if self._override_predictor:
            info = self._override_predictor.get_model_info()
            lines.append(f"\n{BOLD}Override Predictor:{RESET}")
            lines.append(f"  Trained: {info['trained']}")
            lines.append(f"  Training samples: {info['train_samples']}")
            lines.append(f"  Training accuracy: {info['train_accuracy']:.1%}")

            # Show current risk assessment
            context = {
                "emotional_state": self._latest_signals.emotional_state if self._latest_signals else "neutral",
                "cognitive_load": "high" if self._active_stressors and len(self._active_stressors) > 2 else "normal",
                "confidence_at_time": 0.5,
                "weighted_vote_at_time": 0.5,
                "override_type": "took_rejected",
                "regime": "NORMAL",
            }
            pred = self._override_predictor.predict_loss_probability(context)
            risk_color = RED if pred.risk_level in ("critical", "high") else (YELLOW if pred.risk_level == "moderate" else GREEN)
            lines.append(f"\n  Current override risk: {risk_color}{pred.loss_probability:.0%} ({pred.risk_level}){RESET}")
            if pred.top_risk_factors:
                lines.append(f"  Top factors: {', '.join(pred.top_risk_factors)}")
        else:
            lines.append(f"\n{DIM}Override predictor not available{RESET}")

        if self._readiness_v2:
            info = self._readiness_v2.get_model_info()
            lines.append(f"\n{BOLD}Readiness V2:{RESET}")
            lines.append(f"  Version: {info['version']}")
            lines.append(f"  Training buffer: {info['buffer_size']}/{info['min_samples']} samples")
            if info['trained']:
                lines.append(f"  R²: {info['r_squared']:.3f}")
                comparison = self._readiness_v2.get_weight_comparison()
                lines.append(f"\n  Weight comparison (v1 → v2):")
                for name, comp in comparison.items():
                    delta_sign = "+" if comp['delta'] > 0 else ""
                    lines.append(
                        f"    {name}: {comp['v1_weight']:.2f} → {comp['v2_weight']:.2f} "
                        f"({delta_sign}{comp['delta']:.3f})"
                    )
            else:
                lines.append(f"  Samples until v2: {info['samples_until_v2']}")
        else:
            lines.append(f"\n{DIM}Readiness v2 model not available{RESET}")

        return "\n".join(lines)

    def end_session(self) -> None:
        """Clean up and log session summary to graph."""
        summary = self.processor.get_session_summary()

        # Log final conversation record
        self.graph.log_conversation(
            conversation_id=self._conversation_id,
            summary=f"Session with {summary['message_count']} messages, "
                    f"net sentiment {summary['net_sentiment']:+.3f}",
            emotional_state=self._latest_signals.emotional_state if self._latest_signals else "neutral",
            topics=self._latest_signals.topics if self._latest_signals else [],
            readiness_impact=self._latest_readiness.readiness_score if self._latest_readiness else 0,
            messages=self._message_history[-20:],  # Keep last 20 messages
        )

        print(f"\n{DIM}Session ended. Self-model updated.{RESET}")
        self.graph.close()
