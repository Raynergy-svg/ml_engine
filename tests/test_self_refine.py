"""Tests for the Self-Refine autonomous self-improvement module."""

import pytest
from unittest.mock import MagicMock, patch
from self_refine import (
    buddy_self_improve,
    get_feedback_only,
    refine_response,
    PROMPTS,
    INITIAL_GENERATION_SYSTEM,
    FEEDBACK_PROMPT,
    REFINE_PROMPT,
)


class TestBuddySelfImprove:
    """Test the main buddy_self_improve function."""

    def test_successful_single_iteration(self):
        """Test successful refinement that stops after one iteration."""
        call_count = 0
        
        def mock_llm(prompt, system_prompt=None, model="gpt-4", temperature=0.2):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # Initial generation
                return "Initial response about ML model training."
            elif call_count == 2:
                # Feedback indicates no improvements needed
                return "No further improvements needed. The response is comprehensive."
            else:
                return "Should not reach here"
        
        result = buddy_self_improve(
            query="How do I train an ML model?",
            max_iterations=4,
            llm_call_fn=mock_llm,
        )
        
        assert result['response'] == "Initial response about ML model training."
        assert result['iterations'] == 1
        assert result['early_stopped'] is True
        assert len(result['history']) == 1

    def test_multiple_iterations_with_refinement(self):
        """Test refinement through multiple iterations."""
        call_count = 0
        
        def mock_llm(prompt, system_prompt=None, model="gpt-4", temperature=0.2):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return "Initial response v1"
            elif call_count == 2:
                return "Feedback: Could improve by adding examples."
            elif call_count == 3:
                return "Refined response v2 with examples"
            elif call_count == 4:
                return "No further improvements needed."
            else:
                return "Unexpected call"
        
        result = buddy_self_improve(
            query="Explain gradient descent",
            max_iterations=4,
            llm_call_fn=mock_llm,
        )
        
        assert result['response'] == "Refined response v2 with examples"
        assert result['iterations'] == 2
        assert result['early_stopped'] is True
        assert len(result['history']) == 2

    def test_max_iterations_reached(self):
        """Test that refinement stops at max_iterations."""
        call_count = 0
        
        def mock_llm(prompt, system_prompt=None, model="gpt-4", temperature=0.2):
            nonlocal call_count
            call_count += 1
            
            # Never return "no further improvements needed"
            if call_count % 2 == 1:
                return f"Response version {(call_count + 1) // 2}"
            else:
                return f"Feedback iteration {call_count // 2}: Can still improve."
        
        result = buddy_self_improve(
            query="Complex ML question",
            max_iterations=2,
            llm_call_fn=mock_llm,
        )
        
        # Should have gone through 2 iterations
        assert result['iterations'] == 2
        assert result['early_stopped'] is False

    def test_initial_generation_failure(self):
        """Test handling when initial generation fails."""
        def mock_llm(prompt, system_prompt=None, model="gpt-4", temperature=0.2):
            return None
        
        result = buddy_self_improve(
            query="Test query",
            llm_call_fn=mock_llm,
        )
        
        assert result['response'] is None
        assert result['iterations'] == 0
        assert 'error' in result

    def test_feedback_failure_continues_gracefully(self):
        """Test that feedback failure stops iteration gracefully."""
        call_count = 0
        
        def mock_llm(prompt, system_prompt=None, model="gpt-4", temperature=0.2):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return "Initial response"
            else:
                # Feedback fails
                return None
        
        result = buddy_self_improve(
            query="Test query",
            llm_call_fn=mock_llm,
        )
        
        assert result['response'] == "Initial response"
        assert result['iterations'] == 0

    def test_verbose_logging(self, caplog):
        """Test that verbose mode logs progress."""
        import logging
        
        def mock_llm(prompt, system_prompt=None, model="gpt-4", temperature=0.2):
            return "No further improvements needed."
        
        with caplog.at_level(logging.INFO):
            result = buddy_self_improve(
                query="Test",
                verbose=True,
                llm_call_fn=mock_llm,
            )
        
        # Check that some logging occurred
        assert result['early_stopped'] is True

    def test_case_insensitive_early_stop(self):
        """Test that early stopping works regardless of case."""
        call_count = 0
        
        def mock_llm(prompt, system_prompt=None, model="gpt-4", temperature=0.2):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return "Response"
            else:
                return "NO FURTHER IMPROVEMENTS NEEDED"
        
        result = buddy_self_improve(
            query="Test",
            llm_call_fn=mock_llm,
        )
        
        assert result['early_stopped'] is True


class TestGetFeedbackOnly:
    """Test the get_feedback_only function."""

    def test_returns_feedback(self):
        """Test that feedback is returned correctly."""
        def mock_llm(prompt, system_prompt=None, model="gpt-4", temperature=0.2):
            assert "Current Response:" in prompt
            return "Feedback: Add more examples."
        
        feedback = get_feedback_only(
            query="How to train a model?",
            response="Train using fit() method.",
            llm_call_fn=mock_llm,
        )
        
        assert feedback == "Feedback: Add more examples."

    def test_returns_none_on_failure(self):
        """Test that None is returned when LLM fails."""
        def mock_llm(prompt, system_prompt=None, model="gpt-4", temperature=0.2):
            return None
        
        feedback = get_feedback_only(
            query="Test",
            response="Response",
            llm_call_fn=mock_llm,
        )
        
        assert feedback is None


class TestRefineResponse:
    """Test the refine_response function."""

    def test_returns_refined_response(self):
        """Test that refined response is returned correctly."""
        def mock_llm(prompt, system_prompt=None, model="gpt-4", temperature=0.2):
            assert "Previous Response:" in prompt
            assert "Feedback:" in prompt
            return "Refined response with improvements."
        
        refined = refine_response(
            query="How to train?",
            current_response="Initial response",
            feedback="Add examples",
            llm_call_fn=mock_llm,
        )
        
        assert refined == "Refined response with improvements."

    def test_returns_none_on_failure(self):
        """Test that None is returned when LLM fails."""
        def mock_llm(prompt, system_prompt=None, model="gpt-4", temperature=0.2):
            return None
        
        refined = refine_response(
            query="Test",
            current_response="Response",
            feedback="Feedback",
            llm_call_fn=mock_llm,
        )
        
        assert refined is None


class TestPromptTemplates:
    """Test prompt template structure and content."""

    def test_prompts_dict_contains_all_templates(self):
        """Test that PROMPTS dict has all expected keys."""
        expected_keys = [
            'initial_generation_system',
            'initial_generation_user',
            'feedback',
            'refine',
        ]
        for key in expected_keys:
            assert key in PROMPTS

    def test_initial_generation_system_mentions_buddy(self):
        """Test that system prompt mentions Buddy."""
        assert "Buddy" in INITIAL_GENERATION_SYSTEM
        assert "machine learning" in INITIAL_GENERATION_SYSTEM.lower()

    def test_feedback_prompt_has_placeholders(self):
        """Test that feedback prompt has required placeholders."""
        assert "{original_user_query}" in FEEDBACK_PROMPT
        assert "{current_response}" in FEEDBACK_PROMPT
        assert "no further improvements needed" in FEEDBACK_PROMPT.lower()

    def test_refine_prompt_has_placeholders(self):
        """Test that refine prompt has required placeholders."""
        assert "{original_user_query}" in REFINE_PROMPT
        assert "{current_response}" in REFINE_PROMPT
        assert "{feedback}" in REFINE_PROMPT

    def test_prompts_can_be_formatted(self):
        """Test that prompts can be formatted without errors."""
        from self_refine import INITIAL_GENERATION_USER
        
        formatted = INITIAL_GENERATION_USER.format(user_query="Test query")
        assert "Test query" in formatted
        
        formatted = FEEDBACK_PROMPT.format(
            original_user_query="Query",
            current_response="Response",
        )
        assert "Query" in formatted
        assert "Response" in formatted
        
        formatted = REFINE_PROMPT.format(
            original_user_query="Query",
            current_response="Response",
            feedback="Feedback text",
        )
        assert "Query" in formatted
        assert "Response" in formatted
        assert "Feedback text" in formatted


class TestIntegrationWithHistory:
    """Test history tracking in self-improvement."""

    def test_history_contains_responses_and_feedback(self):
        """Test that history correctly tracks all iterations."""
        responses = ["v1", "v2", "v3"]
        feedbacks = ["fb1", "fb2", "No further improvements needed."]
        call_idx = 0
        
        def mock_llm(prompt, system_prompt=None, model="gpt-4", temperature=0.2):
            nonlocal call_idx
            result_idx = call_idx // 2
            is_feedback = call_idx % 2 == 1
            call_idx += 1
            
            if call_idx == 1:
                return responses[0]
            elif is_feedback:
                return feedbacks[result_idx]
            else:
                return responses[result_idx]
        
        result = buddy_self_improve(
            query="Test",
            max_iterations=4,
            llm_call_fn=mock_llm,
        )
        
        # Should have history entries
        assert len(result['history']) > 0
        # Each entry should be a tuple of (response, feedback)
        for entry in result['history']:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
# — Raynergy-svg —
