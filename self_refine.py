"""
Self-Refine: Autonomous Self-Improvement for Buddy AI

This module implements the Self-Refine technique (Madaan et al., 2023) where the LLM
iteratively:
1. Generates an initial response
2. Critiques it for errors/improvements
3. Refines the response based on feedback
4. Repeats until satisfied or max iterations reached

This creates autonomous improvement through self-feedback loops without requiring
model weight updates, ideal for ML/code generation, reasoning, or explanations.
"""

import logging
import os
from typing import Optional, Dict, Any, Callable

import openai

logger = logging.getLogger(__name__)

# ============================================================================
# Prompt Templates
# ============================================================================

INITIAL_GENERATION_SYSTEM = """You are Buddy, a highly capable AI assistant specialized in machine learning engineering. You excel at reasoning step-by-step, writing clean code, explaining concepts accurately, and solving ML problems efficiently. Always aim for correctness, clarity, completeness, and optimal solutions."""

INITIAL_GENERATION_USER = """Task: {user_query}

Provide your initial response to the task. Think step-by-step if needed, then give the final output."""

FEEDBACK_PROMPT = """You are an expert self-critic for improving AI outputs, especially in machine learning tasks. Your goal is to identify mistakes, inaccuracies, logical gaps, inefficiencies, unclear explanations, missing edge cases, or any other improvements.

Task: {original_user_query}

Current Response: 
{current_response}

Analyze the current response thoroughly. Provide specific, actionable feedback:
- Point out errors or weaknesses (e.g., "The model selection ignores overfitting risk because...").
- Suggest concrete improvements (e.g., "Add cross-validation to evaluate generalization").
- Cover aspects like correctness, efficiency, readability, completeness, and best practices in ML.

If the response is already optimal with no meaningful improvements possible, explicitly say: "No further improvements needed."

Feedback:"""

REFINE_PROMPT = """You are Buddy, improving your previous response based on expert feedback.

Task: {original_user_query}

Previous Response: 
{current_response}

Feedback: 
{feedback}

Incorporate the feedback fully to produce a better response. Fix all identified issues, enhance clarity/reasoning/code, and avoid repeating past mistakes.

Think step-by-step if helpful, then provide only the refined final response (no need to restate feedback or previous versions).

Refined Response:"""


def _llm_call(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4",
    temperature: float = 0.2,
) -> Optional[str]:
    """Internal LLM call using the OpenAI API.
    
    Args:
        prompt: The user prompt to send.
        system_prompt: Optional system prompt (if None, uses Buddy default).
        model: The model to use (default: gpt-4).
        temperature: Temperature for generation (default: 0.2).
        
    Returns:
        The model's response text, or None if the call failed.
    """
    if not openai.api_key:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            logger.error("OPENAI_API_KEY not set.")
            return None
    
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            # Default to Buddy's system prompt for consistency
            messages.append({
                "role": "system",
                "content": INITIAL_GENERATION_SYSTEM
            })
        messages.append({"role": "user", "content": prompt})
        
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


def buddy_self_improve(
    query: str,
    max_iterations: int = 4,
    model: str = "gpt-4",
    temperature: float = 0.2,
    verbose: bool = False,
    llm_call_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Implement Self-Refine autonomous self-improvement for Buddy.
    
    This function uses the Self-Refine technique to iteratively improve responses:
    1. Generate initial response
    2. Get feedback on the response
    3. Refine based on feedback
    4. Repeat until feedback indicates no improvements needed or max iterations reached
    
    Args:
        query: The user's query/task to respond to.
        max_iterations: Maximum number of feedback-refine cycles (default: 4).
        model: The LLM model to use (default: gpt-4).
        temperature: Temperature for generation (default: 0.2).
        verbose: If True, log intermediate steps.
        llm_call_fn: Optional custom LLM call function for testing.
        
    Returns:
        A dictionary containing:
        - 'response': The final refined response
        - 'iterations': Number of refinement iterations performed
        - 'history': List of (response, feedback) tuples from each iteration
        - 'early_stopped': Whether stopped early due to "no further improvements"
    """
    # Use provided LLM function or default
    llm_call = llm_call_fn if llm_call_fn else _llm_call
    
    # Step 1: Initial generation
    initial_prompt = INITIAL_GENERATION_USER.format(user_query=query)
    current_response = llm_call(
        initial_prompt,
        system_prompt=INITIAL_GENERATION_SYSTEM,
        model=model,
        temperature=temperature,
    )
    
    if current_response is None:
        logger.error("Initial generation failed")
        return {
            'response': None,
            'iterations': 0,
            'history': [],
            'early_stopped': False,
            'error': 'Initial generation failed'
        }
    
    if verbose:
        logger.info(f"Initial response generated ({len(current_response)} chars)")
    
    history = []
    early_stopped = False
    
    # Step 2-4: Feedback and refinement loop
    for i in range(max_iterations):
        # Get feedback
        feedback_prompt = FEEDBACK_PROMPT.format(
            original_user_query=query,
            current_response=current_response,
        )
        feedback = llm_call(
            feedback_prompt,
            model=model,
            temperature=temperature,
        )
        
        if feedback is None:
            logger.warning(f"Feedback generation failed at iteration {i+1}")
            break
        
        if verbose:
            logger.info(f"Iteration {i+1}: Feedback received ({len(feedback)} chars)")
        
        # Check for early stopping
        if "no further improvements needed" in feedback.lower():
            if verbose:
                logger.info(f"Early stopping at iteration {i+1}: No further improvements needed")
            history.append((current_response, feedback))
            early_stopped = True
            break
        
        # Refine response
        refine_prompt_text = REFINE_PROMPT.format(
            original_user_query=query,
            current_response=current_response,
            feedback=feedback,
        )
        refined_response = llm_call(
            refine_prompt_text,
            model=model,
            temperature=temperature,
        )
        
        if refined_response is None:
            logger.warning(f"Refinement failed at iteration {i+1}")
            history.append((current_response, feedback))
            break
        
        if verbose:
            logger.info(f"Iteration {i+1}: Response refined ({len(refined_response)} chars)")
        
        history.append((current_response, feedback))
        current_response = refined_response
    
    return {
        'response': current_response,
        'iterations': len(history),
        'history': history,
        'early_stopped': early_stopped,
    }


def get_feedback_only(
    query: str,
    response: str,
    model: str = "gpt-4",
    temperature: float = 0.2,
    llm_call_fn: Optional[Callable] = None,
) -> Optional[str]:
    """Get feedback on a response without performing refinement.
    
    Useful for evaluating responses or getting improvement suggestions
    without automatically applying them.
    
    Args:
        query: The original user query.
        response: The response to critique.
        model: The LLM model to use.
        temperature: Temperature for generation.
        llm_call_fn: Optional custom LLM call function for testing.
        
    Returns:
        The feedback text, or None if generation failed.
    """
    llm_call = llm_call_fn if llm_call_fn else _llm_call
    
    feedback_prompt = FEEDBACK_PROMPT.format(
        original_user_query=query,
        current_response=response,
    )
    
    return llm_call(feedback_prompt, model=model, temperature=temperature)


def refine_response(
    query: str,
    current_response: str,
    feedback: str,
    model: str = "gpt-4",
    temperature: float = 0.2,
    llm_call_fn: Optional[Callable] = None,
) -> Optional[str]:
    """Refine a response based on provided feedback.
    
    Useful for manual control over the refinement process.
    
    Args:
        query: The original user query.
        current_response: The response to refine.
        feedback: The feedback to incorporate.
        model: The LLM model to use.
        temperature: Temperature for generation.
        llm_call_fn: Optional custom LLM call function for testing.
        
    Returns:
        The refined response, or None if generation failed.
    """
    llm_call = llm_call_fn if llm_call_fn else _llm_call
    
    refine_prompt_text = REFINE_PROMPT.format(
        original_user_query=query,
        current_response=current_response,
        feedback=feedback,
    )
    
    return llm_call(refine_prompt_text, model=model, temperature=temperature)


# Expose prompt templates for customization
PROMPTS = {
    'initial_generation_system': INITIAL_GENERATION_SYSTEM,
    'initial_generation_user': INITIAL_GENERATION_USER,
    'feedback': FEEDBACK_PROMPT,
    'refine': REFINE_PROMPT,
}
# — Raynergy-svg —
