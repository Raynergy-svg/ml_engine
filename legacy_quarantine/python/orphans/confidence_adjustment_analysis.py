#!/usr/bin/env python3
"""
Confidence Level Adjustment Analysis for Trading Scenario

This script analyzes the provided trading scenario and calculates a more realistic
confidence level based on the weak supporting metrics.
"""


def analyze_confidence_adjustment():
    """
    Analyze and adjust the model confidence based on trading scenario factors.
    
    Returns:
        dict: Analysis results with adjusted confidence and justification
    """
    
    # Original values from the scenario
    original_confidence = 0.725
    prediction_direction = 0.486
    tier2_win_probability = 0.335
    
    print("=== CONFIDENCE ADJUSTMENT ANALYSIS ===")
    print(f"Original Model Confidence: {original_confidence}")
    print(f"Prediction Direction: {prediction_direction}")
    print(f"Tier2 Win Probability: {tier2_win_probability}")
    print()
    
    # Factor 1: Prediction Direction Analysis
    # Direction close to 0.5 indicates uncertainty
    direction_uncertainty = abs(prediction_direction - 0.5)
    direction_penalty = 1.0 - (direction_uncertainty * 2)  # Higher penalty for neutral predictions
    
    print("Direction Analysis:")
    print(f"  Distance from neutral (0.5): {direction_uncertainty}")
    print(f"  Direction penalty factor: {direction_penalty}")
    print()

    # Factor 2: Tier2 Win Probability Analysis
    # Win probability below 50% suggests poor reliability
    win_prob_penalty = tier2_win_probability  # Direct scaling - lower win prob = lower confidence

    print("Win Probability Analysis:")
    print(f"  Tier2 win probability: {tier2_win_probability}")
    print(f"  Win probability penalty factor: {win_prob_penalty}")
    print()

    # Factor 3: Combined Adjustment Calculation
    # Apply multiple penalties to original confidence
    adjusted_confidence = original_confidence * direction_penalty * win_prob_penalty

    # Apply minimum floor to prevent over-penalization
    min_confidence = 0.15  # Conservative minimum for any trading decision
    adjusted_confidence = max(adjusted_confidence, min_confidence)

    # Additional conservative adjustment for trading context
    # Trading requires higher standards than general ML predictions
    trading_adjustment = 0.85  # 15% additional penalty for trading risk
    final_confidence = adjusted_confidence * trading_adjustment

    print("Adjustment Calculations:")
    print(f"  Base adjustment: {original_confidence} * {direction_penalty} * {win_prob_penalty} = {adjusted_confidence:.4f}")
    print(f"  Trading context adjustment: {adjusted_confidence:.4f} * {trading_adjustment} = {final_confidence:.4f}")
    print()
    
    # Justification analysis
    print("=== JUSTIFICATION ===")
    print("The original confidence of 0.725 appears inflated because:")
    print("1. Prediction direction (0.486) is very close to neutral (0.5), indicating high uncertainty")
    print("2. Tier2 win probability (0.335) is significantly below 50%, suggesting poor predictive power")
    print("3. Trading decisions require conservative confidence estimates due to financial risk")
    print("4. The combination of neutral prediction and low win rate should substantially reduce confidence")
    print()
    
    print(f"=== FINAL ADJUSTED CONFIDENCE: {final_confidence:.4f} ===")
    
    return {
        'original_confidence': original_confidence,
        'adjusted_confidence': final_confidence,
        'direction_uncertainty': direction_uncertainty,
        'direction_penalty': direction_penalty,
        'win_prob_penalty': win_prob_penalty,
        'justification': [
            "Prediction direction too close to neutral (high uncertainty)",
            "Tier2 win probability significantly below 50% (poor reliability)",
            "Trading context requires conservative confidence estimates",
            "Original confidence appears inflated relative to supporting metrics"
        ]
    }


def generate_trading_recommendation(adjusted_confidence):
    """
    Generate trading recommendation based on adjusted confidence level.
    
    Args:
        adjusted_confidence (float): The adjusted confidence value
        
    Returns:
        str: Trading recommendation
    """
    
    if adjusted_confidence >= 0.7:
        return "HIGH CONFIDENCE: Consider full position size with standard risk management"
    elif adjusted_confidence >= 0.5:
        return "MODERATE CONFIDENCE: Consider reduced position size or wait for confirmation"
    elif adjusted_confidence >= 0.3:
        return "LOW CONFIDENCE: Consider very small position size or avoid trade"
    else:
        return "VERY LOW CONFIDENCE: Avoid trade or wait for better setup"


if __name__ == "__main__":
    # Run the analysis
    results = analyze_confidence_adjustment()
    
    # Generate trading recommendation
    recommendation = generate_trading_recommendation(results['adjusted_confidence'])
    
    print(f"TRADING RECOMMENDATION: {recommendation}")
    print()
    print("=== SUMMARY ===")
    print(f"Original Confidence: {results['original_confidence']:.3f}")
    print(f"Adjusted Confidence: {results['adjusted_confidence']:.3f}")
    print(f"Confidence Reduction: {(results['original_confidence'] - results['adjusted_confidence']):.3f} ({((results['original_confidence'] - results['adjusted_confidence']) / results['original_confidence'] * 100):.1f}%)")
    print()
    print("Key Factors Considered:")
    for i, factor in enumerate(results['justification'], 1):
        print(f"  {i}. {factor}")

# — Raynergy-svg —
