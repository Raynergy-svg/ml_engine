#!/usr/bin/env python3
"""
Test script for improved training report generation.

Verifies that the enhanced markdown training report contains:
1. Professional header with report metadata
2. Executive summary table
3. Structured model performance sections with tables
4. Training configuration details
5. Model artifacts paths
6. System information footer
"""

import re
from pathlib import Path


def test_training_report_format():
    """Test that the training report has professional formatting."""
    
    # Read the training.py file to verify report generation
    training_file = Path("cli/training.py")
    content = training_file.read_text()
    
    # Verify professional elements are present
    checks = {
        "professional_header": "# Advanced Ensemble Training Report",
        "report_id": "**Report ID**",
        "executive_summary": "## Executive Summary",
        "model_performance": "## Model Performance",
        "transformer_section": "### Transformer Network",
        "gradient_boosting": "### Gradient Boosting",
        "random_forest": "### Random Forest",
        "regularized_regression": "### Regularized Regression",
        "training_config": "## Training Configuration",
        "hyperparameters": "### Hyperparameters",
        "data_split": "### Data Split",
        "model_artifacts": "## Model Artifacts",
        "component_models": "### Component Models",
        "metadata_section": "### Metadata",
        "system_info": "## System Information",
        "footer": "Advanced Ensemble Training System",
        "enterprise_footer": "Enterprise-Grade Machine Learning Infrastructure",
    }
    
    passed = []
    failed = []
    
    for check_name, check_string in checks.items():
        if check_string in content:
            passed.append(check_name)
        else:
            failed.append(check_name)
    
    # Print results
    print("=" * 70)
    print("TRAINING REPORT FORMAT VERIFICATION")
    print("=" * 70)
    print()
    
    if passed:
        print(f"✅ PASSED CHECKS ({len(passed)}/{len(checks)}):")
        for item in passed:
            print(f"   • {item}")
        print()
    
    if failed:
        print(f"❌ FAILED CHECKS ({len(failed)}/{len(checks)}):")
        for item in failed:
            print(f"   • {item}")
        print()
    
    # Verify table formatting
    table_patterns = [
        r"\| Parameter \| Value \|",  # Executive summary table
        r"\| Metric \| Value \|",      # Performance tables
        r"\| Component \| Path \|",    # Artifacts table
    ]
    
    table_checks_passed = 0
    for pattern in table_patterns:
        if re.search(pattern, content):
            table_checks_passed += 1
    
    print(f"📊 TABLE FORMATTING: {table_checks_passed}/{len(table_patterns)} tables found")
    print()
    
    # Verify markdown sections are properly separated
    separator_count = content.count("---\n")
    print(f"📝 SECTION SEPARATORS: {separator_count} markdown separators found")
    print()
    
    # Verify professional terminology
    professional_terms = [
        "Enterprise-Grade",
        "Component Models",
        "Feature Sparsity",
        "Hyperparameters",
        "Total Samples",
    ]
    
    terminology_found = sum(1 for term in professional_terms if term in content)
    print(f"🎯 PROFESSIONAL TERMINOLOGY: {terminology_found}/{len(professional_terms)} terms used")
    print()
    
    # Overall assessment
    total_checks = len(checks) + len(table_patterns)
    total_passed = len(passed) + table_checks_passed
    pass_rate = (total_passed / total_checks) * 100
    
    print("=" * 70)
    if pass_rate >= 90:
        print(f"🎉 OVERALL: EXCELLENT ({pass_rate:.0f}% checks passed)")
        print("   Training report format is professional and well-structured!")
    elif pass_rate >= 70:
        print(f"✅ OVERALL: GOOD ({pass_rate:.0f}% checks passed)")
        print("   Training report format is acceptable but could be improved")
    else:
        print(f"⚠️  OVERALL: NEEDS WORK ({pass_rate:.0f}% checks passed)")
        print("   Training report format needs significant improvements")
    print("=" * 70)
    
    return len(failed) == 0 and table_checks_passed == len(table_patterns)


if __name__ == "__main__":
    success = test_training_report_format()
    exit(0 if success else 1)
