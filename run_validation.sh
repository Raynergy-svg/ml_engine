#!/bin/bash
cd /Users/davidcertan/Desktop/ml_engine
python3 -m pytest tests/test_news_sentiment.py tests/test_online_retrainer.py tests/test_rl_features.py -v --tb=short 2>&1 | tail -80
echo "---"
python3 -m pytest tests/test_engine_heads.py tests/test_cli_commands.py tests/test_inference_pipeline.py -v --tb=short 2>&1 | tail -80
