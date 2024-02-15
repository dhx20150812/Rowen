# Retrieve Only When It Needs: Adaptive Retrieval Augmentation for Hallucination Mitigation in Large Language Models

## About Rowen
Rowen is a novel method that meticulously integrate the parametric knowledge within LLMs with external information for hallucination mitigation. Rowen  We start by producing an initial response using our inherent knowledge base. Then, a multilingual detection system assesses the semantic consistency of responses to the same question in different languages. If inconsistencies arise, a retrieval-augmented mechanism is engaged to fetch external information, helping to rectify the reasoning and correct any inaccuracies. If responses remain consistent, the initial internally reasoned answer is retained.
![Overview](fig/method.png)

## Requirements
- openai == 0.28.0
- [Serper Api Key](https://serper.dev/)
- [OpenAI Api Key](https://chat.openai.com)

## Usage
1. Run experiments on the TruthfulQA dataset

```bash
python run_truthfulqa.py
```

2. Run experiments on the StrategyQA dataset

```bash
python run_strategyqa.py
```

Generated responses and evaluation results will be saved automatically.

Please be careful, the experiments are relatively expensive because Rowen calls OpenAI API multiple times for a single question. You can decrease the `k` (number of pertubed questions) in the file to reduce the cost.

## Performance
![Main Result of Rowen](fig/main_results.png)
