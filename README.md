# Risk-Aware Calibration for Multiple-Choice Question Answering

A comprehensive implementation of temperature scaling calibration for large language models on the HellaSwag commonsense reasoning task. This project evaluates model confidence calibration and risk-aware prediction across multiple architectures and calibration settings.

##  Overview

This project implements and evaluates temperature scaling—a post-hoc calibration method that adjusts model confidence to better reflect true prediction accuracy. We compare four different language models on the HellaSwag dataset, analyzing both calibration quality and risk-coverage trade-offs under selective prediction scenarios.

**Key Research Questions:**
- How well do different LLM architectures calibrate after temperature scaling?
- What is the optimal calibration set size for different models?
- How does calibration affect risk-coverage trade-offs in selective prediction?

##  Models

We evaluate four models with different architectural approaches:

| Model | Type | Accuracy | Description |
|-------|------|----------|-------------|
| **RoBERTa-large** | Multiple-Choice Head | ~48% | Fine-tuned on RACE dataset |
| **DeBERTa-v3-base** | NLI Entailment | ~71% | Natural language inference approach |
| **FLAN-T5-XL** | Teacher Forcing | ~30% | Instruction-tuned sequence-to-sequence |
| **Phi-3.5-mini** | Letter-only Scoring | ~74% | Small instruct-tuned causal LM |

## Features

### Calibration
- **Temperature Scaling**: Learns optimal temperature parameter on calibration set
- **Multiple Calibration Ratios**: Tests 5%, 10%, 20%, and 40% splits
- **Stratified Sampling**: Ensures balanced class distribution in cal/test splits

### Metrics
- **Calibration Quality**: ECE, reliability diagrams, Brier score
- **Selective Prediction**: AURC, risk-coverage curves
- **Abstention Analysis**: Coverage at Risk (C@R), Risk at Coverage (R@C)

### Visualization
- Calibration scatter plots (confidence vs. accuracy)
- Reliability histograms with confidence distributions
- Risk-coverage curves
- Per-model and aggregated results

