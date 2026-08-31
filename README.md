# GridSentinel AI

## AI-Powered Electrical Grid Risk Intelligence

GridSentinel AI is an engineering-focused AI system for electrical-grid risk analysis and failure prediction.

The system combines:

- Machine learning failure prediction
- Electrical operating conditions
- Historical fault information
- Operational risk scoring
- Safety Guard logic
- Fault simulation
- Integrated validation

## Current Status

**V1-V5 Integration Baseline: PASS**

- Integration checks: **34/34 PASS**
- Dataset: **51,000 records**
- V2 Failure Predictor: Operational
- Risk Intelligence Engine: Operational
- Safety Guard: Operational
- Fault Simulation: Operational
- Python compilation: Verified

**Phase 6 - Deployment: Not started**

## Architecture

```text
Grid Data
    |
    v
Data Validation
    |
    v
Feature Engineering
    |
    v
Failure Prediction
    |
    v
Operational Risk Engine
    |
    v
Safety Guard
    |
    v
Risk Classification
    |
    +--> LOW
    +--> MEDIUM
    +--> HIGH
    +--> CRITICAL
    |
    v
Engineering Recommendation

## Core Components

### ML Prediction
The V2 Failure Predictor uses a 16-feature input pipeline with feature scaling and a PyTorch neural network.

### Risk Intelligence
The risk engine combines ML probability with physical operating conditions including:
- Temperature
- Loading
- THD
- Voltage
- Frequency
- Power factor
- Previous faults
- Asset criticality
- Active fault type

### Safety Guard
The Safety Guard provides an additional safety layer between model output and the final risk state.

### Fault Simulation
The simulation layer supports controlled electrical-grid fault scenarios including:
- Overload
- Overheating
- Voltage instability
- Harmonic distortion

## Validation

Run the complete V1-V5 integration test:

```bash
python -m ai.test_full_pipeline
Total checks: 34
Passed:       34
Failed:       0

STATUS: PASS
gridsentinel-ai/
├── ai/
│   ├── data/
│   ├── ml/
│   └── test_full_pipeline.py
├── datasets/
├── models/
├── simulation/
├── docs/
├── backend/
├── README.md
└── .gitignore
ML Prediction
      +
Physical Conditions
      +
Historical Faults
      +
Risk Intelligence
      +
Safety Guard
      =
Engineering Risk Decision


The system is designed as an engineering decision-support platform and is not intended to replace certified protection systems, protection relays, or qualified electrical engineers.

## Roadmap

- [x] V1 - Data & ML Foundation
- [x] V2 - Failure Prediction
- [x] V3 - Risk Intelligence
- [x] V4 - Safety Guard
- [x] V5 - Simulation & Integration Validation
- [ ] V6 - Deployment

## Author

**Eng. Moaz Qasem**
Electrical Power Engineering
South Valley University, Egypt

## License

This project is currently under active development.
