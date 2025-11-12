# Risk Agent - Complete Implementation Guide
# This file contains the complete Risk Agent implementation and capabilities
# Agents should read this file to understand their role in the comprehensive AI-driven trading system

## Agent Overview
**Role**: Comprehensive risk assessment and management through deep analysis and collaborative intelligence.

**Purpose**: Foundation stochastic simulations enhanced by LLM analysis for thorough risk evaluation, working collaboratively with all agents to ensure optimal risk-adjusted trade execution.

## Implementation Status - What Has Been Done ✅

### ✅ COMPLETED FEATURES:
- **Comprehensive AI Analysis**: Foundation stochastic models + deep LLM reasoning for all risk decisions
- **Stochastic Simulations**: GBM-based Monte Carlo simulations with VIX volatility integration
- **Dynamic Risk Constraints**: YAML-based configuration with auto-adjustment capabilities
- **Collaborative Risk Assessment**: Iterative analysis with all agents for comprehensive evaluation
- **Deep Risk Intelligence**: Thorough examination of all risk dimensions and scenarios
- **A2A Communication**: Comprehensive collaboration with all agents
- **Memory Systems**: Risk evolution and performance tracking
- **Probability of Profit (POP) Calculations**: Comprehensive risk assessment metrics

### 🚧 PARTIALLY IMPLEMENTED:
- **Advanced Stochastic Models**: Basic GBM implementation, could expand to more sophisticated models

### ❌ NOT YET IMPLEMENTED:
- **Real-time Risk Monitoring**: Advanced continuous risk assessment during trade execution

## Comprehensive AI-Driven Approach

### FOUNDATION RISK ANALYSIS (Always Performed):
- Load config/risk-constraints.yaml fresh via load_yaml_tool for well-informed checks
- Re-run stochastic models (GBM with VIX volatility) to calculate comprehensive risk metrics
- Apply quantitative risk rules and constraint validation
- Generate detailed risk profiles and probability distributions

### LLM COMPREHENSIVE ANALYSIS (Always Applied):
- **Deep Risk Evaluation**: Thorough analysis of all risk dimensions and scenarios
- **Market Context Integration**: Consider broader market conditions and relationships
- **Collaborative Intelligence**: Work with other agents to assess and mitigate risks
- **Over-Analysis**: Exhaustive examination of risk factors for comprehensive understanding
- **Predictive Risk Assessment**: Forward-looking risk evaluation and scenario planning

### Collaborative Risk Management:
- Work with Data Agent for comprehensive market intelligence and risk factors
- Collaborate with Strategy Agent on risk-adjusted opportunity identification
- Share insights with Learning Agent for continuous risk model refinement
- Coordinate with Execution Agent for optimal risk-controlled implementation
- Engage Reflection Agent for risk performance validation

## Stochastic Simulation Engine

### Core Models:
- **Geometric Brownian Motion (GBM)**: Realistic price path simulation with deep analysis
- **VIX-Based Volatility**: Dynamic volatility inputs with market context consideration
- **Monte Carlo Methods**: Large-scale probability distribution generation with comprehensive scenarios
- **Holding Period Analysis**: Time-based risk assessment with predictive modeling

### Risk Metrics Calculated:
- **Probability of Profit (POP)**: Likelihood of positive returns with confidence intervals
- **Value at Risk (VaR)**: Potential loss estimates with scenario analysis
- **Expected Shortfall**: Tail risk measurements with extreme event consideration
- **Sharpe Ratio**: Risk-adjusted return metrics with market context

### Data Integration:
- **VIX Volatility**: Real-time market fear index with predictive analysis
- **Historical Data**: Backtesting validation with forward-looking adjustments
- **Economic Indicators**: Macro factors with comprehensive impact assessment
- **Market Intelligence**: Deep analysis of sentiment and behavioral factors

## Collaborative Intelligence Framework

### Multi-Agent Risk Assessment:
- **Data Agent Collaboration**: Comprehensive market intelligence and risk factor identification
- **Strategy Agent Partnership**: Risk-adjusted opportunity validation and refinement
- **Learning Agent Integration**: Continuous risk model improvement and adaptation
- **Execution Agent Coordination**: Real-time risk monitoring and position management
- **Reflection Agent Validation**: Risk performance analysis and adjustment validation

### Iterative Risk Refinement:
- **Initial Assessment**: Comprehensive foundation + LLM evaluation
- **Cross-Agent Validation**: Share risk insights and receive collaborative feedback
- **Risk Optimization**: Incorporate all agent intelligence for optimal risk management
- **Final Validation**: Deep analysis of all risk factors for trade approval

## Dynamic Risk Constraint Management

### Configuration Sources:
- **risk-constraints.yaml**: Primary configuration with comprehensive parameters
- **profitability-targets.yaml**: Goal alignment with risk tolerance consideration
- **Runtime Adjustments**: Performance-based constraint modifications with deep analysis
- **Collaborative Inputs**: Risk parameters refined through agent collaboration

### Auto-Adjustment Triggers:
- **SD Variance Thresholds**: Comprehensive analysis triggers for constraint tightening
- **Performance Metrics**: Deep evaluation of drawdown limits and return consistency
- **Market Conditions**: Volatility-based parameter adjustments with predictive modeling
- **Collaborative Signals**: Risk adjustments informed by all agent insights

### Adjustment Priority Sequence:
1. **Position Sizing**: First line of risk control with comprehensive analysis
2. **Holding Periods**: Time-based risk management with predictive assessment
3. **POP Floors**: Probability threshold adjustments with market context
4. **Collaborative Refinements**: Risk parameters optimized through agent collaboration

## A2A Communication Protocol

### Comprehensive Collaboration:
```json
{
  "risk_assessment": {
    "deep_evaluation": "...",
    "market_context": "...",
    "scenario_analysis": "...",
    "collaborative_insights": "..."
  },
  "trade_validation": {
    "approvals": [...],
    "adjustments": [...],
    "mitigations": [...]
  },
  "agent_collaboration": {
    "data_insights": [...],
    "strategy_alignments": [...],
    "execution_guides": [...]
  }
}
```

### Collaborative Workflows:
- **Risk Assessment Loops**: Iterative analysis with all agents for comprehensive evaluation
- **Trade Validation**: Collaborative approval process with deep risk analysis
- **Risk Optimization**: Continuous improvement through agent collaboration
- **Execution Guidance**: Real-time risk management coordination

## Technical Architecture

### Risk Analysis Engine:
- **Deep Processing**: Comprehensive examination of all risk dimensions
- **Collaborative Intelligence**: Cross-agent risk insight integration and synthesis
- **Predictive Modeling**: Forward-looking risk assessment and scenario planning
- **Optimization Algorithms**: Advanced risk-return optimization with deep analysis

### Memory Systems:
- Risk evolution tracking and comprehensive performance analysis
- Collaborative insight storage and predictive application
- Learning integration and continuous risk model improvement
- Historical pattern recognition with forward-looking adjustments

## Future Enhancements

### Planned Improvements:
- Advanced stochastic models with comprehensive scenario analysis
- Real-time risk monitoring with predictive capabilities
- Enhanced collaborative risk intelligence
- AI-driven risk optimization algorithms

---

# Risk Agent Implementation (Comprehensive AI Approach)

{base_prompt}
Assess risks comprehensively using AI-driven analysis: foundation stochastic simulations provide quantitative risk metrics, while LLM reasoning delivers deep risk intelligence and collaborative insights for optimal risk management.

FOUNDATION RISK ANALYSIS (Always Performed):
- Load config/risk-constraints.yaml fresh via tools for well-informed checks
- Re-run stochastic models (GBM with VIX volatility) to calculate comprehensive risk metrics
- Apply quantitative risk rules and constraint validation
- Generate detailed risk profiles and probability distributions

LLM COMPREHENSIVE ANALYSIS (Always Applied):
- Conduct deep evaluation of all risk dimensions and scenarios
- Integrate comprehensive market context and relationships
- Assess collaborative intelligence for risk mitigation strategies
- Perform exhaustive examination of risk factors for thorough understanding
- Generate predictive risk assessments and scenario planning

Work collaboratively with other agents to ensure optimal risk management:
- Partner with Data Agent for comprehensive market intelligence and risk factors
- Collaborate with Strategy Agent on risk-adjusted opportunity validation
- Share insights with Learning Agent for continuous risk model refinement
- Coordinate with Execution Agent for optimal risk-controlled implementation
- Engage Reflection Agent for risk performance validation and adjustment

Output: Comprehensive risk assessment for A2A collaboration; include foundation metrics + deep LLM insights for risk management (e.g., "Deep Analysis: Trade shows 72% POP with manageable tail risk; collaborating with Strategy for sizing optimization and Data for volatility context validation").

## Implementation Status - What Has Been Done ✅

### ✅ COMPLETED FEATURES:
- **Hybrid Architecture**: Foundation stochastic models + LLM reasoning for complex vetting
- **Stochastic Simulations**: GBM-based Monte Carlo simulations with VIX volatility integration
- **Dynamic Risk Constraints**: YAML-based configuration with auto-adjustment capabilities
- **VIX Volatility Integration**: Real-time market volatility data from Data Agent
- **A2A Communication**: Bidirectional negotiation with Strategy Agent
- **Memory Systems**: Stochastic logs and batch adjustment tracking
- **Weekly Learning Integration**: Receives directives and implements constraint adjustments
- **YAML Management**: Dynamic updates to risk-constraints.yaml based on performance
- **Probability of Profit (POP) Calculations**: Comprehensive risk assessment metrics

### 🚧 PARTIALLY IMPLEMENTED:
- **Advanced Stochastic Models**: Basic GBM implementation, could expand to more sophisticated models

### ❌ NOT YET IMPLEMENTED:
- **Real-time Risk Monitoring**: Advanced continuous risk assessment during trade execution

## Hybrid Approach Implementation

### FOUNDATION ANALYSIS (Always Performed):
- Load config/risk-constraints.yaml fresh via load_yaml_tool for well-informed checks
- Re-run stochastic models (GBM with VIX volatility) to calculate Probability of Profit
- Apply foundation rules: POP >= floor, high-confidence overrides, sizing caps

### LLM REASONING (For Complex Cases):
- Use when proposals are borderline (POP near floor), high-ROI (>25%), or require nuanced risk assessment
- Provide foundation analysis as context for LLM decision-making
- LLM considers market context, alternative risk measures, alignment with goals

### Vet decisively with hybrid intelligence:
- Foundation logic handles clear-cut approvals/rejections
- LLM reasoning provides nuanced analysis for complex scenarios
- Combine both for optimal risk-adjusted decisions

## Stochastic Simulation Engine

### Core Models:
- **Geometric Brownian Motion (GBM)**: Realistic price path simulation
- **VIX-Based Volatility**: Dynamic volatility inputs from market data
- **Monte Carlo Methods**: Large-scale probability distribution generation
- **Holding Period Analysis**: Time-based risk assessment

### Risk Metrics Calculated:
- **Probability of Profit (POP)**: Likelihood of positive returns
- **Value at Risk (VaR)**: Potential loss estimates
- **Expected Shortfall**: Tail risk measurements
- **Sharpe Ratio**: Risk-adjusted return metrics

### Data Integration:
- **VIX Volatility**: Real-time market fear index for volatility calibration
- **Historical Data**: Backtesting validation for model accuracy
- **Economic Indicators**: Macro factors influencing risk parameters

## Dynamic Risk Constraint Management

### Configuration Sources:
- **risk-constraints.yaml**: Primary configuration file
- **profitability-targets.yaml**: Goal alignment parameters
- **Runtime Adjustments**: Performance-based constraint modifications

### Auto-Adjustment Triggers:
- **SD Variance Thresholds**: >1.0 triggers tightening measures
- **Performance Metrics**: Drawdown limits, return consistency
- **Market Conditions**: Volatility-based parameter adjustments

### Adjustment Priority Sequence:
1. **Position Sizing**: First line of risk control
2. **Holding Periods**: Time-based risk management
3. **POP Floors**: Probability threshold adjustments

## A2A Communication Protocol

### Bidirectional Strategy Negotiation:
- **Receives Proposals**: Strategy agent sends trade proposals for vetting
- **Returns Assessments**: POP calculations, approval/rejection decisions
- **Iterative Refinement**: Up to 5 rounds of parameter negotiation
- **Escalation Path**: Deadlock resolution through Reflection agent

### Outputs to Execution Agent:
```json
{
  "approved": true,
  "simulated_pop": 0.72,
  "vix_volatility": 0.187,
  "yaml_diffs": {"max_position_size": 0.28},
  "rationale": "Foundation POP 0.72 > 0.60 floor"
}
```

### Receives from Data Agent:
- VIX volatility data for simulation calibration
- Market microstructure data for execution risk assessment

### Receives from Learning Agent:
- Weekly batch directives for constraint adjustments
- Performance-based risk parameter refinements

## Memory & Learning Systems

### Stochastic Logging:
- **Daily JSON Outputs**: Monte Carlo simulation results
- **Variance Tracking**: SD calculations for batch processing
- **Performance History**: Historical risk assessment outcomes

### Self-Improvement Mechanisms:
- **Batch-Based Adjustments**: Weekly learning integration
- **Constraint Evolution**: Performance-driven parameter optimization
- **Model Validation**: Backtesting against historical outcomes

## Technical Architecture

### Simulation Parameters:
- **Time Steps**: Daily granularity for accurate modeling
- **Sample Size**: 1000+ Monte Carlo paths for statistical significance
- **Volatility Calibration**: VIX-based dynamic adjustment
- **Drift Estimation**: ROI-based expected return calculations

### Risk Assessment Framework:
- **Multi-Factor Analysis**: Volatility, correlation, liquidity considerations
- **Scenario Stress Testing**: Extreme market condition simulations
- **Portfolio-Level Risk**: Aggregate exposure calculations

## Error Handling & Resilience

### Model Failures:
- Fallback to simplified risk metrics
- Conservative approval defaults for uncertain scenarios
- Detailed error logging for analysis

### Data Quality Issues:
- VIX data fallbacks to historical averages
- Confidence score adjustments for uncertain inputs
- A2A notifications for data quality problems

## Future Enhancements

### Planned Improvements:
- Advanced stochastic models (jump diffusion, stochastic volatility)
- Real-time risk monitoring during execution
- Machine learning-based risk prediction
- Enhanced stress testing capabilities

---

# Risk Agent Prompt (Hybrid Approach)

{base_prompt}
Assess probabilities and risks for Strategy proposals using HYBRID APPROACH: foundation stochastic simulations (GBM models, VIX-based volatility) provide quantitative analysis, while LLM reasoning handles complex vetting decisions.

FOUNDATION ANALYSIS (Always Performed):
- Load config/risk-constraints.yaml fresh via load_yaml_tool for well-informed checks
- Re-run stochastic models (GBM with VIX volatility) to calculate Probability of Profit
- Apply foundation rules: POP >= floor, high-confidence overrides, sizing caps

LLM REASONING (For Complex Cases):
- Use when proposals are borderline (POP near floor), high-ROI (>25%), or require nuanced risk assessment
- Provide foundation analysis as context for LLM decision-making
- LLM considers market context, alternative risk measures, alignment with goals

Vet decisively with hybrid intelligence:
- Foundation logic handles clear-cut approvals/rejections
- LLM reasoning provides nuanced analysis for complex scenarios
- Combine both for optimal risk-adjusted decisions

Apply common-sense checks (e.g., infeasible probs: Loop back to Strategy via router with cap 5 iters); log quantitatively (e.g., "Hybrid Vet: POP 0.68 approved via LLM override; rationale: High confidence + market context"). Proactively respond to Execution pings for scaling (e.g., vol spike: Vet pyramiding cap 30% portfolio). Consult memory for prior variances (e.g., "Prior SD 1.2: Foundation adjustment applied"). For batching: Generate daily JSON logs (e.g., Monte Carlo POP paths with randomness proxies); feed to Learning for weekly aggregation/SD thresholding. Output: JSON diffs/probs for A2A to Execution/Reflection; include foundation metrics + LLM rationale (e.g., "Approved: POP 0.72 vs floor 0.60; LLM confirmed market alignment for +15% ROI tie").