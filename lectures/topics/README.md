# Quantitative Modelling, Per-Topic Notes

A set of self-contained teaching chapters for the Quantitative
Modelling course. Each file is built from the matching deck, the
compiled `Knowledge_Base.md`, the lab notebooks, and our final
project's source code. Each chapter is meant to be read in 10 to 15
minutes and to leave you with both the intuition and the runnable
code patterns for that topic.

## Contents

| File | Topic | One-line summary |
|---|---|---|
| [01_intro_quant_research.md](01_intro_quant_research.md) | Introduction to Quantitative Research | What quant trading is and how the workflow goes from raw data to deployed strategy. |
| [02_ohlcv_microstructure.md](02_ohlcv_microstructure.md) | OHLCV Data and Market Microstructure | The five columns of market data and what happens inside the exchange when you place an order. |
| [03_time_series.md](03_time_series.md) | Financial Time Series | Turning prices into returns, volatility, and moving averages. |
| [04_alpha_signals.md](04_alpha_signals.md) | Alpha Signals and Quant Research | The main families of trading signals and how to tell signal from noise. |
| [05_backtesting.md](05_backtesting.md) | Backtesting Systems and Biases | How to test a strategy on history without lying to yourself, including the no-look-ahead rule. |
| [06_pairs_trading.md](06_pairs_trading.md) | Statistical Arbitrage and Pairs Trading | Trading the relationship between two cointegrated assets using a z-score on the spread. |
| [07_portfolio_risk.md](07_portfolio_risk.md) | Portfolio Construction and Risk Management | Combining strategies, sizing positions, and measuring portfolio-level risk. |
| [08_machine_learning.md](08_machine_learning.md) | Machine Learning for Trading | When ML helps, why most ML models fail in finance, and how to set up a model that has a chance. |
| [09_infrastructure_kdb.md](09_infrastructure_kdb.md) | Quant Infrastructure and KDB+/q | How market data flows from the exchange to the researcher and why KDB+ is built for time-series. |
| [10_robustness_project.md](10_robustness_project.md) | Strategy Robustness and Project Guidance | Why most backtests fail in live trading and what makes a final project land well. |

## Suggested reading order

Topics 01 to 05 are the foundation. Read them in order. They build
on each other directly: data, microstructure, returns and
indicators, signals, backtesting. After topic 05 you can build the
entire final project.

Topics 06 to 10 are extensions. Read them in whatever order
matches your interest. They each stand on their own:

- 06 (pairs trading) is the natural continuation of 04 (alpha signals).
- 07 (portfolio) is the natural continuation of 05 (backtesting).
- 08 (ML) is a different lens on the whole pipeline.
- 09 (infrastructure) is the systems-engineering layer most students
  will only need if they go into quant professionally.
- 10 (robustness) is the meta-topic and is worth re-reading at the
  end as a final checklist before submission.

## Companion documents

- `lectures/Knowledge_Base.md` is the compiled reference document
  for all 11 lectures. Use it as a formula lookup when you need a
  specific equation or definition.
- `Quantitative Modelling Project.md` at the project root is the
  final project brief and grading rubric.
- `lectures/Lecture_*.ipynb` are the original lab notebooks for
  hands-on examples.
- `notebook/project_notebook.ipynb` is our own end-to-end
  implementation of the full workflow on SPY.
