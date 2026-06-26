# Topic 10, Strategy Robustness and Project Guidance

> Why most backtests do not survive contact with live markets, what
> robustness checks separate a real strategy from a lucky one, and
> how to set up the final project so it lands well.

## The big idea

If you have followed the course so far, you can build a strategy
that produces a beautiful equity curve. That is the easy part. The
hard part is knowing whether the beautiful equity curve is real or
an artefact of how you set up the test. Strategies that look great
in backtest and lose money live are not rare. They are the default
outcome.

There is a saying that the goal of quant research is not to find
strategies that worked but to find strategies that continue to
work. The difference matters enormously. A strategy that worked on
2018-2020 SPY might fail on 2021-2024 SPY because the market regime
changed (low rates to high rates, calm to volatile, slow growth to
AI rally). Robustness is the property of holding up across regimes.

For the final project, the grader is not looking for a profitable
strategy. They are looking for a clean, defensible workflow with
correct bookkeeping, honest results, and clear interpretation. A
strategy that loses to buy-and-hold but does so correctly will get
full marks. A strategy that beats buy-and-hold with obvious
look-ahead bias will lose them.

## Key concepts

### Why most strategies fail in live trading

| Cause | Description |
|---|---|
| Overfitting | The strategy memorised noise from the training period. |
| Regime change | Market behaviour shifted; the strategy was built for the old regime. |
| Hidden costs | Spread, slippage, commissions add up to more than the gross edge. |
| Competition | Other people found the same signal and arbitraged it away. |
| Bad assumptions | Backtest used data that would not have been available at trade time. |
| Pure luck | The backtest period was favourable by chance. |

### Market regimes

A regime is a period during which the market behaves consistently.
Common regimes:

- Bull markets. Steady upward drift, low volatility.
- Bear markets. Steady downward drift.
- Crisis markets. Extreme volatility, correlations spike toward 1.
- Sideways markets. No trend, mean reversion thrives.

Different strategies suit different regimes. Trend following loses
money in choppy ranges. Mean reversion loses money in strong trends.
A truly robust strategy works at least roughly across all of them,
or has a regime detector that decides when to trade.

### The four warning signs of overfitting

The lecture's checklist:

1. Too many parameters being tuned simultaneously.
2. Too many ad-hoc rules ("but only on Wednesdays after 11am").
3. Equity curve looks unrealistically smooth.
4. In-sample Sharpe is dramatically higher than out-of-sample Sharpe.

If you see any of these, suspect overfitting. If you see two or
more, the strategy is almost certainly overfit.

### Out-of-sample testing and walk-forward

The minimum viable test is a single train-test split. The data is
divided into two parts. You only get to look at the test set once,
after all decisions are final. Numbers from the test set are the
honest estimate.

Walk-forward is the upgrade. Instead of one split, roll a window
across the entire dataset, retraining at each step. Report the
average out-of-sample performance. Walk-forward is slower but
catches regime sensitivity that a single split would miss.

### Final project rubric, broken down

The spec lists five weighted criteria:

| Component | Weight | What grader checks |
|---|---|---|
| Correctness | 30% | Code runs, math is right, no look-ahead bias. |
| Strategy logic | 25% | Signals match what the report describes. Choices are explained. |
| Code quality | 15% | Modular, readable, variables make sense. |
| Visualization & analysis | 15% | Charts are labelled, interpretation is sensible. |
| Documentation | 15% | Report covers the 10 required sections. README explains how to run. |

Notice that profitability is not on the list. A losing strategy
with strong fundamentals scores higher than a winning strategy with
weak fundamentals.

### Common project mistakes

- No benchmark. Always compare against buy-and-hold.
- No interpretation. Numbers without explanation are not analysis.
- Overfitted logic. Two strategies are enough; do not add five more.
- Missing charts. Every claim should have a chart.
- Weak conclusions. "Buy-and-hold won because the market went up"
  is fine. "Our strategy was profitable" without explaining why
  is not.

## One diagram

The full life cycle of a quant research project, from idea to
go/no-go decision:

```mermaid
flowchart LR
    A[Idea] --> B[Data]
    B --> C[Signal]
    C --> D[Backtest]
    D --> E[Validate]
    E --> F[Deploy or Drop]
```

## Code patterns

### A minimum walk-forward test

```python
import numpy as np

window_train = 252 * 2  # 2 years training
window_test  = 252 * 1  # 1 year test
sharpes = []
for start in range(0, len(df) - window_train - window_test, window_test):
    train = df.iloc[start : start + window_train]
    test  = df.iloc[start + window_train : start + window_train + window_test]
    # fit signal parameters on train (or just use fixed parameters)
    # apply signal to test
    # compute test sharpe
    sharpes.append(test_sharpe)
print("Mean OOS Sharpe:", np.mean(sharpes))
print("Worst window:",    np.min(sharpes))
```

### A robustness check, swapping the asset

```python
for ticker in ["SPY", "NVDA", "BTC-USD"]:
    df = load_ohlcv(ticker, "2018-01-01", "2024-12-31")
    df = build_indicators(df)
    sig = ma_crossover_signal(df)
    bt  = run_backtest(sig)
    m   = evaluate(bt)
    print(f"{ticker}: Sharpe={m.sharpe_ratio:.2f}, MaxDD={m.max_drawdown:.1%}")
```

A strategy that has Sharpe 0.8 on SPY, -0.2 on NVDA, and -0.5 on
BTC-USD probably worked on SPY by coincidence. A strategy that has
positive Sharpe across all three is more likely real.

## Common pitfalls

- Tuning parameters on the test set. The moment you look at the
  test set, change a parameter, and rerun, the test set has become
  part of the training set.
- Reporting only the in-sample Sharpe. If the report does not
  mention out-of-sample at all, the grader will assume there was
  none.
- Using a single asset and a single window. Even if the project
  rubric allows it, mentioning the limitation in the report
  demonstrates awareness.
- Hiding bad results. An honest "buy-and-hold beat both active
  strategies on Sharpe over this period" with a discussion of why
  is much stronger than a fudged result.

> The point of the project is to demonstrate that the workflow is
> correct, not that the strategy is profitable. Bookkeeping over
> bragging.

## How this shows up in our project

- The report's section 8 (Key Observations) explicitly states that
  buy-and-hold beat both active strategies on Sharpe, with an
  honest discussion of why.
- The report's section 9 (Limitations) lists single-asset and
  single-window as the top limitation.
- `verify.py` at the project root runs 49 invariant checks that
  catch the four biases (look-ahead, overfitting via fixed
  parameters, cost-regime consistency, metric formulas) before
  submission.
- The project does not currently include a walk-forward test. That
  is the single highest-leverage extension if you want to do more.

## Further reading

- `lectures/Knowledge_Base.md` Lecture 11 section.
- The project spec document `Quantitative Modelling Project.md` for
  the exact grading rubric.
