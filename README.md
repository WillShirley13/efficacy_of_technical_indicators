# Efficacy of Technical Indicators

A data-driven investigation into whether common technical indicators (RSI, MACD, Bollinger Bands, etc.) have genuine predictive power over equity price movements, using machine learning.

## Approach

**Phase 1:** Train per-indicator models across 3 prediction horizons (3 / 10 / 20 trading days) to quantify each indicator's standalone predictive capability.

**Phase 2:** Aggregate top performers into a unified multi-indicator strategy model.

### Target Variable

Outcomes are labelled using **volatility-scaled triple barriers** so the definition of a "tradeable move" adapts to market conditions:

- **Label 1** — take-profit barrier hit first within N days
- **Label −1** — stop-loss barrier hit first
- **Label 0** — neither barrier hit (time-out)

Barrier distances scale with recent rolling volatility (`m × σ_N`), with `m` treated as a hyperparameter.

### Indicators

10 indicators spanning momentum (RSI, Stochastic), trend (SMA, EMA, ADX), volatility (Bollinger Bands, ATR), volume (OBV, Volume RoC), and hybrid (MACD) paradigms. Each uses horizon-matched parameters.

### Assets

Five major index ETFs: **SPY**, **QQQ**, **IWM**, **EFA**, **EEM**.

## Setup

**Requirements:** Python ≥ 3.11, MySQL

## Current Status

- Data pipeline (acquisition → cleaning → indicator computation) — **complete**
- Triple-barrier target generation — **complete**
- Feature engineering — **in progress** (6 of 10 indicators done)
- Modelling — **not started**
