"""ZeroChart backend agent.

Fetches daily stock data, computes technical indicators, evaluates an
Asymmetric Risk-Reward signal, and upserts the result into a Supabase
``assets_status`` table.

Signal logic
------------
Filter A ("Stable"): long-term trend
    Green if 50-day SMA > 200-day SMA, else Red.

Filter B ("Radar"): asymmetric risk-reward trigger
    Green if ALL of the following hold:
        - reward/risk ratio >= 3.0
        - volume spike (3-day avg vs prior 14-day avg) > 1.5x
        - current price > 50-day SMA
    Otherwise Yellow.

Risk/Reward definitions
-----------------------
    Target  = max price (High) over the last 20 trading days
    Floor   = 50-day SMA
    risk %   = (current_price - 50_SMA) / current_price
    reward % = (target - current_price) / current_price
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from supabase import create_client

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT"]

# Technical indicator windows.
SMA_FAST = 10
SMA_SHORT = 20
SMA_TREND = 50
SMA_LONG = 200
RSI_PERIOD = 14

# Risk / reward parameters.
TARGET_WINDOW = 20
VOLUME_SPIKE_RECENT = 3
VOLUME_SPIKE_PRIOR = 14

# Filter B thresholds.
RADAR_MIN_RATIO = 3.0
RADAR_MIN_VOLUME_SPIKE = 1.5

TABLE_NAME = "assets_status"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("zerochart.agent")


# --------------------------------------------------------------------------- #
# Indicator helpers
# --------------------------------------------------------------------------- #


def compute_sma(close: pd.Series, window: int) -> pd.Series:
    """Return the simple moving average for the given window."""
    return close.rolling(window=window).mean()


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Return the Relative Strength Index using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def volume_spike_ratio(volume: pd.Series) -> float:
    """Ratio of the recent 3-day average volume vs the prior 14-day average."""
    if len(volume) < VOLUME_SPIKE_RECENT + VOLUME_SPIKE_PRIOR:
        return float("nan")

    recent = volume.iloc[-VOLUME_SPIKE_RECENT:].mean()
    prior = volume.iloc[-(VOLUME_SPIKE_RECENT + VOLUME_SPIKE_PRIOR):-VOLUME_SPIKE_RECENT].mean()

    if prior == 0:
        return float("nan")
    return float(recent / prior)


def safe_ratio(numerator: float, denominator: float) -> float:
    """Divide two floats, returning NaN when the denominator is ~zero."""
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return float("nan")
    return float(numerator / denominator)


def _num_or_none(value: float, ndigits: int = 6) -> float | None:
    """Round a float for storage, returning None when the value is NaN."""
    if pd.isna(value):
        return None
    return round(float(value), ndigits)


# --------------------------------------------------------------------------- #
# Signal evaluation
# --------------------------------------------------------------------------- #


def evaluate_stable(sma_50: float, sma_200: float) -> tuple[str, str]:
    """Filter A: long-term trend signal and explanation."""
    if pd.isna(sma_50) or pd.isna(sma_200):
        return "Red", "Insufficient history to compute the 50/200-day moving averages."

    if sma_50 > sma_200:
        signal = "Green"
        explanation = (
            f"The 50-day SMA ({sma_50:.2f}) is above the 200-day SMA ({sma_200:.2f}), "
            "confirming an upward long-term trend."
        )
    else:
        signal = "Red"
        explanation = (
            f"The 50-day SMA ({sma_50:.2f}) is at or below the 200-day SMA ({sma_200:.2f}), "
            "signaling a downward long-term trend."
        )
    return signal, explanation


def evaluate_radar(
    *,
    reward_risk_ratio: float,
    volume_spike: float,
    current_price: float,
    sma_50: float,
    reward_pct: float,
) -> tuple[str, str]:
    """Filter B: asymmetric risk-reward trigger and explanation."""
    price_above_sma = bool(current_price > sma_50) if not pd.isna(sma_50) else False
    ratio_ok = bool(reward_risk_ratio >= RADAR_MIN_RATIO) if not pd.isna(reward_risk_ratio) else False
    volume_ok = bool(volume_spike > RADAR_MIN_VOLUME_SPIKE) if not pd.isna(volume_spike) else False

    ratio_str = f"{reward_risk_ratio:.2f}x" if not pd.isna(reward_risk_ratio) else "n/a"
    volume_str = f"{volume_spike:.2f}x" if not pd.isna(volume_spike) else "n/a"

    if ratio_ok and volume_ok and price_above_sma:
        signal = "Green"
        explanation = (
            f"Radar triggered: reward/risk ratio is {ratio_str} (>= {RADAR_MIN_RATIO:.1f}x), "
            f"volume spike is {volume_str} (> {RADAR_MIN_VOLUME_SPIKE:.1f}x), and the price "
            f"({current_price:.2f}) is above its 50-day SMA."
        )
    else:
        signal = "Yellow"
        conditions = []
        conditions.append(f"reward/risk {ratio_str} (needs >= {RADAR_MIN_RATIO:.1f}x)")
        conditions.append(f"volume spike {volume_str} (needs > {RADAR_MIN_VOLUME_SPIKE:.1f}x)")
        conditions.append(f"price vs 50-day SMA: {'above' if price_above_sma else 'below'}")
        explanation = "Radar not triggered: " + "; ".join(conditions) + "."
    return signal, explanation


# --------------------------------------------------------------------------- #
# Per-ticker analysis
# --------------------------------------------------------------------------- #


def analyze_ticker(ticker: str) -> dict[str, Any] | None:
    """Fetch data and compute the full signal payload for a single ticker."""
    logger.info("Fetching 1y of daily data for %s", ticker)

    df = yf.download(
        ticker,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        logger.warning("No data returned for %s; skipping", ticker)
        return None

    # yf.download may return a MultiIndex on columns for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]
    volume = df["Volume"]
    high = df["High"]

    if len(close) < SMA_LONG:
        logger.warning(
            "%s has only %d rows (< %d); not enough data for the 200-day SMA",
            ticker,
            len(close),
            SMA_LONG,
        )

    current_price = float(close.iloc[-1])

    sma_10 = float(compute_sma(close, SMA_FAST).iloc[-1])
    sma_20 = float(compute_sma(close, SMA_SHORT).iloc[-1])
    sma_50 = float(compute_sma(close, SMA_TREND).iloc[-1])
    sma_200 = float(compute_sma(close, SMA_LONG).iloc[-1])
    rsi_14 = float(compute_rsi(close, RSI_PERIOD).iloc[-1])

    target = float(high.iloc[-TARGET_WINDOW:].max())
    spike = volume_spike_ratio(volume)

    risk_pct = safe_ratio(current_price - sma_50, current_price)
    reward_pct = safe_ratio(target - current_price, current_price)
    reward_risk_ratio = safe_ratio(reward_pct, risk_pct)

    stable_signal, stable_explanation = evaluate_stable(sma_50, sma_200)
    radar_signal, radar_explanation = evaluate_radar(
        reward_risk_ratio=reward_risk_ratio,
        volume_spike=spike,
        current_price=current_price,
        sma_50=sma_50,
        reward_pct=reward_pct,
    )

    logger.info(
        "%s | price=%.2f | SMA10=%.2f SMA20=%.2f SMA50=%.2f SMA200=%.2f RSI=%.2f | "
        "risk=%.2f%% reward=%.2f%% ratio=%.2fx | vol_spike=%.2fx | stable=%s radar=%s",
        ticker,
        current_price,
        sma_10,
        sma_20,
        sma_50,
        sma_200,
        rsi_14,
        risk_pct * 100,
        reward_pct * 100,
        reward_risk_ratio,
        spike,
        stable_signal,
        radar_signal,
    )

    return {
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "stable_signal": stable_signal,
        "stable_explanation": stable_explanation,
        "radar_signal": radar_signal,
        "radar_explanation": radar_explanation,
        "risk_pct": _num_or_none(risk_pct),
        "reward_pct": _num_or_none(reward_pct),
        "volume_spike": _num_or_none(spike),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# Supabase upsert
# --------------------------------------------------------------------------- #


def upsert_to_supabase(rows: list[dict[str, Any]]) -> None:
    """Upsert signal rows into Supabase, keyed on the ``ticker`` column."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error(
            "Missing Supabase credentials. Copy .env.example to .env and set "
            "SUPABASE_URL and SUPABASE_KEY before running."
        )
        raise SystemExit(1)

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # on_conflict="ticker" requires a UNIQUE constraint on the ticker column.
    response = client.table(TABLE_NAME).upsert(rows, on_conflict="ticker").execute()

    if hasattr(response, "error") and response.error:
        logger.error("Supabase upsert failed: %s", response.error)
        raise RuntimeError(f"Supabase upsert failed: {response.error}")

    logger.info("Upserted %d row(s) into %s", len(rows), TABLE_NAME)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    logger.info("ZeroChart agent starting for tickers: %s", ", ".join(TICKERS))

    rows: list[dict[str, Any]] = []
    for ticker in TICKERS:
        try:
            payload = analyze_ticker(ticker)
        except Exception:
            logger.exception("Failed to analyze %s", ticker)
            continue

        if payload is not None:
            rows.append(payload)

    if not rows:
        logger.error("No ticker produced a valid payload; nothing to upsert.")
        return

    upsert_to_supabase(rows)
    logger.info("ZeroChart agent finished successfully.")


if __name__ == "__main__":
    main()
