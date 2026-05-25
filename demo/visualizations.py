"""
Plotly visualizations for the P4 counterfactual demo.

All figures are pure functions over the existing demo state — they read
from the model + DataFrames and return go.Figure objects. No Streamlit
imports here; app.py wraps them with st.plotly_chart.

Functions:
    risk_gauge(value, title, baseline=None)
        Single risk gauge with traffic-light zones + population base-rate
        threshold marker. When `baseline` is provided, shows delta below
        the value.

    feature_delta_bar(delta_df)
        Horizontal bar chart of feature deltas, color-coded by healthy
        direction per the BRFSS feature taxonomy.

    risk_waterfall(predict_fn, query, best_cf, feature_order)
        Cumulative-attribution waterfall: starting from the baseline risk,
        apply each changed feature one at a time and plot the resulting
        per-step delta. Order is feature_order (deterministic); the total
        reduction is order-invariant.
"""
from __future__ import annotations

from typing import Callable, Iterable

import pandas as pd
import plotly.graph_objects as go


# Population base rate (BRFSS 2021 test split). Drawn as a threshold line
# on the risk gauges so the audience sees "where the average sits".
POPULATION_BASE_RATE = 0.142

# Healthy-direction colour. CFs respect the per-query taxonomy so all
# changes should already be in the healthy direction.
COLOR_HEALTHY = "#2ca02c"       # green
COLOR_UNHEALTHY = "#d62728"     # red (defensive — should not normally trigger)
COLOR_TOTAL = "#1f77b4"         # blue (waterfall totals)


# Feature mutability for the bar-chart colour logic. Mirrors the
# Mutability classes in src/.../feature_taxonomy.py. Kept here as a local
# constant so visualizations.py has zero import dependency on src/ (it
# can run in any env that has plotly + pandas).
_HEALTHY_DOWN = {
    "HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack", "DiffWalk",
    "Smoker", "HvyAlcoholConsump", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth",
}
_HEALTHY_UP = {
    "PhysActivity", "Fruits", "Veggies",
    "AnyHealthcare", "CholCheck",
    "Education", "Income",
}
# BMI is BIDIRECTIONAL — healthy is moving toward 18.5–24.9 normal range.


def _direction_color(feature: str, current: float, cf_value: float) -> str:
    """Return the bar colour for one feature change. Green = healthy."""
    delta = cf_value - current
    if delta == 0:
        return COLOR_HEALTHY  # not used (zero-delta rows are filtered)
    if feature in _HEALTHY_DOWN:
        return COLOR_HEALTHY if delta < 0 else COLOR_UNHEALTHY
    if feature in _HEALTHY_UP:
        return COLOR_HEALTHY if delta > 0 else COLOR_UNHEALTHY
    if feature == "BMI":
        # Healthy = moving toward normal range (18.5–24.9). Use 22 as midpoint.
        return COLOR_HEALTHY if abs(cf_value - 22) < abs(current - 22) else COLOR_UNHEALTHY
    # Immutable (Age, Sex) — should not appear in deltas.
    return COLOR_UNHEALTHY


# ─────────────────────────────────────────────────────────────────────
# 1. Risk gauge
# ─────────────────────────────────────────────────────────────────────
def risk_gauge(
    value: float,
    title: str,
    baseline: float | None = None,
    height: int = 220,
) -> go.Figure:
    """
    Plotly risk gauge with traffic-light zones [0,0.2] green, (0.2,0.5]
    yellow, (0.5,1] red, plus a threshold marker at the population base
    rate (0.142). When `baseline` is given, the indicator shows the
    signed delta below the value (green if decreasing — desired direction
    for desired_class=0).
    """
    indicator_mode = "gauge+number"
    delta = None
    if baseline is not None:
        indicator_mode += "+delta"
        delta = {
            "reference": baseline,
            "decreasing": {"color": "green"},
            "increasing": {"color": "red"},
            "valueformat": ".3f",
        }

    fig = go.Figure(
        go.Indicator(
            mode=indicator_mode,
            value=value,
            delta=delta,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 14}},
            number={"valueformat": ".3f", "font": {"size": 32}},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1, "tickfont": {"size": 10}},
                "bar": {"color": "#1f3a5f", "thickness": 0.6},
                "steps": [
                    {"range": [0.0, 0.2], "color": "#d4edda"},
                    {"range": [0.2, 0.5], "color": "#fff3cd"},
                    {"range": [0.5, 1.0], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.85,
                    "value": POPULATION_BASE_RATE,
                },
            },
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# 2. Feature delta bar chart
# ─────────────────────────────────────────────────────────────────────
def feature_delta_bar(delta_df: pd.DataFrame, height: int | None = None) -> go.Figure | None:
    """
    Horizontal bar chart of feature changes. Each bar's length is the raw
    Δ (CF − current), so direction is encoded by left/right of the zero
    line and magnitude by length. Colour encodes healthy/unhealthy
    direction per the BRFSS taxonomy (CFs from per-query mode should be
    all-healthy).

    Returns None if delta_df is empty.
    """
    if delta_df.empty:
        return None

    if height is None:
        height = 80 + 38 * len(delta_df)

    colors = [
        _direction_color(row["feature"], float(row["current"]), float(row["counterfactual"]))
        for _, row in delta_df.iterrows()
    ]

    # Text on bars: "current → CF" pair. Bars going left (negative delta)
    # show text to the LEFT of the bar; positive to the RIGHT.
    text_labels = [
        f"{row['current']:.3g} → {row['counterfactual']:.3g}"
        for _, row in delta_df.iterrows()
    ]

    fig = go.Figure(
        go.Bar(
            y=delta_df["feature"],
            x=delta_df["delta"],
            orientation="h",
            marker_color=colors,
            text=text_labels,
            textposition="auto",
            insidetextanchor="middle",
            hovertemplate="<b>%{y}</b><br>Δ = %{x:.3g}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=40),
        xaxis=dict(
            title="Δ (counterfactual − current)",
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=1,
        ),
        yaxis=dict(autorange="reversed"),  # first feature on top
        bargap=0.25,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# 3. Risk waterfall (cumulative attribution)
# ─────────────────────────────────────────────────────────────────────
def risk_waterfall(
    predict_fn: Callable[[pd.DataFrame], float],
    query: pd.Series,
    best_cf: pd.Series,
    feature_order: Iterable[str],
    height: int = 360,
) -> go.Figure | None:
    """
    Waterfall: P(Diabetes=1) starting at the baseline, then the marginal
    change from applying each differing feature one at a time, ending at
    the best-CF risk.

    Order along the x-axis is `feature_order`. Note the *total* reduction
    is order-invariant; the per-step deltas are not — they depend on the
    order because the model is non-additive. The caption in app.py flags
    this caveat.

    Args:
        predict_fn: callable taking a 1-row DataFrame in feature_order
            and returning P(Diabetes=1) as float.
        query:    Patient series.
        best_cf:  Best counterfactual series (post-rounding).
        feature_order: column order the model expects.

    Returns None if no features differ.
    """
    feature_order = list(feature_order)
    changed = [f for f in feature_order if f in best_cf.index and query[f] != best_cf[f]]
    if not changed:
        return None

    # Start from the patient, apply changes incrementally.
    current = query.copy()
    baseline_risk = predict_fn(pd.DataFrame([current])[feature_order])
    risks = [baseline_risk]
    deltas: list[float] = []
    for f in changed:
        current[f] = best_cf[f]
        risk = predict_fn(pd.DataFrame([current])[feature_order])
        deltas.append(risk - risks[-1])
        risks.append(risk)

    labels = ["Baseline"] + changed + ["Best CF"]
    y_values = [baseline_risk] + deltas + [None]
    measure = ["absolute"] + ["relative"] * len(changed) + ["total"]
    # Annotation text above each bar
    text_labels = (
        [f"{baseline_risk:.3f}"]
        + [f"{d:+.3f}" for d in deltas]
        + [f"{risks[-1]:.3f}"]
    )

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measure,
            x=labels,
            y=y_values,
            text=text_labels,
            textposition="outside",
            connector={"line": {"color": "lightgray"}},
            decreasing={"marker": {"color": COLOR_HEALTHY}},
            increasing={"marker": {"color": COLOR_UNHEALTHY}},
            totals={"marker": {"color": COLOR_TOTAL}},
        )
    )
    fig.update_layout(
        height=height,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=60),
        yaxis=dict(title="P(Diabetes = 1)", range=[0, max(risks) * 1.15]),
        xaxis=dict(tickangle=-25),
    )
    return fig
