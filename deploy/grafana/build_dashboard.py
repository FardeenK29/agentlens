"""Generate the AgentLens Grafana dashboard JSON (run: python deploy/grafana/build_dashboard.py)."""

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
A = 'gen_ai_agent_name=~"$agent"'
_id = 0


def panel(title, kind, targets, x, y, w, h, unit=None, desc="", **extra):
    global _id
    _id += 1
    p = {
        "id": _id, "title": title, "type": kind, "datasource": DS, "description": desc,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"refId": chr(65 + i), "datasource": DS, "expr": e, "legendFormat": l, "range": True}
                    for i, (e, l) in enumerate(targets)],
        "fieldConfig": {"defaults": {"unit": unit or "short"}, "overrides": []},
        "options": {},
    }
    if kind == "stat":
        p["options"] = {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                        "colorMode": "value", "graphMode": "none", "textMode": "value"}
        p["fieldConfig"]["defaults"].setdefault("color", {"mode": "fixed", "fixedColor": "#73BF69"})
        p["targets"][0]["instant"] = True
        p["targets"][0]["range"] = False
    if kind == "timeseries":
        p["interval"] = "30s"
        p["fieldConfig"]["defaults"]["custom"] = {"lineWidth": 2, "fillOpacity": 10, "spanNulls": True,
                                                  "showPoints": "auto"}
        p["options"] = {"legend": {"displayMode": "table", "placement": "bottom", "calcs": ["mean", "max"]},
                        "tooltip": {"mode": "multi"}}
    p.update(extra)
    return p


def row(title, y):
    global _id
    _id += 1
    return {"id": _id, "type": "row", "title": title, "collapsed": False, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []}


runs = f'sum(increase(agentlens_agent_runs_total{{{A}}}[$__range]))'
tok = lambda t: f'sum(increase(gen_ai_client_token_usage_sum{{{A},gen_ai_token_type="{t}"}}[$__range]))'
projected_run_cost = f'(({tok("input")} * $price_in) + ({tok("output")} * $price_out)) / 1e6 / {runs}'

panels = [
    row("Overview (selected time range)", 0),
    panel("Agent runs", "stat", [(f"round({runs})", "")], 0, 1, 4, 4, desc="Completed graph invocations"),
    panel("Run success rate", "stat",
          [(f'sum(increase(agentlens_agent_runs_total{{{A},status="ok"}}[$__range])) / {runs}', "")],
          4, 1, 4, 4, unit="percentunit",
          fieldConfig={"defaults": {"unit": "percentunit", "thresholds": {"mode": "absolute", "steps": [
              {"color": "red", "value": None}, {"color": "orange", "value": 0.95}, {"color": "green", "value": 0.99}]}},
              "overrides": []}),
    panel("p95 run latency", "stat",
          [(f'histogram_quantile(0.95, sum by (le) (increase(agentlens_agent_run_duration_seconds_bucket{{{A}}}[$__range])))', "")],
          8, 1, 4, 4, unit="s"),
    panel("Tokens per run", "stat", [(f'({tok("input")} + {tok("output")}) / {runs}', "")], 12, 1, 4, 4,
          desc="Average input + output tokens per agent run"),
    panel("Tool error rate", "stat",
          [(f'sum(increase(agentlens_tool_calls_total{{{A},status="error"}}[$__range])) / sum(increase(agentlens_tool_calls_total{{{A}}}[$__range]))', "")],
          16, 1, 4, 4, unit="percentunit",
          fieldConfig={"defaults": {"unit": "percentunit", "thresholds": {"mode": "absolute", "steps": [
              {"color": "green", "value": None}, {"color": "orange", "value": 0.02}, {"color": "red", "value": 0.05}]}},
              "overrides": []}),
    panel("Projected cost / 1k runs", "stat", [(f'{projected_run_cost} * 1000', "")], 20, 1, 4, 4, unit="currencyUSD",
          desc="Measured tokens priced at the $price_in / $price_out per-1M-token rates set above. "
               "Use it to compare models or to see what a local model's traffic would cost on a hosted API."),

    row("Latency", 5),
    panel("Node latency p95", "timeseries",
          [(f'histogram_quantile(0.95, sum by (le, agentlens_node) (rate(agentlens_node_duration_seconds_bucket{{{A}}}[$__rate_interval])))', "{{agentlens_node}}")],
          0, 6, 8, 8, unit="s", desc="Time spent in each LangGraph node"),
    panel("LLM call latency p50 / p95", "timeseries",
          [(f'histogram_quantile(0.50, sum by (le, gen_ai_request_model) (rate(gen_ai_client_operation_duration_seconds_bucket{{{A}}}[$__rate_interval])))', "p50 {{gen_ai_request_model}}"),
           (f'histogram_quantile(0.95, sum by (le, gen_ai_request_model) (rate(gen_ai_client_operation_duration_seconds_bucket{{{A}}}[$__rate_interval])))', "p95 {{gen_ai_request_model}}")],
          8, 6, 8, 8, unit="s"),
    panel("Tool latency p95", "timeseries",
          [(f'histogram_quantile(0.95, sum by (le, gen_ai_tool_name) (rate(agentlens_tool_duration_seconds_bucket{{{A}}}[$__rate_interval])))', "{{gen_ai_tool_name}}")],
          16, 6, 8, 8, unit="s"),

    row("Tokens & cost", 14),
    panel("Token throughput", "timeseries",
          [(f'sum by (gen_ai_token_type) (rate(gen_ai_client_token_usage_sum{{{A}}}[$__rate_interval]))', "{{gen_ai_token_type}}")],
          0, 15, 8, 8, unit="short", desc="Tokens per second"),
    panel("Avg input tokens per LLM call, by node", "timeseries",
          [(f'sum by (agentlens_node) (rate(gen_ai_client_token_usage_sum{{{A},gen_ai_token_type="input"}}[$__rate_interval])) / '
            f'sum by (agentlens_node) (rate(gen_ai_client_token_usage_count{{{A},gen_ai_token_type="input"}}[$__rate_interval]))', "{{agentlens_node}}")],
          8, 15, 8, 8, desc="Growing context per call is the most common cost leak in agent loops"),
    panel("Spend (priced models) vs projected", "timeseries",
          [(f'sum(rate(agentlens_llm_cost_usd_total{{{A}}}[$__rate_interval])) * 3600 or vector(0)', "actual $/h"),
           (f'(sum(rate(gen_ai_client_token_usage_sum{{{A},gen_ai_token_type="input"}}[$__rate_interval])) * $price_in + '
            f'sum(rate(gen_ai_client_token_usage_sum{{{A},gen_ai_token_type="output"}}[$__rate_interval])) * $price_out) / 1e6 * 3600', "projected $/h")],
          16, 15, 8, 8, unit="currencyUSD"),

    row("Reliability", 23),
    panel("Tool calls by status", "timeseries",
          [(f'sum by (gen_ai_tool_name, status) (increase(agentlens_tool_calls_total{{{A}}}[$__rate_interval]))', "{{gen_ai_tool_name}} {{status}}")],
          0, 24, 12, 8, desc="Calls per interval", fieldConfig={"defaults": {"unit": "short", "custom": {"drawStyle": "bars", "stacking": {"mode": "normal"}, "fillOpacity": 80}}, "overrides": []}),
    panel("Error rate by tool", "bargauge",
          [(f'(sum by (gen_ai_tool_name) (increase(agentlens_tool_calls_total{{{A},status="error"}}[$__range])) '
            f'or 0 * sum by (gen_ai_tool_name) (increase(agentlens_tool_calls_total{{{A}}}[$__range]))) / '
            f'sum by (gen_ai_tool_name) (increase(agentlens_tool_calls_total{{{A}}}[$__range]))', "{{gen_ai_tool_name}}")],
          12, 24, 12, 8, unit="percentunit",
          options={"orientation": "horizontal", "displayMode": "basic", "showUnfilled": True, "namePlacement": "left", "reduceOptions": {"calcs": ["lastNotNull"]}},
          fieldConfig={"defaults": {"unit": "percentunit", "min": 0, "max": 0.3, "thresholds": {"mode": "absolute", "steps": [
              {"color": "green", "value": None}, {"color": "orange", "value": 0.02}, {"color": "red", "value": 0.05}]}}, "overrides": []}),
]
for p in panels:  # instant queries for bar gauges too
    if p["type"] == "bargauge":
        p["targets"][0].update({"instant": True, "range": False})

dashboard = {
    "uid": "agentlens", "title": "AgentLens - Agent Observability", "tags": ["llm", "agents", "opentelemetry"],
    "timezone": "browser", "schemaVersion": 39, "refresh": "10s", "time": {"from": "now-30m", "to": "now"},
    "links": [{"title": "Traces in Jaeger", "type": "link", "url": "http://localhost:16686/search?service=support-agent", "targetBlank": True, "icon": "external link"}],
    "templating": {"list": [
        {"name": "agent", "label": "Agent", "type": "query", "datasource": DS, "query": "label_values(agentlens_agent_runs_total, gen_ai_agent_name)",
         "includeAll": True, "multi": True, "current": {"text": "All", "value": "$__all"}, "refresh": 2},
        {"name": "price_in", "label": "Price in ($/1M)", "type": "textbox", "query": "0.15", "current": {"text": "0.15", "value": "0.15"}},
        {"name": "price_out", "label": "Price out ($/1M)", "type": "textbox", "query": "0.60", "current": {"text": "0.60", "value": "0.60"}},
    ]},
    "panels": panels,
}

out = Path(__file__).parent / "dashboards" / "agentlens.json"
out.write_text(json.dumps(dashboard, indent=2))
print(f"wrote {out} ({len(panels)} panels)")
