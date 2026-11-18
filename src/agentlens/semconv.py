"""Attribute and metric names.

``gen_ai.*`` names follow the OpenTelemetry GenAI semantic conventions so the
data works with any backend that understands them. ``agentlens.*`` names are
agent-level extras the conventions don't cover yet (node timing, run cost).
"""

# --- GenAI semantic convention attributes ---
OPERATION_NAME = "gen_ai.operation.name"
PROVIDER_NAME = "gen_ai.provider.name"
REQUEST_MODEL = "gen_ai.request.model"
RESPONSE_MODEL = "gen_ai.response.model"
REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
REQUEST_TEMPERATURE = "gen_ai.request.temperature"
RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
RESPONSE_ID = "gen_ai.response.id"
USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
USAGE_CACHE_READ_TOKENS = "gen_ai.usage.cache_read.input_tokens"
USAGE_CACHE_WRITE_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
AGENT_NAME = "gen_ai.agent.name"
CONVERSATION_ID = "gen_ai.conversation.id"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
TOOL_CALL_RESULT = "gen_ai.tool.call.result"
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"
TOKEN_TYPE = "gen_ai.token.type"
DATA_SOURCE_ID = "gen_ai.data_source.id"
ERROR_TYPE = "error.type"

# Operation names
OP_CHAT = "chat"
OP_INVOKE_AGENT = "invoke_agent"
OP_EXECUTE_TOOL = "execute_tool"
OP_RETRIEVE = "retrieval"

# --- AgentLens extras ---
NODE = "agentlens.node"
COST_USD = "agentlens.cost.usd"
RUN_COST_USD = "agentlens.run.cost.usd"
RUN_LLM_CALLS = "agentlens.run.llm_calls"
RUN_TOOL_CALLS = "agentlens.run.tool_calls"
RETRIEVED_DOCS = "agentlens.retrieval.documents"

# --- Metric names ---
M_OP_DURATION = "gen_ai.client.operation.duration"
M_TOKEN_USAGE = "gen_ai.client.token.usage"
M_LLM_COST = "agentlens.llm.cost"
M_NODE_DURATION = "agentlens.node.duration"
M_TOOL_CALLS = "agentlens.tool.calls"
M_TOOL_DURATION = "agentlens.tool.duration"
M_RUNS = "agentlens.agent.runs"
M_RUN_DURATION = "agentlens.agent.run.duration"
M_RUN_COST = "agentlens.agent.run.cost"
