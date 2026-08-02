"""Central configuration, loaded once from environment variables / .env."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Models
SUPERVISOR_MODEL = "gpt-4o-mini"
WORKER_MODEL = "gpt-4o-mini"
ASSEMBLER_MODEL = "gpt-4o-mini"

# Execution
MAX_WORKER_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
PARALLEL_WORKERS = True  # set False to run sequentially (comparison mode)

# Output
OUTPUT_DIR = "./outputs"
SAVE_RUN_LOG = True

# Pricing (USD per 1K tokens, gpt-4o-mini blended input+output estimate) -- used only
# for the CLI's rough cost estimate, not a billing-accurate figure.
EST_COST_PER_1K_TOKENS = 0.0003
