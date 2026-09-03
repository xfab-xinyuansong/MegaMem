class TokenUsageCallback:
    def __init__(self):
        """Initialise empty token-usage counters and per-model cost table."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        # source_usage: {source: {model: {'prompt_tokens', 'completion_tokens', 'total_tokens'}}}
        self.source_usage = {}
        # model_usage: {model: {'prompt_tokens', 'completion_tokens', 'total_tokens'}}
        self.model_usage = {}

        # Optional cost table - USD per token, derived from per-million pricing.
        self.cost_per_token = {
            "chat_low": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
            "chat_high": {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
            "judge": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
        }

    def update(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        source: str = "",
    ):
        """Add a usage record for a (source, model) pair."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens

        per_source = self.source_usage.setdefault(source, {})
        per_source.setdefault(
            model,
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

        self.model_usage.setdefault(
            model,
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

        bucket = per_source[model]
        bucket["prompt_tokens"] += prompt_tokens
        bucket["completion_tokens"] += completion_tokens
        bucket["total_tokens"] += prompt_tokens + completion_tokens

        global_bucket = self.model_usage[model]
        global_bucket["prompt_tokens"] += prompt_tokens
        global_bucket["completion_tokens"] += completion_tokens
        global_bucket["total_tokens"] += prompt_tokens + completion_tokens

    def token_usage_report(self) -> str:
        """Render a human-readable usage + cost report.

        # Example Usage
            callback = TokenUsageCallback()
            callback.update(500, 200, model="chat_high", source="test1")
            callback.update(1000, 300, model="chat_low", source="test1")
            callback.update(800, 400, model="judge", source="test2")

            # Get the report as a string
            report = callback.generate_usage_report()
            print(report)  # You can print, log, or save it to a file
        """
        grand_total_cost = 0.0
        for model, usage in self.model_usage.items():
            if model in self.cost_per_token:
                grand_total_cost += (
                    usage["prompt_tokens"] * self.cost_per_token[model]["input"]
                    + usage["completion_tokens"] * self.cost_per_token[model]["output"]
                )

        lines = []
        lines.append("===========================")
        lines.append("      TOKEN USAGE REPORT   ")
        lines.append("===========================")
        lines.append("Total Prompt Tokens     : " + str(self.prompt_tokens))
        lines.append("Total Completion Tokens : " + str(self.completion_tokens))
        lines.append("Total Tokens            : " + str(self.total_tokens))
        lines.append("Total Estimated Cost    : $" + format(grand_total_cost, ".6f"))
        lines.append("\n==============================")
        lines.append("   BREAKDOWN BY SOURCE/MODEL  ")
        lines.append("==============================")

        per_source_cost = {}
        for source, models in self.source_usage.items():
            lines.append("\nSource: " + source)
            source_total = 0.0
            for model, usage in models.items():
                model_cost = 0.0
                if model in self.cost_per_token:
                    model_cost = (
                        usage["prompt_tokens"] * self.cost_per_token[model]["input"]
                        + usage["completion_tokens"]
                        * self.cost_per_token[model]["output"]
                    )
                source_total += model_cost
                lines.append("  - Model: " + model)
                lines.append("    Prompt Tokens    : " + str(usage["prompt_tokens"]))
                lines.append(
                    "    Completion Tokens: " + str(usage["completion_tokens"])
                )
                lines.append("    Total Tokens     : " + str(usage["total_tokens"]))
                lines.append("    Cost             : $" + format(model_cost, ".6f"))
            per_source_cost[source] = source_total
            lines.append(
                "  >> Total Cost for Source: $" + format(source_total, ".6f")
            )
            lines.append("--------------------------------")

        lines.append("\n===========================")
        lines.append("   COST BREAKDOWN BY MODEL")
        lines.append("===========================")
        for model, usage in self.model_usage.items():
            if model in self.cost_per_token:
                model_cost = (
                    usage["prompt_tokens"] * self.cost_per_token[model]["input"]
                    + usage["completion_tokens"] * self.cost_per_token[model]["output"]
                )
                lines.append("Model: " + model)
                lines.append("  Prompt Tokens    : " + str(usage["prompt_tokens"]))
                lines.append("  Completion Tokens: " + str(usage["completion_tokens"]))
                lines.append("  Total Tokens     : " + str(usage["total_tokens"]))
                lines.append("  Total Cost       : $" + format(model_cost, ".6f"))
                lines.append("---------------------------")

        return "\n".join(lines)
