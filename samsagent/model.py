import ollama


class OllamaModel:
    def __init__(self, config):
        self.config = config
        self.client = ollama.AsyncClient(host=config.ollama_url, timeout=config.model_timeout)

    async def check(self):
        info = await self.client.show(self.config.model)
        if "tools" not in (info.capabilities or []):
            raise ValueError(f"Model {self.config.model!r} does not advertise tool support")
        limits = [v for k, v in (info.modelinfo or {}).items() if k.endswith(".context_length")]
        if limits and self.config.context_tokens > max(limits):
            raise ValueError("Configured context exceeds the model's context length")

    async def chat(self, messages, tools=None, output_tokens=None):
        response = await self.client.chat(
            model=self.config.model, messages=messages, tools=tools or None, stream=False,
            options={"num_ctx": self.config.context_tokens,
                     "num_predict": output_tokens or self.config.output_tokens},
            keep_alive="30m",
        )
        return response.model_dump(exclude_none=True)
