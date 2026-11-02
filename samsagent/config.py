from pathlib import Path
import tomllib

from pydantic import BaseModel, Field, model_validator


class MCPServer(BaseModel):
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def transport(self):
        if bool(self.command) == bool(self.url):
            raise ValueError("Specify exactly one of command or url")
        return self


class Config(BaseModel):
    ollama_url: str
    model: str
    data_dir: Path = Path("data")
    context_tokens: int = Field(16384, ge=4096)
    output_tokens: int = Field(2048, ge=256)
    inference_concurrency: int = Field(1, ge=1)
    agent_concurrency: int = Field(8, ge=2)
    max_depth: int = Field(3, ge=1)
    model_timeout: float = Field(300, gt=0)
    tool_timeout: float = Field(60, gt=0)
    port: int = Field(8765, ge=1024, le=65535)
    allowed_hosts: list[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"])
    mcp: dict[str, MCPServer] = Field(default_factory=dict)

    @model_validator(mode="after")
    def limits(self):
        if self.output_tokens >= self.context_tokens // 2:
            raise ValueError("output_tokens must be less than half the context window")
        if "machine" in self.mcp:
            raise ValueError("machine is reserved for the bundled MCP server")
        return self

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            config = cls(**tomllib.load(f))
        config.data_dir = config.data_dir.expanduser().resolve()
        return config

