from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "rho"
    extraction_model: str = "Qwen/Qwen3-0.6B"  # vLLM path (rho.extraction.llm)
    temperature: float = 0.2
    # Extraction backend (P6): "ollama" or "vllm". The plan specifies vLLM +
    # Outlines, but that needs CUDA the calibration host does not have, so the
    # default is the Ollama path — the same deviation JD analysis (P4) and
    # rewriting (P5) already took. Set to "vllm" on a GPU host.
    extraction_backend: str = "ollama"
    extraction_model_ollama: str = "qwen2.5:14b"
    # JD analysis via Ollama (P4): the vLLM path in rho.jd.llm needs CUDA, which
    # the calibration host does not have. temperature is pinned to 0 at the call
    # site for reproducibility.
    jd_model: str = "qwen2.5:14b"
    ollama_base_url: str = "http://localhost:11434"
    # Rewriting (P5) runs on the same CUDA-less host as JD analysis, so it takes
    # the same Ollama path. Temperature is pinned at the call site (0.6): the
    # rewriter is meant to be creative, and the verification gate is what makes
    # that safe.
    rewrite_model: str = "qwen2.5:14b"
    # Matcher semantic bands (P3). Provisional defaults — never swept against a
    # labelled match set. Exposed here so P7 can tune them without code changes.
    sem_hi: float = 0.65  # >= this cosine counts a requirement "present"
    sem_lo: float = 0.45  # >= this (but < sem_hi) counts it "weak"


settings = Settings()
