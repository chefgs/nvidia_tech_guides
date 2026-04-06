---
layout: page
title: "NVIDIA API Catalog Explained: Hosted AI Models and APIs for Developers"
permalink: /nvidia-api-catalog/api-catalog-intro/
---

# NVIDIA API Catalog Explained: Hosted AI Models and APIs for Developers

The **NVIDIA API Catalog**, accessible at [build.nvidia.com](https://build.nvidia.com), is NVIDIA's hosted platform for developers to discover, try, and integrate AI models and microservices through standard API endpoints. It is the cloud-hosted counterpart to NVIDIA NIM: where NIM lets you run AI inference microservices on your own GPU infrastructure, the API Catalog lets you call the same models over HTTPS without provisioning any hardware.

For developers building AI-powered applications, the API Catalog provides a fast path from "I want to try this model" to "I have code that calls this model" — in minutes, without a GPU, without container setup, and without infrastructure overhead.

> **The NVIDIA API Catalog is the fastest way to integrate NVIDIA-hosted AI models into applications. It is designed for prototyping and production API access to hundreds of AI models across multiple modalities.**

---

## What the NVIDIA API Catalog Provides

The catalog is organized into model and service categories that span the core AI application space:

### Large Language Models (LLMs)

The catalog hosts a wide range of open and proprietary large language models for text generation, instruction following, code generation, and reasoning:

- **Meta Llama series** — Llama 3.1, Llama 3.3, and newer variants (8B, 70B, 405B parameter sizes)
- **Mistral AI models** — Mistral 7B, Mistral Large, Mixtral MoE
- **Microsoft Phi** — Phi-3 Mini, Phi-3 Medium, Phi-4
- **Google Gemma** — Gemma 2B, Gemma 7B, Gemma 2
- **Qwen** — Alibaba's Qwen series
- **DeepSeek** — DeepSeek R1 and DeepSeek series
- **NVIDIA proprietary models** — NVIDIA Llama Nemotron, Minitron

All LLM endpoints expose an **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/completions`), so any application using the OpenAI Python SDK or HTTP client can switch to NVIDIA-hosted models with a base URL change.

### Vision Language Models (VLMs)

Vision-language models process both images and text, enabling:

- Image description and captioning
- Visual question answering
- Document and chart understanding
- Multi-image reasoning

Available models include **Llama 3.2 Vision**, **Microsoft Phi-3.5 Vision**, **Google PaliGemma**, **NVLM**, and others. VLM endpoints accept image inputs as base64-encoded data or URLs alongside text messages.

### Embedding Models

Embedding models convert text (and sometimes images) into dense vector representations for semantic search, retrieval, and clustering:

- **NVIDIA NV-Embed-v2** — NVIDIA's general-purpose text embedding model
- **NVIDIA NV-EmbedQA** — optimized for Q&A and retrieval use cases
- **Snowflake Arctic Embed** — high-performance open embedding model
- **E5 Mistral** — embedding model from Microsoft Research

Embedding endpoints follow the OpenAI `/v1/embeddings` format, making them compatible with LangChain, LlamaIndex, and other retrieval frameworks.

### Reranking Models

Rerankers take a query and a set of retrieved documents and re-score them for relevance — a crucial step in high-quality RAG pipelines:

- **NVIDIA NV-Rerankqa-Mistral-4B** — cross-encoder reranker optimized for Q&A
- Other reranking models tuned for specific domains

### Image Generation Models

Visual generative AI models for creating images from text prompts:

- **Stable Diffusion XL** — Stability AI's flagship text-to-image model
- **Stable Diffusion 3** — improved architecture for higher quality and prompt adherence
- **FLUX** — Black Forest Labs' diffusion model
- **Cosmos** — NVIDIA's world foundation model for physical AI simulation

### Speech and Audio Models

- **Parakeet** — NVIDIA's speech recognition models (ASR)
- **riva-asr** — Automatic speech recognition
- **riva-tts** — Text-to-speech synthesis

### Biology and Scientific Models

The catalog also includes domain-specific models for scientific applications:

- **ESM2** and **ESMFold** — protein sequence and structure models from Meta
- **AlphaFold2** — protein structure prediction from DeepMind
- **MolMIM** — molecular generation for drug discovery
- **DiffDock** — protein-ligand docking
- **NVIDIA BioNeMo** — foundation models for drug discovery

This makes the catalog valuable not just for conversational AI but for computational biology and scientific computing applications.

### Code Generation Models

- **CodeLlama** — Meta's code-focused Llama variant
- **StarCoder 2** — BigCode's state-of-the-art code generation model
- **Granite Code** — IBM's Granite coding models

---

## API Format and Compatibility

The NVIDIA API Catalog uses standard REST APIs:

### OpenAI-Compatible Chat Completions

```bash
curl -X POST "https://integrate.api.nvidia.com/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -d '{
    "model": "meta/llama-3.3-70b-instruct",
    "messages": [{"role": "user", "content": "Explain GPU parallelism in one paragraph."}],
    "max_tokens": 512
  }'
```

### Embeddings

```bash
curl -X POST "https://integrate.api.nvidia.com/v1/embeddings" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -d '{
    "input": "NVIDIA GPUs accelerate AI workloads",
    "model": "nvidia/nv-embed-v2",
    "encoding_format": "float"
  }'
```

### OpenAI Python SDK

Because the endpoints are OpenAI-compatible, the OpenAI Python SDK works with a simple base URL override:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="$NVIDIA_API_KEY"
)

response = client.chat.completions.create(
    model="meta/llama-3.3-70b-instruct",
    messages=[{"role": "user", "content": "What is RAPIDS?"}]
)
print(response.choices[0].message.content)
```

---

## Free Tier and Rate Limits

NVIDIA provides a **free tier** for the API Catalog:

- Each new account receives a number of free API credits
- Individual model pages on `build.nvidia.com` show a free "Try in Playground" interface — no API key required for browser-based testing
- Free tier API calls are rate-limited and credit-limited
- For production usage, paid plans are available through NVIDIA's commercial offerings

The free tier is sufficient for prototyping, integration testing, and evaluating whether a particular model meets your application's requirements before committing to infrastructure.

---

## API Catalog vs NIM: Choosing the Right Path

The API Catalog and NIM serve complementary purposes:

| Aspect | NVIDIA API Catalog | NVIDIA NIM (Self-hosted) |
|---|---|---|
| **Infrastructure** | NVIDIA-hosted — no GPUs needed | Your GPU infrastructure required |
| **Setup time** | Minutes (get API key, start calling) | Hours to days (containers, Kubernetes) |
| **Data privacy** | Data sent to NVIDIA's servers | Data stays in your environment |
| **Cost model** | Pay-per-token / subscription | Infrastructure + software licensing |
| **Customization** | Limited — use models as provided | Full control over model, runtime, config |
| **Latency** | Dependent on network + queue | Controlled by your hardware |
| **Best for** | Prototyping, development, variable workloads | Enterprise self-hosting, regulated data, high throughput |

The common pattern is to **prototype using the API Catalog**, then **deploy to self-hosted NIM** for production workloads requiring data privacy, consistent latency, or cost efficiency at scale.

---

## Integration with Frameworks

The NVIDIA API Catalog integrates directly with popular AI application frameworks:

### LangChain

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")
response = llm.invoke("What is cuDNN?")
```

### LlamaIndex

```python
from llama_index.llms.nvidia import NVIDIA

llm = NVIDIA(model="meta/llama-3.3-70b-instruct")
response = llm.complete("Explain TensorRT in simple terms.")
```

The `langchain-nvidia-ai-endpoints` and `llama-index-llms-nvidia` packages abstract away the API details and connect directly to the NVIDIA API Catalog backend.

---

## Summary

The NVIDIA API Catalog provides developers with fast, standards-compatible API access to hundreds of AI models across language, vision, speech, embeddings, and scientific domains — all without provisioning GPU infrastructure. Its OpenAI-compatible endpoints mean minimal friction for teams already using OpenAI-style tooling. The catalog serves as both a discovery platform for evaluating NVIDIA-hosted models and an integration point for building production applications that can later be migrated to self-hosted NIM deployments for enterprise requirements.

---

## References

1. [NVIDIA API Catalog — build.nvidia.com](https://build.nvidia.com)
2. [NVIDIA API Catalog Documentation](https://docs.api.nvidia.com/)
3. [NVIDIA NIM for Developers](https://developer.nvidia.com/nim)
4. [LangChain NVIDIA AI Endpoints](https://python.langchain.com/docs/integrations/providers/nvidia/)
5. [LlamaIndex NVIDIA Integration](https://docs.llamaindex.ai/en/stable/examples/llm/nvidia/)

---

[← Back to NVIDIA API Catalog](../) · [← Back to Home](../../)
