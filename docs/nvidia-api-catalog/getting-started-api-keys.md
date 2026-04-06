---
layout: page
title: "Getting Started with NVIDIA APIs: Organizations, API Keys, and Your First API Call"
permalink: /nvidia-api-catalog/getting-started-api-keys/
---

# Getting Started with NVIDIA APIs: Organizations, API Keys, and Your First API Call

This guide walks through the complete process of accessing the NVIDIA API Catalog: creating an account, setting up an organization, generating API keys, and making your first authenticated API call. The NVIDIA API Catalog is hosted at [build.nvidia.com](https://build.nvidia.com) and provides access to hundreds of AI models through standard REST APIs.

---

## Step 1: Create an NVIDIA Account

Before you can generate API keys, you need an NVIDIA account.

1. Go to [build.nvidia.com](https://build.nvidia.com) and click **Sign In** or **Get Started**.
2. If you do not have an account, click **Create Account** to register.
3. You can sign up using:
   - Email address and password
   - Sign in with Google
   - Sign in with GitHub
4. Complete email verification if prompted.

You can also create an account at [developer.nvidia.com](https://developer.nvidia.com) — the same credentials work across both sites.

---

## Step 2: Explore the API Catalog

Once signed in, you arrive at the NVIDIA API Catalog on [build.nvidia.com](https://build.nvidia.com).

The catalog is organized by model category:

- **Chat** — Large language models for text generation and instruction following
- **Embedding** — Text and multimodal embedding models for semantic search and retrieval
- **Reranking** — Cross-encoder models for ranking retrieved documents
- **Image** — Text-to-image generation models
- **Vision** — Vision-language models that accept image and text input
- **Speech** — Speech recognition and text-to-speech models
- **Biology** — Protein folding, molecular generation, and drug discovery models
- **3D / Simulation** — Models for 3D understanding and physical world simulation

Each model card shows:
- Model name and provider
- A brief description and key capabilities
- A **Try Now** or **API** button for direct access

Before generating an API key, you can try any model in the browser playground — no key required. This is useful for quick evaluation.

---

## Step 3: Create an Organization

NVIDIA's API platform supports **organizations** — shared workspaces where teams can collaborate, share API keys, manage billing, and set usage policies.

### Why Use an Organization?

- **Centralized billing**: Usage charges are tracked at the organization level, not per individual user
- **Team access management**: Invite team members and control their roles
- **Shared API keys**: Keys created under an organization are available to all authorized members
- **Usage monitoring**: View consumption across your team in one place
- **Enterprise controls**: For paid plans, organizations are required for invoicing and support

### How to Create an Organization

1. After signing in, click your profile avatar or name in the top-right corner.
2. Select **Organizations** or **My Organization** from the dropdown menu.
3. Click **Create Organization** or **New Organization**.
4. Fill in the organization details:
   - **Organization Name**: Your company or project name (e.g., `Acme AI Team`)
   - **Organization Slug**: A URL-friendly identifier (e.g., `acme-ai`)
   - **Description** (optional): Brief description of your team or use case
5. Click **Create** to create the organization.

You are automatically assigned the **Owner** role in the organization.

### Inviting Team Members

To add collaborators to your organization:

1. Navigate to **Organization Settings** → **Members**.
2. Click **Invite Members**.
3. Enter the email address of the person to invite.
4. Select their role:
   - **Owner**: Full administrative access — can manage billing, keys, and members
   - **Admin**: Can manage keys and invite members but not change billing
   - **Member**: Can use the API catalog and view organization keys
5. Send the invitation. The invitee receives an email to join.

---

## Step 4: Generate an API Key

API keys authenticate your requests to the NVIDIA API endpoints. Keys can be scoped to a personal account or to an organization.

### Generating a Personal API Key

1. Sign in to [build.nvidia.com](https://build.nvidia.com).
2. Click your profile avatar → **API Keys** (or navigate to any model page and click **Get API Key**).
3. Click **Generate API Key** or **+ New Key**.
4. Optionally, give the key a descriptive name (e.g., `dev-laptop`, `ci-pipeline`, `rag-prototype`).
5. Click **Generate**.
6. **Copy the key immediately** — the full key value is only displayed once. Store it securely.

### Generating an Organization API Key

1. Sign in and navigate to your organization via the top-right profile menu.
2. Go to **Organization Settings** → **API Keys**.
3. Click **Generate API Key**.
4. Give the key a name that identifies its purpose (e.g., `prod-backend`, `data-team`, `staging`).
5. Optionally set an **expiry date** for security compliance.
6. Click **Generate** and copy the key securely.

### API Key Security Best Practices

- **Never commit API keys to source code or version control.** Use environment variables or secrets management tools instead.
- Store keys in a secrets manager (AWS Secrets Manager, HashiCorp Vault, GitHub Secrets, etc.) for team environments.
- Rotate keys regularly — generate a new key before revoking the old one to avoid downtime.
- Use separate keys for development, staging, and production environments so you can revoke one without affecting others.
- Name keys descriptively so you can identify which application or service is using each key.

---

## Step 5: Set Your API Key in the Environment

The standard way to use your API key is via an environment variable:

### Linux / macOS

```bash
export NVIDIA_API_KEY="nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Add this line to your `~/.bashrc`, `~/.zshrc`, or `~/.profile` to make it persistent across terminal sessions.

### Windows (Command Prompt)

```cmd
set NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Windows (PowerShell)

```powershell
$env:NVIDIA_API_KEY = "nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### In Python (using python-dotenv)

Create a `.env` file in your project root (add `.env` to `.gitignore`):

```
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Load it in your Python code:

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ["NVIDIA_API_KEY"]
```

---

## Step 6: Make Your First API Call

With your API key ready, you can make authenticated calls to any model in the catalog.

The base URL for NVIDIA API calls is:

```
https://integrate.api.nvidia.com/v1
```

### Using curl

```bash
curl -X POST "https://integrate.api.nvidia.com/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -d '{
    "model": "meta/llama-3.3-70b-instruct",
    "messages": [
      {
        "role": "user",
        "content": "What is NVIDIA CUDA and why does it matter for AI?"
      }
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

### Using Python (OpenAI SDK)

The endpoints are OpenAI-compatible, so the OpenAI Python SDK works without modification:

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

response = client.chat.completions.create(
    model="meta/llama-3.3-70b-instruct",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI assistant specializing in NVIDIA technologies."
        },
        {
            "role": "user",
            "content": "Explain the difference between NIM and TensorRT in simple terms."
        }
    ],
    max_tokens=512,
    temperature=0.6
)

print(response.choices[0].message.content)
```

### Using Python (Embeddings)

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

response = client.embeddings.create(
    model="nvidia/nv-embed-v2",
    input="NVIDIA RAPIDS accelerates data science on GPUs",
    encoding_format="float"
)

embedding_vector = response.data[0].embedding
print(f"Embedding dimensions: {len(embedding_vector)}")
```

### Installing Required Packages

```bash
pip install openai python-dotenv
```

---

## Step 7: Find Model Names in the Catalog

Each model in the catalog has an identifier in the format `provider/model-name`. You can find the correct identifier for any model:

1. Go to [build.nvidia.com](https://build.nvidia.com) and browse or search for a model.
2. Click the model card to open its detail page.
3. Click the **API** tab to see sample code — the model name appears in the request body.
4. Alternatively, click **Get API Key** to see the full quick-start code snippet pre-filled with the model identifier.

Example model identifiers:

| Model | Identifier |
|---|---|
| Llama 3.3 70B Instruct | `meta/llama-3.3-70b-instruct` |
| Mistral 7B Instruct | `mistralai/mistral-7b-instruct-v0.3` |
| Mixtral 8x7B Instruct | `mistralai/mixtral-8x7b-instruct-v0.1` |
| Microsoft Phi-3 Medium | `microsoft/phi-3-medium-4k-instruct` |
| DeepSeek R1 | `deepseek-ai/deepseek-r1` |
| NV-Embed-v2 (embeddings) | `nvidia/nv-embed-v2` |
| Stable Diffusion XL | `stability-ai/sdxl` |
| Llama 3.2 11B Vision | `meta/llama-3.2-11b-vision-instruct` |

---

## Step 8: Streaming Responses

For chat completions, streaming returns tokens as they are generated rather than waiting for the full response. This is important for responsive user interfaces:

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

stream = client.chat.completions.create(
    model="meta/llama-3.3-70b-instruct",
    messages=[{"role": "user", "content": "Explain GPU memory hierarchy in detail."}],
    max_tokens=1024,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Step 9: Managing Usage and Billing

### Viewing Usage

1. Sign in to [build.nvidia.com](https://build.nvidia.com).
2. Navigate to your profile or organization settings.
3. Select **Usage** or **Billing** from the navigation.
4. View token consumption broken down by model, date range, and API key.

### Understanding Rate Limits

The free tier has rate limits per model:

- Requests per minute (RPM) — typically 5–10 RPM on the free tier
- Tokens per minute (TPM)
- Requests per day (RPD)

Rate limit errors return HTTP `429 Too Many Requests`. Implement exponential backoff retry logic:

```python
import time
import openai

def call_with_retry(client, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except openai.RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
```

### Upgrading to a Paid Plan

For production workloads, upgrade from the free tier through the NVIDIA Developer portal:

1. Go to your organization settings → **Billing**.
2. Select a plan appropriate for your usage volume.
3. Add a payment method.
4. Higher-tier plans provide higher rate limits, priority access, and enterprise support options.

---

## Step 10: Integration with LangChain and LlamaIndex

For building RAG systems, agents, or pipelines, the native NVIDIA integrations in popular frameworks are the most convenient path.

### LangChain

```bash
pip install langchain-nvidia-ai-endpoints
```

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
import os

# Chat model
llm = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    api_key=os.environ["NVIDIA_API_KEY"]
)
response = llm.invoke("What are the key benefits of NVIDIA NIM?")
print(response.content)

# Embedding model
embedder = NVIDIAEmbeddings(
    model="nvidia/nv-embed-v2",
    api_key=os.environ["NVIDIA_API_KEY"]
)
vectors = embedder.embed_documents(["RAPIDS", "cuDNN", "TensorRT"])
```

### LlamaIndex

```bash
pip install llama-index-llms-nvidia llama-index-embeddings-nvidia
```

```python
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
import os

llm = NVIDIA(
    model="meta/llama-3.3-70b-instruct",
    api_key=os.environ["NVIDIA_API_KEY"]
)

embed_model = NVIDIAEmbedding(
    model="nvidia/nv-embed-v2",
    api_key=os.environ["NVIDIA_API_KEY"]
)
```

---

## Transitioning from API Catalog to Self-Hosted NIM

The NVIDIA API Catalog is designed as a stepping stone. When your application is ready for production requirements — data privacy, predictable latency, cost efficiency at scale — you can migrate to self-hosted NIM with minimal code changes.

The key migration steps are:

1. **Deploy the NIM container** for your target model on your GPU infrastructure (see [NVIDIA NIM documentation](https://docs.nvidia.com/nim/))
2. **Update the base URL** in your code from `https://integrate.api.nvidia.com/v1` to your local NIM endpoint (e.g., `http://localhost:8000/v1`)
3. **Remove or update the API key** — self-hosted NIM does not require the cloud API key

```python
# API Catalog (prototyping)
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

# Self-hosted NIM (production)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-required"  # Or your NIM auth config
)
```

Because both use OpenAI-compatible APIs, your application code does not change — only the endpoint configuration changes.

---

## Summary

Getting started with the NVIDIA API Catalog takes only a few minutes:

1. Create an NVIDIA account at [build.nvidia.com](https://build.nvidia.com)
2. Create an organization for team-based access and billing management
3. Generate an API key (personal or organization-scoped)
4. Set the key as an environment variable
5. Start calling NVIDIA-hosted AI models using the OpenAI-compatible API

The catalog's OpenAI compatibility, free tier, and broad model selection make it the practical starting point for any NVIDIA AI integration project — whether you are building a chatbot, a RAG system, an embeddings pipeline, or a vision application.

---

## References

1. [NVIDIA API Catalog — build.nvidia.com](https://build.nvidia.com)
2. [NVIDIA API Catalog Documentation — docs.api.nvidia.com](https://docs.api.nvidia.com/)
3. [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/)
4. [LangChain NVIDIA AI Endpoints](https://python.langchain.com/docs/integrations/providers/nvidia/)
5. [LlamaIndex NVIDIA LLM Integration](https://docs.llamaindex.ai/en/stable/examples/llm/nvidia/)
6. [OpenAI Python SDK](https://github.com/openai/openai-python)
7. [NVIDIA Developer Program](https://developer.nvidia.com/)

---

[← Back to NVIDIA API Catalog](../) · [← Back to Home](../../)
