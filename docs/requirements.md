# System Requirements

coreason-search (Service L) is designed to run as a high-performance microservice.

## Software Dependencies

The following Python packages are required:

*   **FastAPI** (`>=0.115.0`): Web framework for the API.
*   **Uvicorn** (`>=0.27.0`): ASGI server.
*   **AnyIO** (`>=4.12.1`): Asynchronous I/O support.
*   **HTTPX** (`>=0.28.1`): Async HTTP client.
*   **LanceDB** (`>=0.27.1`): Vector database.
*   **Pydantic** (`>=2.12.5`): Data validation.
*   **Coreason Identity** (`>=0.4.2`): Authentication/Authorization.

## Hardware Requirements

*   **RAM:** Minimum 16GB recommended (for loading embedding models and DB indices).
*   **Storage:** Fast NVMe SSD recommended for LanceDB persistence (`/tmp/lancedb` or configured volume).
*   **GPU:** NVIDIA GPU (CUDA) recommended for `HuggingFaceEmbedder` (if not using `MockEmbedder`).

## Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `APP__DATABASE_URI` | Path to LanceDB storage | `/tmp/lancedb` |
| `APP__EMBEDDING__PROVIDER` | Embedder provider (`auto`, `hf`, `mock`) | `auto` |
| `APP__EMBEDDING__MODEL_NAME` | HF Model ID | `Alibaba-NLP/gte-Qwen2-7B-instruct` |
| `APP__ENV` | Environment (`development`, `production`) | `development` |
