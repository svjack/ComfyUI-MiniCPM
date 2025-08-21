# ComfyUI-MiniCPM

🖼️ A ComfyUI custom node for MiniCPM vision-language models, enabling high-quality image captioning and analysis.

---

## Features

- Supports MiniCPM models including **MiniCPM-V-4 (GGUF)**.  
- Multiple caption types to suit different use cases (Describe, Caption, Analyze, etc.).  
- Memory management options to balance VRAM usage and speed.  
- Auto-downloads model files on first use for easy setup.  
- Customizable parameters: max tokens, temperature, top-p/k sampling, repetition penalty.

---

## Installation

Clone the repo into your ComfyUI custom nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/1038lab/comfyui-minicpm.git
````

Install required dependencies:

```bash
pip install -r ComfyUI/custom_nodes/comfyui-minicpm/requirements.txt
```

---

## Supported Models

| Model              | Description                            |
| ------------------ | -------------------------------------- |
| MiniCPM-V-4 (GGUF) | Latest stable GGUF model, best quality |

> The models will be automatically downloaded on first run.
> Manual download and placement into `models/LLM/GGUF` is also supported.

---

## Usage

1. Add the **MiniCPM** node from the `🧪AILab` category in ComfyUI.
2. Connect an image input node to the MiniCPM node.
3. Select the model variant (default is MiniCPM-V-4 GGUF).
4. Choose caption type and adjust parameters as needed.
5. Execute your workflow to generate captions or analysis.

---

## Configuration Defaults

```json
{
  "context_window": 4096,
  "gpu_layers": -1,
  "cpu_threads": 4,
  "default_max_tokens": 1024,
  "default_temperature": 0.8,
  "default_top_p": 0.9,
  "default_top_k": 80,
  "default_repetition_penalty": 1.05,
  "default_system_prompt": "You are MiniCPM, a helpful, concise and knowledgeable vision-language assistant. Answer directly and stay on task."
}
```

---

## Caption Types

* **Describe:** Describe this image in detail.
* **Caption:** Write a concise caption for this image.
* **Analyze:** Analyze the main elements and scene in this image.
* **Identify:** What objects and subjects do you see in this image?
* **Explain:** Explain what's happening in this image.
* **List:** List the main objects visible in this image.
* **Scene:** Describe the scene and setting of this image.
* **Details:** What are the key details in this image?

---

## Tips

* Adjust temperature (0.6–0.8) for balancing creativity and coherence.
* Use top-p (0.9) and top-k (80) sampling for natural output diversity.
* Lower max tokens or precision (bf16/fp16) for faster generation on less powerful GPUs.
* Memory modes help optimize VRAM usage: default, balanced, max savings.

---

## License

GPL-3.0 License

```

If you want me to help with adding more sections or tweaks, just say!
```
