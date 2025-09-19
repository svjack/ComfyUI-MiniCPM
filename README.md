```python
import os
import time
import subprocess
from pathlib import Path
import glob
import shutil

# Configuration
SEEDS = [42]
INPUT_DIR = 'ComfyUI/input'
OUTPUT_DIR = 'ComfyUI/output'
PYTHON_PATH = '/environment/miniconda3/envs/system/bin/python'
SOURCE_IMAGE_DIR = 'Eula_Lawrence_Images'  # 本地图像源目录

def copy_and_sort_images():
    """从源目录拷贝JPEG文件到输入目录，并按字典序排序"""
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    # 获取源目录中所有jpeg文件（支持.jpg和.jpeg扩展名）
    image_patterns = ['*.jpg', '*.jpeg']
    image_paths = []
    
    for pattern in image_patterns:
        image_paths.extend(glob.glob(os.path.join(SOURCE_IMAGE_DIR, pattern)))
    
    # 按字典序排序
    image_paths.sort()
    
    # 拷贝文件到输入目录
    copied_paths = []
    for src_path in image_paths:
        filename = os.path.basename(src_path)
        dst_path = os.path.join(INPUT_DIR, filename)
        shutil.copy2(src_path, dst_path)
        copied_paths.append(dst_path)
    
    return copied_paths

def get_latest_output_count():
    """Return the number of TXT files in the output directory"""
    try:
        return len(list(Path(OUTPUT_DIR).glob('*.txt')))
    except:
        return 0

def wait_for_new_output(initial_count):
    """Wait until a new TXT file appears in the output directory"""
    timeout = 300  # 5 minutes timeout
    start_time = time.time()

    while time.time() - start_time < timeout:
        current_count = get_latest_output_count()
        if current_count > initial_count:
            time.sleep(1)  # additional 1 second delay
            return True
        time.sleep(0.5)
    return False

def generate_script(image_path):
    """Generate the script with image captioning workflow"""
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *

with Workflow():
    # 加载图像
    image, _ = LoadImage('{image_path}')
    
    # 使用MiniCPM-V模型生成图像描述
    string = AILabMiniCPMV(image, None, 'MiniCPM-V-4.5', 'Details', '图片中的角色叫优菈，请用中文描述图片，详尽描述人物的外貌、衣着、表情和环境设定及图片风格，并且在你给出的描述中提到人物的名字"优菈"。', 'Auto', 'Clear After Run', 1050658963842269)
    
    # 保存生成的文本描述
    _ = SaveStringKJ(string, 'text', 'output', '.txt')
"""
    return script_content

def main():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 拷贝并排序图像文件
    image_paths = copy_and_sort_images()
    print(f"Copied {len(image_paths)} images from {SOURCE_IMAGE_DIR} to {INPUT_DIR}")
    
    if not image_paths:
        print("No JPEG images found in the source directory.")
        return

    # Main processing loop
    for i, image_path in enumerate(image_paths):
        # Generate script
        image_path = image_path.split("/")[-1]
        script = generate_script(image_path)

        # Write script to file
        script_filename = f'run_image_captioning.py'
        with open(script_filename, 'w') as f:
            f.write(script)

        # Get current output count before running
        initial_count = get_latest_output_count()

        # Run the script
        print(f"Processing image {i+1}/{len(image_paths)}")
        print(f"Image: {os.path.basename(image_path)}")
        result = subprocess.run([PYTHON_PATH, script_filename], capture_output=True, text=True)
        
        # Check if execution was successful
        if result.returncode != 0:
            print(f"Error running script: {result.stderr}")
            continue

        # Wait for new output
        if not wait_for_new_output(initial_count):
            print("Timeout waiting for new output. Continuing to next image.")
            continue
        
        print(f"Successfully generated description for image {i+1}")

if __name__ == "__main__":
    main()

from datasets import load_dataset
import pathlib
import pandas as pd
import numpy as np
ds = load_dataset("svjack/Eula_Lawrence_Images")["train"]

def r_func(x):
    with open(x, "r") as f:
        return f.read().strip()

l0 = pd.Series(
    pathlib.Path("ComfyUI/output").rglob("*.txt")
).map(str).map(lambda x: np.nan if "ipynb" in x else x).dropna().sort_values().map(r_func)

ds = ds.select(range(len(l0))).add_column("prompt", l0)
ds
```

# ComfyUI-MiniCPM

A custom ComfyUI node for MiniCPM vision-language models, supporting v4, v4.5, and v4 GGUF formats, enabling high-quality image captioning and visual analysis.

**🎉 Now supports MiniCPM-V-4.5! The latest model with enhanced capabilities.**

---
## News & Updates
- **2025/08/28**: Update ComfyUI-MIniCPM to **v1.1.1** ( [update.md](update.md#v111-2025-08-28) )
- **2025/08/27**: Update ComfyUI-MIniCPM to **v1.1.0** ( [update.md](update.md#v110-2025-08-27) )
[![MiniCPM v4 VS v45](example_workflows/MiniCPM_v4VSv45.jpg)](example_workflows/MiniCPM_v4VSv45.json)
- Added support for **MiniCPM-V-4.5** models (Transformers)
  
## Features
- MiniCPM-V-4 GGUF
[![MiniCPM-V-4-GGUF](example_workflows/MiniCPM-V-4-GGUF.jpg)](example_workflows/MiniCPM-V-4-GGUF.json)
- MiniCPM-V-4 Batch Images
[![MiniCPM-V-4_batchImages](example_workflows/MiniCPM-V-4_batchImages.jpg)](example_workflows/MiniCPM-V-4_batchImages.json)
- MiniCPM-V-4 video
[![MiniCPM-V-4_video](example_workflows/MiniCPM-V-4_video.jpg)](example_workflows/MiniCPM-V-4_video.json)

- Supports **MiniCPM-V-4.5 (Transformers)** and **MiniCPM-V-4.0 (GGUF)** models
- **Latest MiniCPM-V-4.5** with enhanced capabilities via Transformers
- Multiple caption types to suit different use cases (Describe, Caption, Analyze, etc.)
- Memory management options to balance VRAM usage and speed
- Auto-downloads model files on first use for easy setup
- Customizable parameters: max tokens, temperature, top-p/k sampling, repetition penalty
- Advanced node with full parameter control
- Legacy node for backward compatibility
- Comprehensive GGUF quantization options for V4.0 models

---

## Installation

Clone the repo into your ComfyUI custom nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/1038lab/comfyui-minicpm.git
```

Install required dependencies:

```bash
cd ComfyUI/custom_nodes/comfyui-minicpm
ComfyUI\python_embeded\python pip install -r requirements.txt
ComfyUI\python_embeded\python llama_cpp_install.py
```

> [!note]
> `llama-cpp-python` CUDA Installation for ComfyUI Portable
> - [llama_cpp_install.md](llama_cpp_install/llama_cpp_install.md)
---

## Supported Models

### Transformers Models
| Model                | Description                                    |
| -------------------- | ---------------------------------------------- |
| **MiniCPM-V-4.5**        | 🌟 **Latest V4.5 version with enhanced capabilities** |
| **MiniCPM-V-4.5-int4**   | 🌟 **V4.5 4-bit quantized version, smaller memory footprint** |
| MiniCPM-V-4          | V4.0 full precision version, higher quality   |
| MiniCPM-V-4-int4     | V4.0 4-bit quantized version, smaller memory footprint |

https://huggingface.co/openbmb/MiniCPM-V-4_5  
https://huggingface.co/openbmb/MiniCPM-V-4_5-int4  
https://huggingface.co/openbmb/MiniCPM-V-4
https://huggingface.co/openbmb/MiniCPM-V-4-int4

### GGUF Models

> **Note**: MiniCPM-V-4.5 GGUF models are temporarily unavailable due to llama-cpp-python compatibility issues. Please use MiniCPM-V-4.5 Transformers models or MiniCPM-V-4.0 GGUF models.

#### MiniCPM-V-4.0 (Fully Supported)
| Model                | Size      | Description                           |
| -------------------- | --------- | ------------------------------------- |
| **MiniCPM-V-4 (Q4_K_M)** | ~2.19GB   | **Recommended balance of quality/size** |
| MiniCPM-V-4 (Q4_0)      | ~2.08GB   | Standard 4-bit quantization          |
| MiniCPM-V-4 (Q4_1)      | ~2.29GB   | 4-bit quantization improved          |
| MiniCPM-V-4 (Q4_K_S)    | ~2.09GB   | 4-bit K-quants small                 |
| MiniCPM-V-4 (Q5_0)      | ~2.51GB   | 5-bit quantization                   |
| MiniCPM-V-4 (Q5_1)      | ~2.72GB   | 5-bit quantization improved          |
| MiniCPM-V-4 (Q5_K_M)    | ~2.56GB   | 5-bit K-quants medium                |
| MiniCPM-V-4 (Q5_K_S)    | ~2.51GB   | 5-bit K-quants small                 |
| MiniCPM-V-4 (Q6_K)      | ~2.96GB   | Very high quality                    |
| MiniCPM-V-4 (Q8_0)      | ~3.83GB   | Highest quality quantized            |

https://huggingface.co/openbmb/MiniCPM-V-4-gguf

> The models will be automatically downloaded on first run.
> Manual download and placement into `models/LLM` (transformers) or `models/LLM/GGUF` (GGUF) is also supported.

---

## Available Nodes

### 1. MiniCPM-4-V-Transformers
- Basic transformers-based node with essential parameters
- Supports image and video input
- Memory management options
- Preset prompt types

### 2. MiniCPM-4-V-Transformers Advanced
- Full-featured transformers-based node
- All parameters customizable
- System prompt support
- Advanced video processing options

### 3. MiniCPM-4-V-GGUF
- GGUF-based node with essential parameters
- Optimized for performance

### 4. MiniCPM-4-V-GGUF Advanced
- Full-featured GGUF-based node
- All parameters customizable

### 5. MiniCPM (Legacy)
- Original node for backward compatibility
- Basic functionality

---

## Usage

1. Add the **MiniCPM** node from the `🧪AILab` category in ComfyUI.
2. Connect an image or video input node to the MiniCPM node.
3. Select the model variant (default is MiniCPM-V-4-int4 for transformers).
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
  "default_temperature": 0.7,
  "default_top_p": 0.9,
  "default_top_k": 100,
  "default_repetition_penalty": 1.10,
  "default_system_prompt": "You are MiniCPM-V, a helpful, concise and knowledgeable vision-language assistant. Answer directly and stay on task."
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
* **Summarize:** Summarize the key content of this image in 1-2 sentences.
* **Emotion:** Describe the emotions or mood conveyed by this image.
* **Style:** Describe the artistic or visual style of this image.
* **Location:** Where might this image be taken? Analyze the setting or location.
* **Question:** What question could be asked based on this image?
* **Creative:** Describe this image as if writing the beginning of a short story.

---

## Memory Management Options

* **Keep in Memory:** Model stays loaded for faster subsequent runs
* **Clear After Run:** Model is unloaded after each run to save memory
* **Global Cache:** Model is cached globally and shared between nodes

---

## Tips

### VRAM Requirements
* **4-6GB VRAM**: Use MiniCPM-V-4-int4 or GGUF Q4 models
* **8GB VRAM**: Use MiniCPM-V-4.5-int4 (recommended)
* **12GB+ VRAM**: Can use full MiniCPM-V-4.5
* **CUDA OOM Error**: Try int4 quantized models or CPU mode

### General Tips
* 🌟 **Try MiniCPM-V-4.5 Transformers first** - enhanced capabilities over V4.0
* For **best balance**: use MiniCPM-V-4 (Q4_K_M) GGUF model
* For **highest quality**: use MiniCPM-V-4.5 Transformers
* For **low VRAM**: use MiniCPM-V-4.5-int4 or MiniCPM-V-4 (Q4_0) GGUF
* Adjust temperature (0.6–0.8) for balancing creativity and coherence.
* Use top-p (0.9) and top-k (80) sampling for natural output diversity.
* Lower max tokens or precision (bf16/fp16) for faster generation on less powerful GPUs.
* Memory modes help optimize VRAM usage: default, balanced, max savings.
* Transformers models offer better quality but use more memory.
* GGUF models are more memory-efficient but may have slightly lower quality.

---

## License


GPL-3.0 License

