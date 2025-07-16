# AIoT-Enhanced Seasonal Landscape Transformation Camera System 

## Overview
This project provides an all-in-one pipeline for feature extraction, plug-and-play (PnP) image generation, and StyleID-based style transfer. The main script, all_in_one_scripts.py, automates the entire workflow from input image to stylized output.

## Pipeline Overview

1. **Feature Extraction**  
   Extracts features from the input image using a pre-trained model.

2. **Plug-and-Play (PnP) Image Generation**  
   Generates translated images using the PnP method.

3. **StyleID-based Style Transfer**  
   Applies style transfer to the content images using StyleID.

## Usage

### 1. Prepare Input and Style Images

- Place your input image in the `input_images/` directory (default: `./input_images/example.jpg`).
- Place your style images in the `style_images/` directory.

### 2. Configure Paths

You can modify the following variables in `all_in_one_scripts.py` as needed:

```python
input_image = "./input_images/example.jpg"
experiment_name = "test_experiment"
configs_directory = "./configs"
models_directory = "./models"
style_images_dir = "./style_images"
```

### 3. Run the Pipeline

```bash
python all_in_one_scripts.py
```

The script will:

- Extract features from the input image.
- Generate PnP images.
- Copy PnP outputs to the content directory.
- Run StyleID style transfer.

### 4. Check Results

Final stylized images will be saved in:

```
./outputs/<experiment_name>/styled_results/
```

## Requirements

- Python 3.8+
- All dependencies listed in `environment.yaml` 
- Pre-trained model [checkpoints](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main) in the `models/` directory

## API Usage

To enable integration with web frontends (such as `index.html`) or AIoT devices, the pipeline is wrapped as a RESTful API using Flask. The API supports file upload, custom parameters, and returns output paths for further processing or display.

### API Endpoint

Start the API server:

```bash
python request_api.py
```

**Endpoint:**  
`POST /run_pipeline`

**Form Data:**
- `input_image` (file, required): The input image file.
- `style_images` (file, multiple, required): One or more style image files.
- `experiment_name` (string, optional): Name for the experiment (default: `test_experiment`).

**Response:**
```json
{
  "status": "success",
  "output_dir": "outputs/test_experiment/styled_results",
  "exp_output_root": "outputs/test_experiment",
  "content_images_dir": "outputs/test_experiment/pnp_outputs",
  "precomputed_features_dir": "outputs/test_experiment/precomputed_feats"
}
```

### Example: Frontend Integration

You can call the API from your frontend (e.g., in `index.html`) using JavaScript:

```javascript
const formData = new FormData();
formData.append('input_image', inputFile.files[0]);
for (const file of styleFiles) {
  formData.append('style_images', file);
}
formData.append('experiment_name', 'my_experiment');

fetch('http://localhost:5000/run_pipeline', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => {
  if (data.status === 'success') {
    // Display or download the result image(s) from data.output_dir
    alert('Success! Output dir: ' + data.output_dir);
  } else {
    alert('Error: ' + data.msg);
  }
});
```
### Example: AIoT Device Integration

AIoT devices can use HTTP POST requests to interact with the API, enabling automated or remote image processing. For example, using Python's `requests` library:

```python
import requests

files = {
    'input_image': open('path/to/input.jpg', 'rb'),
    'style_images': (open('path/to/style1.jpg', 'rb'), open('path/to/style2.jpg', 'rb'))
}
data = {'experiment_name': 'device_experiment'}

response = requests.post('http://<server-ip>:5000/run_pipeline', files=files, data=data)
print(response.json())
```

This allows edge devices or smart cameras to upload images for seasonal transformation and receive processed results, which can then be displayed locally, sent to the cloud, or used for further IoT applications.

---

#### Output

- **Feature extraction results:** `./outputs/<experiment_name>/precomputed_feats/`
- **PnP outputs:** `./outputs/<experiment_name>/pnp_outputs/`
- **Final styled images:** `./outputs/<experiment_name>/styled_results/`

## Customization

- **Input image:** Upload via API or change the path in the script.
- **Experiment name:** Set via API parameter or script variable.
- **Style images:** Upload via API or add to the `./style_images/` directory.
- **Model/config paths:** Adjust the paths in the script if your directory structure is different.

---

## Notes

- Ensure all required models and configuration files are present in the specified directories.
- Style images must be prepared in advance in the `style_images/` directory.
- For custom experiments, adjust the paths and parameters in `all_in_one_scripts.py`.

## Troubleshooting

- If no images are copied in Step 2.5, check if PnP generation completed successfully and outputs are in the correct folder.
- For missing dependencies or model files, refer to the error messages and ensure all paths are correct.
