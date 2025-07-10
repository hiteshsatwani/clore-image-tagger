# Advanced Image Tagger for Recommendation Engine

A high-performance Python script optimized for g5.xlarge GPU instances that analyzes images and generates comprehensive tags, categories, and gender classifications for recommendation systems.

## Features

### Core Capabilities
- **Advanced Tag Generation**: Multi-modal tag extraction using state-of-the-art models
- **Object Detection**: YOLOv8x-based object detection for comprehensive item identification
- **Category Classification**: 50+ clothing/item categories with high accuracy
- **Gender Classification**: Nuanced gender classification (male/female/unisex)
- **Style Analysis**: Detailed style attribute extraction (aesthetic, occasion, fit, design)
- **Batch Processing**: Optimized for processing multiple images efficiently

### AI Models Used
- **BLIP-2 (2.7B)**: Advanced image captioning and understanding
- **OpenCLIP (ViT-B-32)**: Superior image-text matching with LAION training
- **YOLOv8x**: State-of-the-art object detection
- **Vision Transformer (Large)**: Advanced visual feature extraction
- **Sentence Transformers**: Semantic similarity analysis

## Installation

1. Install dependencies (requires CUDA-capable GPU):
```bash
pip install -r requirements.txt
```

2. First run will download model weights (~15GB total)

## Usage

### Single Image Processing

#### Basic Usage
```bash
python advanced_image_tagger.py path/to/image.jpg
```

#### Verbose Output
```bash
python advanced_image_tagger.py path/to/image.jpg --verbose
```

#### Save Results
```bash
python advanced_image_tagger.py path/to/image.jpg --output results.json
```

### Batch Processing

#### Process Directory
```bash
python batch_processor.py /path/to/images --output /path/to/results
```

#### Recursive Processing
```bash
python batch_processor.py /path/to/images --recursive --output /path/to/results
```

#### Generate Report
```bash
python batch_processor.py /path/to/images --output /path/to/results --report
```

### Advanced Options
```bash
# Disable GPU optimizations
python advanced_image_tagger.py image.jpg --no-gpu-opt

# Custom worker threads for batch processing
python batch_processor.py /images --workers 4 --output /results
```

## Example Output

```json
{
  "image_path": "sample_dress.jpg",
  "processing_time": 2.34,
  "caption": "a woman wearing an elegant blue evening dress with floral patterns",
  "detected_objects": [
    {"class": "person", "confidence": 0.98},
    {"class": "dress", "confidence": 0.94}
  ],
  "category": "evening dress",
  "category_confidence": 0.923,
  "gender": "female",
  "gender_confidence": 0.887,
  "style_attributes": [
    ["elegant", 0.934],
    ["formal", 0.876],
    ["evening", 0.823],
    ["floral", 0.789],
    ["fitted", 0.734]
  ],
  "all_tags": ["elegant", "formal", "evening", "blue", "floral", "fitted", "dress"],
  "recommendations": {
    "primary_tags": ["elegant", "formal", "evening", "blue", "floral", "fitted", "dress"],
    "style_tags": ["elegant", "formal", "evening", "floral", "fitted"],
    "category_tags": ["evening dress"],
    "object_tags": ["dress"]
  }
}
```

## Performance Optimizations

### GPU Utilization
- **Mixed Precision**: FP16 inference for faster processing
- **CUDA Optimizations**: cuDNN benchmarking and TensorFloat-32
- **Memory Management**: Efficient GPU memory usage
- **Batch Processing**: Optimized for multiple images

### Recommended Hardware
- **GPU**: NVIDIA g5.xlarge (A10G) or better
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ for model weights

## Model Categories

### Clothing Types (50+)
- **Tops**: t-shirt, shirt, blouse, tank top, crop top, sweater, cardigan, hoodie, jacket, blazer, coat
- **Bottoms**: jeans, pants, trousers, leggings, shorts, skirt, mini skirt, midi skirt, maxi skirt
- **Dresses**: dress, midi dress, maxi dress, cocktail dress, evening gown
- **Suits**: suit, jumpsuit, romper, overalls
- **Footwear**: sneakers, boots, sandals, heels, flats, loafers, athletic shoes
- **Accessories**: bag, purse, backpack, hat, cap, sunglasses, jewelry, belt, scarf
- **Intimates**: underwear, bra, lingerie, swimsuit, bikini
- **Activewear**: athletic wear, yoga pants, sports bra, workout clothes

### Style Attributes
- **Aesthetic**: minimalist, bohemian, vintage, modern, classic, trendy, elegant, chic
- **Occasion**: casual, formal, business, cocktail, evening, party, athletic
- **Fit**: fitted, slim fit, loose, oversized, flowy, structured, tailored
- **Design**: patterned, striped, floral, solid color, textured, embellished

## Gender Classifications

- **male**: Masculine clothing styles and men's fashion
- **female**: Feminine clothing styles and women's fashion
- **unisex**: Gender-neutral, androgynous, or unisex items

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ GPU memory recommended
- Internet connection for initial model downloads

## Batch Processing Features

- **Parallel Processing**: Multi-threaded image processing
- **Progress Tracking**: Real-time processing status
- **Error Handling**: Robust error handling for corrupted images
- **CSV Export**: Summary results in CSV format
- **Detailed Reports**: Comprehensive processing reports
- **Memory Management**: Efficient memory usage for large batches