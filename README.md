# Streetwear & Gen Z Fashion Image Tagger

A specialized Python toolkit for analyzing streetwear and Gen Z fashion images, optimized for GPU acceleration and designed for fashion recommendation systems and trend analysis.

## Features

### Core Capabilities
- **Streetwear-Specific Tagging**: Specialized models for streetwear and Gen Z fashion analysis
- **Multi-Image Product Analysis**: Consensus-based analysis across multiple product images
- **Brand Detection**: Recognition of popular streetwear and fashion brands
- **Style Analysis**: Gen Z aesthetics, vibes, fits, and trend classification
- **Batch Processing**: Efficient processing of large image collections with analytics
- **Memory Optimization**: Efficient models for coexistence with other ML pipelines

### AI Models Used
- **BLIP/BLIP-2**: Image captioning and understanding (configurable model size)
- **OpenCLIP (ViT-B-32)**: Fashion-optimized image-text matching with LAION training
- **YOLOv8**: Object detection for clothing items and accessories
- **Custom Classification**: Streetwear-specific category and style classification

## Installation

1. Install dependencies (requires CUDA-capable GPU):
```bash
pip install -r requirements.txt
```

2. First run will download model weights (~15GB total)

## Usage

### Single Image Streetwear Analysis

#### Basic Usage
```bash
python streetwear_tagger.py path/to/image.jpg
```

#### Verbose Output
```bash
python streetwear_tagger.py path/to/image.jpg --verbose
```

#### Save Results
```bash
python streetwear_tagger.py path/to/image.jpg --output results.json
```

### Multi-Image Product Analysis

#### Analyze Product with Multiple Images
```bash
python multi_image_streetwear_tagger.py image1.jpg image2.jpg image3.jpg --product-id "PROD123"
```

#### Memory-Efficient Mode
```bash
python multi_image_streetwear_tagger.py *.jpg --memory-efficient --output results.json
```

### Batch Processing

#### Process Directory
```bash
python streetwear_batch_processor.py /path/to/images --output /path/to/results
```

#### Recursive Processing with Report
```bash
python streetwear_batch_processor.py /path/to/images --recursive --output /path/to/results --report
```

#### Custom Worker Threads
```bash
python streetwear_batch_processor.py /images --workers 4 --output /results
```

## Example Output

### Single Image Analysis
```json
{
  "image_path": "streetwear_hoodie.jpg",
  "processing_time": 1.82,
  "caption": "a person wearing an oversized black hoodie with graphic design",
  "detected_objects": [
    {"class": "person", "confidence": 0.95},
    {"class": "handbag", "confidence": 0.87}
  ],
  "category": "oversized_hoodie",
  "category_confidence": 0.891,
  "gender": "unisex",
  "gender_confidence": 0.823,
  "style_analysis": {
    "aesthetic": {"style": "streetwear", "confidence": 0.934},
    "vibe": {"style": "casual", "confidence": 0.876},
    "fit": {"style": "oversized", "confidence": 0.923},
    "trend": {"style": "layered", "confidence": 0.789}
  },
  "detected_brands": ["supreme", "nike"],
  "colors": ["black", "white", "red"],
  "tags": ["streetwear", "hoodie", "oversized", "black", "graphic", "casual", "urban"],
  "streetwear_recommendations": {
    "primary_category": "oversized_hoodie",
    "aesthetic": "streetwear",
    "vibe": "casual",
    "fit": "oversized",
    "trend": "layered",
    "colors": ["black", "white"],
    "brands": ["supreme", "nike"],
    "tags": ["streetwear", "hoodie", "oversized", "black", "graphic", "casual", "urban"]
  }
}
```

### Multi-Image Product Analysis
```json
{
  "product_id": "PROD123",
  "processing_time": 4.56,
  "total_images": 3,
  "processed_images": 3,
  "best_image": "product_front.jpg",
  "consensus": {
    "consensus_category": "graphic_tee",
    "category_confidence": 0.912,
    "consensus_gender": "unisex",
    "gender_confidence": 0.845,
    "consensus_style": {
      "aesthetic": "y2k",
      "vibe": "indie",
      "fit": "oversized"
    },
    "detected_brands": ["thrasher"],
    "comprehensive_tags": ["graphic", "tee", "oversized", "y2k", "indie", "streetwear"]
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

## Streetwear Categories

### Clothing Types (30+)
- **Tops**: oversized_hoodie, cropped_hoodie, graphic_tee, oversized_tee, tank_top, long_sleeve, sweatshirt, zip_hoodie
- **Bottoms**: baggy_jeans, skinny_jeans, cargo_pants, joggers, shorts, wide_leg_pants, leather_pants  
- **Outerwear**: bomber_jacket, denim_jacket, windbreaker, puffer_jacket, coach_jacket, leather_jacket
- **Footwear**: sneakers, chunky_sneakers, high_tops, skate_shoes, boots, slides
- **Accessories**: bucket_hat, beanie, cap, crossbody_bag, backpack, chain_necklace, sunglasses

### Gen Z Style Analysis
- **Aesthetics**: y2k, cyber, grunge, dark academia, cottagecore, indie, alt, kawaii, harajuku, minimalist, maximalist, vintage, retro
- **Vibes**: main character, that girl, clean girl, baddie, soft girl, e-girl, indie sleaze, normcore, gorpcore, academia
- **Fits**: oversized, baggy, loose, fitted, cropped, high waisted, wide leg, skinny, relaxed, structured
- **Trends**: layered, matching sets, color blocking, monochromatic, oversized blazer, cargo pants, platform shoes, chunky sneakers

### Brand Recognition
- **Streetwear**: Supreme, Off-White, Anti Social Social Club, Golf Wang, Kith, Fear of God, Yeezy, Palace, BAPE, Stussy, Thrasher
- **Athletic**: Nike, Adidas, Vans, Converse, Champion

### Gender Classifications
- **male**: Masculine streetwear and men's fashion
- **female**: Feminine streetwear and women's fashion  
- **unisex**: Gender-neutral streetwear and unisex items

## Requirements

- Python 3.7+
- PyTorch 1.13+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ GPU memory recommended (16GB+ for full models)
- Internet connection for initial model downloads

## Project Structure

- `streetwear_tagger.py` - Single image streetwear analysis
- `multi_image_streetwear_tagger.py` - Multi-image product analysis with consensus
- `streetwear_batch_processor.py` - Batch processing with analytics and reports
- `requirements.txt` - Python dependencies

## Batch Processing Features

- **Parallel Processing**: Multi-threaded image processing
- **Progress Tracking**: Real-time processing status
- **Error Handling**: Robust error handling for corrupted images
- **CSV Export**: Summary results in CSV format
- **Detailed Reports**: Comprehensive streetwear trend analysis
- **Memory Management**: Efficient memory usage for large batches
- **Analytics**: Trend analysis, brand detection, and style insights