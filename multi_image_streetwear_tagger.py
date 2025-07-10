#!/usr/bin/env python3
"""
Multi-Image Streetwear Product Tagger
Optimized for products with multiple images and memory-efficient operation
alongside Stable Diffusion XL pipeline.
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import gc
from collections import Counter, defaultdict
import statistics

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoProcessor, AutoModel,
)
from ultralytics import YOLO
import open_clip
import cv2

warnings.filterwarnings("ignore")


class MultiImageStreetwearTagger:
    def __init__(self, use_gpu_optimization: bool = True, memory_efficient: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gpu_optimization = use_gpu_optimization
        self.memory_efficient = memory_efficient
        
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.load_models()
        self.init_streetwear_knowledge()
        
    def load_models(self):
        """Load models with memory optimization for SDXL coexistence"""
        print("Loading memory-optimized models...")
        
        # Use smaller, more efficient models when memory_efficient is True
        if self.memory_efficient:
            print("Loading BLIP base model (memory efficient)...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base", 
                torch_dtype=torch.float16
            )
        else:
            print("Loading BLIP-2 model...")
            self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.blip_model = AutoModel.from_pretrained(
                "Salesforce/blip2-opt-2.7b", 
                torch_dtype=torch.float16
            )
        
        self.blip_model.to(self.device)
        
        # OpenCLIP for classification
        print("Loading OpenCLIP model...")
        self.openclip_model, _, self.openclip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        self.openclip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.openclip_model.to(self.device)
        
        # Lighter YOLO model for memory efficiency
        print("Loading YOLOv8 model...")
        yolo_model = 'yolov8n.pt' if self.memory_efficient else 'yolov8s.pt'
        self.yolo_model = YOLO(yolo_model)
        
        # GPU optimizations
        if self.use_gpu_optimization and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        print("Models loaded successfully!")
    
    def init_streetwear_knowledge(self):
        """Initialize streetwear knowledge base"""
        self.streetwear_categories = {
            # Tops
            "oversized_hoodie": ["oversized hoodie", "baggy hoodie", "drop shoulder hoodie"],
            "cropped_hoodie": ["cropped hoodie", "crop hoodie", "short hoodie"],
            "graphic_tee": ["graphic t-shirt", "band tee", "logo tee", "printed tee"],
            "oversized_tee": ["oversized t-shirt", "baggy tee", "drop shoulder tee"],
            "sweatshirt": ["crewneck sweatshirt", "pullover sweatshirt"],
            "zip_hoodie": ["zip up hoodie", "zip hoodie", "full zip hoodie"],
            
            # Bottoms
            "baggy_jeans": ["baggy jeans", "wide leg jeans", "relaxed fit jeans"],
            "skinny_jeans": ["skinny jeans", "tight jeans", "slim fit jeans"],
            "cargo_pants": ["cargo pants", "utility pants", "tactical pants"],
            "joggers": ["joggers", "sweatpants", "track pants"],
            "shorts": ["basketball shorts", "athletic shorts", "board shorts"],
            "wide_leg_pants": ["wide leg pants", "palazzo pants", "flared pants"],
            
            # Outerwear
            "bomber_jacket": ["bomber jacket", "flight jacket", "varsity jacket"],
            "denim_jacket": ["denim jacket", "jean jacket", "trucker jacket"],
            "windbreaker": ["windbreaker", "track jacket", "shell jacket"],
            "puffer_jacket": ["puffer jacket", "down jacket", "quilted jacket"],
            "coach_jacket": ["coach jacket", "snap jacket", "cotton jacket"],
            
            # Footwear
            "sneakers": ["sneakers", "athletic shoes", "running shoes"],
            "chunky_sneakers": ["chunky sneakers", "dad shoes", "bulky sneakers"],
            "high_tops": ["high top sneakers", "basketball shoes", "ankle sneakers"],
            "skate_shoes": ["skate shoes", "skateboard shoes", "vans style"],
            "boots": ["combat boots", "doc martens", "ankle boots"],
            
            # Accessories
            "bucket_hat": ["bucket hat", "fisherman hat", "sun hat"],
            "beanie": ["beanie", "knit hat", "winter hat"],
            "cap": ["baseball cap", "snapback", "fitted cap"],
            "crossbody_bag": ["crossbody bag", "sling bag", "chest bag"],
            "backpack": ["backpack", "bookbag", "daypack"],
            "chain_necklace": ["chain necklace", "gold chain", "silver chain"]
        }
        
        self.genz_styles = {
            "aesthetics": [
                "y2k", "cyber", "grunge", "dark academia", "cottagecore", "indie", "alt",
                "kawaii", "harajuku", "minimalist", "maximalist", "vintage", "retro"
            ],
            "vibes": [
                "main character", "that girl", "clean girl", "baddie", "soft girl",
                "e-girl", "indie sleaze", "normcore", "gorpcore", "academia"
            ],
            "fits": [
                "oversized", "baggy", "loose", "fitted", "cropped", "high waisted",
                "wide leg", "skinny", "relaxed", "structured"
            ],
            "trends": [
                "layered", "matching sets", "color blocking", "monochromatic",
                "oversized blazer", "cargo pants", "platform shoes", "chunky sneakers"
            ]
        }
        
        self.brand_indicators = {
            "supreme": ["supreme", "box logo", "bogo"],
            "off_white": ["off white", "off-white", "quotation marks"],
            "anti_social_social_club": ["anti social social club", "assc"],
            "golf_wang": ["golf wang", "golf"],
            "kith": ["kith", "kith nyc"],
            "fear_of_god": ["fear of god", "fog", "essentials"],
            "yeezy": ["yeezy", "adidas yeezy"],
            "palace": ["palace", "palace skateboards"],
            "bape": ["bape", "a bathing ape"],
            "stussy": ["stussy", "stüssy"],
            "thrasher": ["thrasher", "thrasher magazine"],
            "nike": ["nike", "swoosh"],
            "adidas": ["adidas", "three stripes"],
            "vans": ["vans", "off the wall"],
            "converse": ["converse", "chuck taylor"]
        }
    
    def assess_image_quality(self, image: Image.Image) -> float:
        """Assess image quality for best image selection"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate sharpness using Laplacian variance
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Calculate contrast
        contrast = np.std(gray)
        
        # Composite quality score
        quality_score = (laplacian_var / 1000) + (brightness / 255) + (contrast / 128)
        
        return min(quality_score, 1.0)  # Cap at 1.0
    
    def memory_cleanup(self):
        """Cleanup GPU memory between operations"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def preprocess_image(self, image_path: str) -> Tuple[Image.Image, np.ndarray, float]:
        """Preprocess image with quality assessment"""
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Resize if too large (memory optimization)
            if max(image.size) > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            quality_score = self.assess_image_quality(image)
            
            return image, cv_image, quality_score
            
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")
    
    def analyze_single_image(self, image_path: str) -> Dict:
        """Analyze a single image with memory cleanup"""
        image, cv_image, quality_score = self.preprocess_image(image_path)
        
        # Generate caption
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if self.memory_efficient:
                out = self.blip_model.generate(**inputs, max_length=50)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            else:
                generated_ids = self.blip_model.generate(**inputs, max_length=50)
                caption = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Object detection
        results = self.yolo_model(image, verbose=False)
        detected_objects = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    detected_objects.append({
                        'class': result.names[int(box.cls)],
                        'confidence': float(box.conf)
                    })
        
        # Classifications
        category, category_conf = self.classify_streetwear_category(image)
        gender, gender_conf = self.classify_streetwear_gender(image)
        style_analysis = self.analyze_genz_style(image)
        
        # Brand detection
        detected_brands = self.detect_brands(caption)
        
        # Memory cleanup
        self.memory_cleanup()
        
        return {
            "image_path": image_path,
            "quality_score": round(quality_score, 3),
            "caption": caption,
            "detected_objects": detected_objects,
            "category": category,
            "category_confidence": round(category_conf, 3),
            "gender": gender,
            "gender_confidence": round(gender_conf, 3),
            "style_analysis": style_analysis,
            "detected_brands": detected_brands
        }
    
    def classify_streetwear_category(self, image: Image.Image) -> Tuple[str, float]:
        """Classify streetwear category"""
        all_categories = []
        category_map = {}
        
        for main_cat, variations in self.streetwear_categories.items():
            all_categories.extend(variations)
            for variation in variations:
                category_map[variation] = main_cat
        
        prediction, confidence = self.classify_with_openclip(image, all_categories)
        main_category = category_map.get(prediction, prediction)
        
        return main_category, confidence
    
    def classify_streetwear_gender(self, image: Image.Image) -> Tuple[str, float]:
        """Classify gender for streetwear"""
        gender_candidates = [
            "mens streetwear", "womens streetwear", "unisex streetwear",
            "masculine street fashion", "feminine street fashion", "gender neutral fashion"
        ]
        
        prediction, confidence = self.classify_with_openclip(image, gender_candidates)
        
        if any(term in prediction for term in ["mens", "masculine"]):
            return "male", confidence
        elif any(term in prediction for term in ["womens", "feminine"]):
            return "female", confidence
        else:
            return "unisex", confidence
    
    def analyze_genz_style(self, image: Image.Image) -> Dict:
        """Analyze Gen Z style attributes"""
        style_analysis = {}
        
        # Aesthetic
        aesthetic, aesthetic_conf = self.classify_with_openclip(image, self.genz_styles["aesthetics"])
        style_analysis["aesthetic"] = {"style": aesthetic, "confidence": round(aesthetic_conf, 3)}
        
        # Vibe
        vibe, vibe_conf = self.classify_with_openclip(image, self.genz_styles["vibes"])
        style_analysis["vibe"] = {"style": vibe, "confidence": round(vibe_conf, 3)}
        
        # Fit
        fit, fit_conf = self.classify_with_openclip(image, self.genz_styles["fits"])
        style_analysis["fit"] = {"style": fit, "confidence": round(fit_conf, 3)}
        
        return style_analysis
    
    def classify_with_openclip(self, image: Image.Image, candidates: List[str]) -> Tuple[str, float]:
        """Classification using OpenCLIP"""
        image_tensor = self.openclip_preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = self.openclip_tokenizer(candidates).to(self.device)
        
        with torch.no_grad():
            image_features = self.openclip_model.encode_image(image_tensor)
            text_features = self.openclip_model.encode_text(text_tokens)
            
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            similarities = (image_features @ text_features.T).softmax(dim=-1)
        
        best_idx = similarities.argmax().item()
        confidence = similarities[0, best_idx].item()
        
        return candidates[best_idx], confidence
    
    def detect_brands(self, caption: str) -> List[str]:
        """Detect brands from caption"""
        detected_brands = []
        caption_lower = caption.lower()
        
        for brand, indicators in self.brand_indicators.items():
            for indicator in indicators:
                if indicator in caption_lower:
                    detected_brands.append(brand)
                    break
        
        return list(set(detected_brands))
    
    def consensus_algorithm(self, results: List[Dict]) -> Dict:
        """Generate consensus from multiple image results"""
        if not results:
            return {"error": "No results to process"}
        
        # Weight by quality score
        total_weight = sum(r["quality_score"] for r in results)
        
        # Category consensus (weighted voting)
        category_votes = defaultdict(float)
        for result in results:
            weight = result["quality_score"] / total_weight
            category_votes[result["category"]] += weight * result["category_confidence"]
        
        best_category = max(category_votes.items(), key=lambda x: x[1])
        
        # Gender consensus
        gender_votes = defaultdict(float)
        for result in results:
            weight = result["quality_score"] / total_weight
            gender_votes[result["gender"]] += weight * result["gender_confidence"]
        
        best_gender = max(gender_votes.items(), key=lambda x: x[1])
        
        # Style consensus
        aesthetic_votes = defaultdict(float)
        vibe_votes = defaultdict(float)
        fit_votes = defaultdict(float)
        
        for result in results:
            weight = result["quality_score"] / total_weight
            style = result["style_analysis"]
            
            aesthetic_votes[style["aesthetic"]["style"]] += weight * style["aesthetic"]["confidence"]
            vibe_votes[style["vibe"]["style"]] += weight * style["vibe"]["confidence"]
            fit_votes[style["fit"]["style"]] += weight * style["fit"]["confidence"]
        
        # Aggregate brands
        all_brands = set()
        for result in results:
            all_brands.update(result["detected_brands"])
        
        # Combine captions from best quality images
        best_results = sorted(results, key=lambda x: x["quality_score"], reverse=True)[:3]
        combined_caption = " | ".join([r["caption"] for r in best_results])
        
        # Generate comprehensive tags
        tags = set()
        for result in results:
            tags.update([
                result["category"],
                result["gender"],
                result["style_analysis"]["aesthetic"]["style"],
                result["style_analysis"]["vibe"]["style"],
                result["style_analysis"]["fit"]["style"]
            ])
            tags.update(result["detected_brands"])
        
        return {
            "consensus_category": best_category[0],
            "category_confidence": round(best_category[1], 3),
            "consensus_gender": best_gender[0],
            "gender_confidence": round(best_gender[1], 3),
            "consensus_style": {
                "aesthetic": max(aesthetic_votes.items(), key=lambda x: x[1])[0],
                "vibe": max(vibe_votes.items(), key=lambda x: x[1])[0],
                "fit": max(fit_votes.items(), key=lambda x: x[1])[0]
            },
            "detected_brands": list(all_brands),
            "combined_caption": combined_caption,
            "comprehensive_tags": list(tags)[:15],
            "image_count": len(results),
            "avg_quality_score": round(statistics.mean([r["quality_score"] for r in results]), 3)
        }
    
    def analyze_product(self, image_paths: List[str], product_id: str = None) -> Dict:
        """Analyze a product with multiple images"""
        start_time = time.time()
        product_id = product_id or f"product_{int(time.time())}"
        
        print(f"Analyzing product {product_id} with {len(image_paths)} images...")
        
        # Analyze individual images
        individual_results = []
        for i, image_path in enumerate(image_paths):
            try:
                print(f"Processing image {i+1}/{len(image_paths)}: {Path(image_path).name}")
                result = self.analyze_single_image(image_path)
                individual_results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        if not individual_results:
            return {"error": "No images could be processed", "product_id": product_id}
        
        # Generate consensus
        consensus = self.consensus_algorithm(individual_results)
        
        # Find best quality image
        best_image = max(individual_results, key=lambda x: x["quality_score"])
        
        processing_time = time.time() - start_time
        
        return {
            "product_id": product_id,
            "processing_time": round(processing_time, 2),
            "total_images": len(image_paths),
            "processed_images": len(individual_results),
            "best_image": best_image["image_path"],
            "consensus": consensus,
            "individual_results": individual_results,
            "recommendations": {
                "primary_category": consensus["consensus_category"],
                "gender": consensus["consensus_gender"],
                "aesthetic": consensus["consensus_style"]["aesthetic"],
                "vibe": consensus["consensus_style"]["vibe"],
                "fit": consensus["consensus_style"]["fit"],
                "brands": consensus["detected_brands"],
                "tags": consensus["comprehensive_tags"]
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Multi-Image Streetwear Product Tagger")
    parser.add_argument("images", nargs="+", help="Paths to product images")
    parser.add_argument("--product-id", help="Product ID")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--memory-efficient", action="store_true", help="Use memory-efficient models")
    
    args = parser.parse_args()
    
    # Validate images
    valid_images = [img for img in args.images if os.path.exists(img)]
    if not valid_images:
        print("Error: No valid images found")
        return
    
    # Initialize tagger
    tagger = MultiImageStreetwearTagger(memory_efficient=args.memory_efficient)
    
    # Analyze product
    try:
        result = tagger.analyze_product(valid_images, args.product_id)
        
        if args.verbose:
            print("\n" + "="*60)
            print("MULTI-IMAGE STREETWEAR PRODUCT ANALYSIS")
            print("="*60)
            print(f"Product ID: {result['product_id']}")
            print(f"Images: {result['processed_images']}/{result['total_images']}")
            print(f"Best Image: {Path(result['best_image']).name}")
            print(f"Processing Time: {result['processing_time']}s")
            
            consensus = result['consensus']
            print(f"\nConsensus Results:")
            print(f"Category: {consensus['consensus_category']} ({consensus['category_confidence']})")
            print(f"Gender: {consensus['consensus_gender']} ({consensus['gender_confidence']})")
            print(f"Aesthetic: {consensus['consensus_style']['aesthetic']}")
            print(f"Vibe: {consensus['consensus_style']['vibe']}")
            print(f"Fit: {consensus['consensus_style']['fit']}")
            print(f"Brands: {', '.join(consensus['detected_brands']) if consensus['detected_brands'] else 'None'}")
            print(f"Tags: {', '.join(consensus['comprehensive_tags'])}")
        else:
            print(json.dumps(result, indent=2))
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()