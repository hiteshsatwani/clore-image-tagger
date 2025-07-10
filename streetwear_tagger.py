#!/usr/bin/env python3
"""
Streetwear & Gen Z Fashion Image Tagger
Specialized for streetwear, hypebeast, and Gen Z fashion trends.
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import re

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel,
    AutoProcessor, AutoModel,
    pipeline
)
from ultralytics import YOLO
import timm
import open_clip
from sentence_transformers import SentenceTransformer
import cv2

warnings.filterwarnings("ignore")


class StreetwearTagger:
    def __init__(self, use_gpu_optimization: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gpu_optimization = use_gpu_optimization
        
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.load_models()
        self.init_streetwear_knowledge()
        
    def load_models(self):
        """Load models optimized for streetwear detection"""
        print("Loading streetwear-optimized models...")
        
        # BLIP-2 for image understanding
        print("Loading BLIP-2 model...")
        self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.blip_model = AutoModel.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        self.blip_model.to(self.device)
        
        # OpenCLIP for fashion classification
        print("Loading OpenCLIP model...")
        self.openclip_model, _, self.openclip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        self.openclip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.openclip_model.to(self.device)
        
        # YOLOv8 for object detection
        print("Loading YOLOv8 model...")
        self.yolo_model = YOLO('yolov8x.pt')
        
        # Enable GPU optimizations
        if self.use_gpu_optimization and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        print("Models loaded successfully!")
    
    def init_streetwear_knowledge(self):
        """Initialize streetwear-specific knowledge base"""
        
        # Streetwear Categories
        self.streetwear_categories = {
            # Tops
            "oversized_hoodie": ["oversized hoodie", "baggy hoodie", "drop shoulder hoodie"],
            "cropped_hoodie": ["cropped hoodie", "crop hoodie", "short hoodie"],
            "graphic_tee": ["graphic t-shirt", "band tee", "logo tee", "printed tee"],
            "oversized_tee": ["oversized t-shirt", "baggy tee", "drop shoulder tee"],
            "tank_top": ["tank top", "muscle tank", "sleeveless shirt"],
            "long_sleeve": ["long sleeve shirt", "thermal shirt", "layering piece"],
            "sweatshirt": ["crewneck sweatshirt", "pullover sweatshirt"],
            "zip_hoodie": ["zip up hoodie", "zip hoodie", "full zip hoodie"],
            
            # Bottoms
            "baggy_jeans": ["baggy jeans", "wide leg jeans", "relaxed fit jeans"],
            "skinny_jeans": ["skinny jeans", "tight jeans", "slim fit jeans"],
            "cargo_pants": ["cargo pants", "utility pants", "tactical pants"],
            "joggers": ["joggers", "sweatpants", "track pants"],
            "shorts": ["basketball shorts", "athletic shorts", "board shorts"],
            "wide_leg_pants": ["wide leg pants", "palazzo pants", "flared pants"],
            "leather_pants": ["leather pants", "faux leather pants", "pleather pants"],
            
            # Outerwear
            "bomber_jacket": ["bomber jacket", "flight jacket", "varsity jacket"],
            "denim_jacket": ["denim jacket", "jean jacket", "trucker jacket"],
            "windbreaker": ["windbreaker", "track jacket", "shell jacket"],
            "puffer_jacket": ["puffer jacket", "down jacket", "quilted jacket"],
            "coach_jacket": ["coach jacket", "snap jacket", "cotton jacket"],
            "leather_jacket": ["leather jacket", "moto jacket", "biker jacket"],
            
            # Footwear
            "sneakers": ["sneakers", "athletic shoes", "running shoes"],
            "chunky_sneakers": ["chunky sneakers", "dad shoes", "bulky sneakers"],
            "high_tops": ["high top sneakers", "basketball shoes", "ankle sneakers"],
            "skate_shoes": ["skate shoes", "skateboard shoes", "vans style"],
            "boots": ["combat boots", "doc martens", "ankle boots"],
            "slides": ["slides", "slip on sandals", "pool slides"],
            
            # Accessories
            "bucket_hat": ["bucket hat", "fisherman hat", "sun hat"],
            "beanie": ["beanie", "knit hat", "winter hat"],
            "cap": ["baseball cap", "snapback", "fitted cap"],
            "crossbody_bag": ["crossbody bag", "sling bag", "chest bag"],
            "backpack": ["backpack", "bookbag", "daypack"],
            "chain_necklace": ["chain necklace", "gold chain", "silver chain"],
            "sunglasses": ["sunglasses", "shades", "eyewear"]
        }
        
        # Gen Z Style Attributes
        self.genz_styles = {
            "aesthetics": [
                "y2k", "cyber", "futuristic", "retro-futuristic", "tech wear",
                "grunge", "soft grunge", "dark academia", "light academia",
                "cottagecore", "goblincore", "fairycore", "vintage", "thrifted",
                "indie", "alt", "alternative", "emo", "scene", "goth",
                "kawaii", "harajuku", "japanese street fashion", "korean fashion",
                "minimalist", "maximalist", "colorful", "monochrome", "neon",
                "pastel", "earth tones", "neutral", "bold", "statement"
            ],
            "vibes": [
                "main character", "that girl", "clean girl", "coquette", "baddie",
                "soft girl", "e-girl", "vsco girl", "indie sleaze", "blokecore",
                "normcore", "gorpcore", "cottagecore", "dark academia",
                "light academia", "academia", "coastal grandmother", "mob wife"
            ],
            "occasions": [
                "casual", "everyday", "going out", "party", "festival", "concert",
                "date night", "brunch", "coffee run", "class", "work", "weekend",
                "vacation", "travel", "gym", "athleisure", "loungewear"
            ],
            "fits": [
                "oversized", "baggy", "loose", "relaxed", "fitted", "tight",
                "cropped", "high waisted", "low rise", "mid rise", "wide leg",
                "straight leg", "skinny", "slim", "flared", "bootcut"
            ],
            "trends": [
                "layered", "matching sets", "co-ord", "twinning", "color blocking",
                "monochromatic", "mixed prints", "texture mixing", "vintage mix",
                "high-low mix", "sporty chic", "athleisure", "comfort wear",
                "gender neutral", "unisex", "androgynous", "oversized blazer",
                "cargo pants", "platform shoes", "chunky sneakers", "bucket hat"
            ]
        }
        
        # Brand Recognition Patterns
        self.brand_indicators = {
            "supreme": ["supreme", "box logo", "bogo"],
            "off_white": ["off white", "off-white", "virgil abloh", "quotation marks"],
            "anti_social_social_club": ["anti social social club", "assc"],
            "golf_wang": ["golf wang", "golf", "tyler the creator"],
            "kith": ["kith", "kith nyc"],
            "fear_of_god": ["fear of god", "fog", "essentials"],
            "yeezy": ["yeezy", "adidas yeezy", "kanye"],
            "palace": ["palace", "palace skateboards"],
            "bape": ["bape", "a bathing ape", "baby milo"],
            "stussy": ["stussy", "stüssy"],
            "thrasher": ["thrasher", "thrasher magazine"],
            "champion": ["champion", "champion athletics"],
            "nike": ["nike", "swoosh", "just do it"],
            "adidas": ["adidas", "three stripes", "trefoil"],
            "vans": ["vans", "off the wall"],
            "converse": ["converse", "chuck taylor", "all star"]
        }
        
        # Color Trends
        self.color_trends = [
            "neon green", "hot pink", "electric blue", "cyber yellow",
            "sage green", "dusty pink", "lavender", "baby blue",
            "earth brown", "mushroom", "clay", "terracotta",
            "black", "white", "grey", "beige", "cream",
            "tie dye", "gradient", "ombre", "color block"
        ]
    
    def preprocess_image(self, image_path: str) -> Tuple[Image.Image, np.ndarray]:
        """Enhanced preprocessing for streetwear images"""
        try:
            image = Image.open(image_path).convert("RGB")
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Enhance contrast for better text/logo detection
            cv_image = cv2.convertScaleAbs(cv_image, alpha=1.2, beta=15)
            
            enhanced_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            return enhanced_image, cv_image
            
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")
    
    def detect_streetwear_objects(self, image: Image.Image) -> List[Dict]:
        """Enhanced object detection for streetwear items"""
        results = self.yolo_model(image, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_name = result.names[int(box.cls)]
                    confidence = float(box.conf)
                    
                    # Filter for clothing-relevant objects
                    if class_name in ['person', 'handbag', 'backpack', 'tie', 'suitcase', 'sports ball'] or confidence > 0.3:
                        detection = {
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': box.xyxy[0].cpu().numpy().tolist()
                        }
                        detections.append(detection)
        
        return detections
    
    def generate_streetwear_caption(self, image: Image.Image) -> str:
        """Generate caption focused on streetwear elements"""
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.blip_model.generate(**inputs, max_length=100, num_beams=5)
        
        caption = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption.strip()
    
    def classify_streetwear_category(self, image: Image.Image) -> Tuple[str, float]:
        """Classify using streetwear-specific categories"""
        # Flatten all category options
        all_categories = []
        category_map = {}
        
        for main_cat, variations in self.streetwear_categories.items():
            all_categories.extend(variations)
            for variation in variations:
                category_map[variation] = main_cat
        
        prediction, confidence = self.classify_with_openclip(image, all_categories)
        main_category = category_map.get(prediction, prediction)
        
        return main_category, confidence
    
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
    
    def analyze_genz_style(self, image: Image.Image, caption: str) -> Dict:
        """Analyze Gen Z style attributes"""
        style_analysis = {}
        
        # Aesthetic classification
        aesthetics = self.genz_styles["aesthetics"]
        aesthetic, aesthetic_conf = self.classify_with_openclip(image, aesthetics)
        style_analysis["aesthetic"] = {"style": aesthetic, "confidence": aesthetic_conf}
        
        # Vibe classification
        vibes = self.genz_styles["vibes"]
        vibe, vibe_conf = self.classify_with_openclip(image, vibes)
        style_analysis["vibe"] = {"style": vibe, "confidence": vibe_conf}
        
        # Fit classification
        fits = self.genz_styles["fits"]
        fit, fit_conf = self.classify_with_openclip(image, fits)
        style_analysis["fit"] = {"style": fit, "confidence": fit_conf}
        
        # Trend detection
        trends = self.genz_styles["trends"]
        trend, trend_conf = self.classify_with_openclip(image, trends)
        style_analysis["trend"] = {"style": trend, "confidence": trend_conf}
        
        return style_analysis
    
    def detect_brands(self, caption: str) -> List[str]:
        """Detect streetwear brands from caption"""
        detected_brands = []
        caption_lower = caption.lower()
        
        for brand, indicators in self.brand_indicators.items():
            for indicator in indicators:
                if indicator in caption_lower:
                    detected_brands.append(brand)
                    break
        
        return list(set(detected_brands))
    
    def classify_streetwear_gender(self, image: Image.Image) -> Tuple[str, float]:
        """Gender classification for streetwear"""
        gender_candidates = [
            "mens streetwear", "womens streetwear", "unisex streetwear",
            "masculine street fashion", "feminine street fashion", "gender neutral fashion",
            "boys clothing", "girls clothing", "unisex clothing"
        ]
        
        prediction, confidence = self.classify_with_openclip(image, gender_candidates)
        
        # Map to simple categories
        if any(term in prediction for term in ["mens", "masculine", "boys"]):
            return "male", confidence
        elif any(term in prediction for term in ["womens", "feminine", "girls"]):
            return "female", confidence
        else:
            return "unisex", confidence
    
    def extract_streetwear_colors(self, image: Image.Image) -> List[str]:
        """Extract color information relevant to streetwear"""
        color_candidates = self.color_trends
        
        # Get top 3 colors
        image_tensor = self.openclip_preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = self.openclip_tokenizer(color_candidates).to(self.device)
        
        with torch.no_grad():
            image_features = self.openclip_model.encode_image(image_tensor)
            text_features = self.openclip_model.encode_text(text_tokens)
            
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            similarities = (image_features @ text_features.T).softmax(dim=-1)
        
        top_indices = similarities.topk(3).indices.cpu().numpy()[0]
        top_colors = [color_candidates[i] for i in top_indices]
        
        return top_colors
    
    def generate_streetwear_tags(self, caption: str, detected_objects: List[Dict], 
                                style_analysis: Dict, colors: List[str]) -> List[str]:
        """Generate streetwear-specific tags"""
        tags = set()
        
        # Extract from caption
        caption_words = caption.lower().split()
        streetwear_keywords = [
            "streetwear", "urban", "casual", "trendy", "fashion", "style",
            "outfit", "look", "aesthetic", "vibe", "fit", "drip"
        ]
        
        for word in caption_words:
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) > 2 and clean_word not in ["the", "and", "with", "wearing"]:
                tags.add(clean_word)
        
        # Add style attributes
        for attr_type, attr_data in style_analysis.items():
            if attr_data["confidence"] > 0.3:
                tags.add(attr_data["style"])
        
        # Add colors
        tags.update(colors)
        
        # Add detected objects
        for obj in detected_objects:
            if obj["confidence"] > 0.5:
                tags.add(obj["class"])
        
        return list(tags)[:20]  # Limit to top 20 tags
    
    def analyze_streetwear_image(self, image_path: str) -> Dict:
        """Comprehensive streetwear image analysis"""
        start_time = time.time()
        print(f"Analyzing streetwear image: {image_path}")
        
        # Load and preprocess
        image, cv_image = self.preprocess_image(image_path)
        
        # Generate caption
        print("Generating streetwear caption...")
        caption = self.generate_streetwear_caption(image)
        
        # Object detection
        print("Detecting streetwear objects...")
        detected_objects = self.detect_streetwear_objects(image)
        
        # Category classification
        print("Classifying streetwear category...")
        category, category_conf = self.classify_streetwear_category(image)
        
        # Gender classification
        print("Classifying gender...")
        gender, gender_conf = self.classify_streetwear_gender(image)
        
        # Style analysis
        print("Analyzing Gen Z style...")
        style_analysis = self.analyze_genz_style(image, caption)
        
        # Brand detection
        print("Detecting brands...")
        detected_brands = self.detect_brands(caption)
        
        # Color analysis
        print("Extracting colors...")
        colors = self.extract_streetwear_colors(image)
        
        # Generate tags
        print("Generating streetwear tags...")
        tags = self.generate_streetwear_tags(caption, detected_objects, style_analysis, colors)
        
        processing_time = time.time() - start_time
        
        result = {
            "image_path": image_path,
            "processing_time": round(processing_time, 2),
            "caption": caption,
            "detected_objects": detected_objects,
            "category": category,
            "category_confidence": round(category_conf, 3),
            "gender": gender,
            "gender_confidence": round(gender_conf, 3),
            "style_analysis": {
                k: {"style": v["style"], "confidence": round(v["confidence"], 3)} 
                for k, v in style_analysis.items()
            },
            "detected_brands": detected_brands,
            "colors": colors,
            "tags": tags,
            "streetwear_recommendations": {
                "primary_category": category,
                "aesthetic": style_analysis.get("aesthetic", {}).get("style", ""),
                "vibe": style_analysis.get("vibe", {}).get("style", ""),
                "fit": style_analysis.get("fit", {}).get("style", ""),
                "trend": style_analysis.get("trend", {}).get("style", ""),
                "colors": colors[:2],
                "brands": detected_brands,
                "tags": tags[:10]
            },
            "model_info": {
                "specialized_for": "streetwear_genz_fashion",
                "models": ["blip2-opt-2.7b", "openclip-vit-b-32", "yolov8x"],
                "device": str(self.device)
            }
        }
        
        print(f"Streetwear analysis completed in {processing_time:.2f} seconds")
        return result


def main():
    parser = argparse.ArgumentParser(description="Streetwear & Gen Z Fashion Image Tagger")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-gpu-opt", action="store_true", help="Disable GPU optimizations")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    # Initialize streetwear tagger
    use_gpu_opt = not args.no_gpu_opt
    tagger = StreetwearTagger(use_gpu_optimization=use_gpu_opt)
    
    # Analyze image
    try:
        result = tagger.analyze_streetwear_image(args.image_path)
        
        # Display results
        if args.verbose:
            print("\n" + "="*60)
            print("STREETWEAR & GEN Z FASHION ANALYSIS")
            print("="*60)
            print(f"Image: {result['image_path']}")
            print(f"Processing Time: {result['processing_time']}s")
            print(f"Caption: {result['caption']}")
            print(f"Category: {result['category']} (confidence: {result['category_confidence']})")
            print(f"Gender: {result['gender']} (confidence: {result['gender_confidence']})")
            print(f"Aesthetic: {result['style_analysis']['aesthetic']['style']}")
            print(f"Vibe: {result['style_analysis']['vibe']['style']}")
            print(f"Fit: {result['style_analysis']['fit']['style']}")
            print(f"Trend: {result['style_analysis']['trend']['style']}")
            print(f"Colors: {', '.join(result['colors'])}")
            print(f"Brands: {', '.join(result['detected_brands']) if result['detected_brands'] else 'None detected'}")
            print(f"Tags: {', '.join(result['tags'])}")
        else:
            print(json.dumps(result, indent=2))
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
            
    except Exception as e:
        print(f"Error analyzing image: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()