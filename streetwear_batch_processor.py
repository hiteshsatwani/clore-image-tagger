#!/usr/bin/env python3
"""
Streetwear Batch Processing Script
Optimized for processing multiple streetwear images with trend analysis.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from collections import Counter

from streetwear_tagger import StreetwearTagger
from PIL import Image
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StreetwearBatchProcessor:
    def __init__(self, use_gpu_optimization: bool = True, max_workers: int = 2):
        self.tagger = StreetwearTagger(use_gpu_optimization=use_gpu_optimization)
        self.max_workers = max_workers
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def find_images(self, directory: str, recursive: bool = True) -> List[str]:
        """Find all supported image files in directory"""
        image_files = []
        path = Path(directory)
        
        if recursive:
            for ext in self.supported_formats:
                image_files.extend(path.rglob(f'*{ext}'))
                image_files.extend(path.rglob(f'*{ext.upper()}'))
        else:
            for ext in self.supported_formats:
                image_files.extend(path.glob(f'*{ext}'))
                image_files.extend(path.glob(f'*{ext.upper()}'))
        
        return [str(f) for f in image_files]
    
    def validate_image(self, image_path: str) -> bool:
        """Validate if image can be processed"""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception as e:
            logger.warning(f"Invalid image {image_path}: {e}")
            return False
    
    def process_single_image(self, image_path: str) -> Dict:
        """Process a single streetwear image"""
        try:
            if not self.validate_image(image_path):
                return {
                    "image_path": image_path,
                    "error": "Invalid image format",
                    "processed": False
                }
            
            result = self.tagger.analyze_streetwear_image(image_path)
            result["processed"] = True
            return result
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                "image_path": image_path,
                "error": str(e),
                "processed": False
            }
    
    def process_batch(self, image_paths: List[str], output_dir: str = None) -> Dict:
        """Process multiple streetwear images"""
        start_time = time.time()
        results = []
        successful = 0
        failed = 0
        
        logger.info(f"Processing {len(image_paths)} streetwear images...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self.process_single_image, path): path 
                for path in image_paths
            }
            
            for future in as_completed(future_to_path):
                result = future.result()
                results.append(result)
                
                if result.get("processed", False):
                    successful += 1
                    logger.info(f"✓ Processed: {result['image_path']}")
                else:
                    failed += 1
                    logger.error(f"✗ Failed: {result['image_path']}")
        
        processing_time = time.time() - start_time
        
        # Generate streetwear analytics
        analytics = self.generate_streetwear_analytics(results)
        
        summary = {
            "batch_info": {
                "total_images": len(image_paths),
                "successful": successful,
                "failed": failed,
                "processing_time": round(processing_time, 2),
                "avg_time_per_image": round(processing_time / len(image_paths), 2) if image_paths else 0
            },
            "streetwear_analytics": analytics,
            "results": results
        }
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save comprehensive results
            results_file = os.path.join(output_dir, "streetwear_batch_results.json")
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save streetwear analytics
            analytics_file = os.path.join(output_dir, "streetwear_analytics.json")
            with open(analytics_file, 'w') as f:
                json.dump(analytics, f, indent=2)
            
            # Save CSV
            csv_file = os.path.join(output_dir, "streetwear_summary.csv")
            self.save_streetwear_csv(results, csv_file)
            
            logger.info(f"Results saved to {output_dir}")
        
        return summary
    
    def generate_streetwear_analytics(self, results: List[Dict]) -> Dict:
        """Generate comprehensive streetwear analytics"""
        successful_results = [r for r in results if r.get("processed", False)]
        
        if not successful_results:
            return {"error": "No successful results to analyze"}
        
        # Category distribution
        categories = Counter()
        for result in successful_results:
            categories[result.get("category", "unknown")] += 1
        
        # Gender distribution
        genders = Counter()
        for result in successful_results:
            genders[result.get("gender", "unknown")] += 1
        
        # Aesthetic trends
        aesthetics = Counter()
        for result in successful_results:
            aesthetic = result.get("style_analysis", {}).get("aesthetic", {}).get("style", "")
            if aesthetic:
                aesthetics[aesthetic] += 1
        
        # Vibe trends
        vibes = Counter()
        for result in successful_results:
            vibe = result.get("style_analysis", {}).get("vibe", {}).get("style", "")
            if vibe:
                vibes[vibe] += 1
        
        # Fit trends
        fits = Counter()
        for result in successful_results:
            fit = result.get("style_analysis", {}).get("fit", {}).get("style", "")
            if fit:
                fits[fit] += 1
        
        # Brand analysis
        brands = Counter()
        for result in successful_results:
            for brand in result.get("detected_brands", []):
                brands[brand] += 1
        
        # Color trends
        colors = Counter()
        for result in successful_results:
            for color in result.get("colors", []):
                colors[color] += 1
        
        # Tag frequency
        tags = Counter()
        for result in successful_results:
            for tag in result.get("tags", []):
                tags[tag] += 1
        
        analytics = {
            "total_processed": len(successful_results),
            "category_distribution": dict(categories.most_common(10)),
            "gender_distribution": dict(genders),
            "aesthetic_trends": dict(aesthetics.most_common(10)),
            "vibe_trends": dict(vibes.most_common(10)),
            "fit_trends": dict(fits.most_common(10)),
            "brand_presence": dict(brands.most_common(10)),
            "color_trends": dict(colors.most_common(10)),
            "top_tags": dict(tags.most_common(20)),
            "insights": self.generate_insights(categories, aesthetics, vibes, fits, brands, colors)
        }
        
        return analytics
    
    def generate_insights(self, categories: Counter, aesthetics: Counter, 
                         vibes: Counter, fits: Counter, brands: Counter, colors: Counter) -> Dict:
        """Generate trend insights"""
        insights = {}
        
        # Top category
        if categories:
            top_category = categories.most_common(1)[0]
            insights["dominant_category"] = {
                "category": top_category[0],
                "percentage": round(top_category[1] / sum(categories.values()) * 100, 1)
            }
        
        # Top aesthetic
        if aesthetics:
            top_aesthetic = aesthetics.most_common(1)[0]
            insights["trending_aesthetic"] = {
                "aesthetic": top_aesthetic[0],
                "percentage": round(top_aesthetic[1] / sum(aesthetics.values()) * 100, 1)
            }
        
        # Top vibe
        if vibes:
            top_vibe = vibes.most_common(1)[0]
            insights["popular_vibe"] = {
                "vibe": top_vibe[0],
                "percentage": round(top_vibe[1] / sum(vibes.values()) * 100, 1)
            }
        
        # Brand diversity
        insights["brand_diversity"] = {
            "unique_brands": len(brands),
            "most_common_brand": brands.most_common(1)[0][0] if brands else "None"
        }
        
        # Color trends
        insights["color_trends"] = {
            "dominant_colors": [color for color, _ in colors.most_common(3)],
            "color_diversity": len(colors)
        }
        
        return insights
    
    def save_streetwear_csv(self, results: List[Dict], csv_file: str):
        """Save streetwear results as CSV"""
        import csv
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'image_path', 'processed', 'category', 'category_confidence',
                'gender', 'gender_confidence', 'aesthetic', 'vibe', 'fit', 'trend',
                'colors', 'brands', 'top_tags', 'processing_time'
            ])
            
            for result in results:
                if result.get("processed", False):
                    style = result.get("style_analysis", {})
                    writer.writerow([
                        result['image_path'],
                        'Success',
                        result.get('category', ''),
                        result.get('category_confidence', ''),
                        result.get('gender', ''),
                        result.get('gender_confidence', ''),
                        style.get('aesthetic', {}).get('style', ''),
                        style.get('vibe', {}).get('style', ''),
                        style.get('fit', {}).get('style', ''),
                        style.get('trend', {}).get('style', ''),
                        ', '.join(result.get('colors', [])),
                        ', '.join(result.get('detected_brands', [])),
                        ', '.join(result.get('tags', [])[:5]),
                        result.get('processing_time', '')
                    ])
                else:
                    writer.writerow([
                        result['image_path'],
                        'Failed',
                        '', '', '', '', '', '', '', '', '', '', '',
                        result.get('error', '')
                    ])
    
    def generate_streetwear_report(self, summary: Dict) -> str:
        """Generate detailed streetwear report"""
        batch_info = summary["batch_info"]
        analytics = summary["streetwear_analytics"]
        
        report = f"""
STREETWEAR & GEN Z FASHION BATCH ANALYSIS REPORT
================================================

Batch Summary:
- Total Images: {batch_info['total_images']}
- Successful: {batch_info['successful']} ({batch_info['successful']/batch_info['total_images']*100:.1f}%)
- Failed: {batch_info['failed']} ({batch_info['failed']/batch_info['total_images']*100:.1f}%)
- Processing Time: {batch_info['processing_time']}s
- Average Time: {batch_info['avg_time_per_image']}s per image

STREETWEAR ANALYTICS
===================

Category Distribution:
"""
        
        for category, count in analytics.get("category_distribution", {}).items():
            percentage = count / analytics.get("total_processed", 1) * 100
            report += f"- {category}: {count} ({percentage:.1f}%)\n"
        
        report += f"\nGender Distribution:\n"
        for gender, count in analytics.get("gender_distribution", {}).items():
            percentage = count / analytics.get("total_processed", 1) * 100
            report += f"- {gender}: {count} ({percentage:.1f}%)\n"
        
        report += f"\nTrending Aesthetics:\n"
        for aesthetic, count in list(analytics.get("aesthetic_trends", {}).items())[:5]:
            report += f"- {aesthetic}: {count}\n"
        
        report += f"\nPopular Vibes:\n"
        for vibe, count in list(analytics.get("vibe_trends", {}).items())[:5]:
            report += f"- {vibe}: {count}\n"
        
        report += f"\nBrand Presence:\n"
        for brand, count in analytics.get("brand_presence", {}).items():
            report += f"- {brand}: {count}\n"
        
        report += f"\nColor Trends:\n"
        for color, count in list(analytics.get("color_trends", {}).items())[:5]:
            report += f"- {color}: {count}\n"
        
        # Add insights
        insights = analytics.get("insights", {})
        if insights:
            report += f"\nKEY INSIGHTS:\n"
            
            if "dominant_category" in insights:
                dom_cat = insights["dominant_category"]
                report += f"- Dominant Category: {dom_cat['category']} ({dom_cat['percentage']}%)\n"
            
            if "trending_aesthetic" in insights:
                trend_aes = insights["trending_aesthetic"]
                report += f"- Trending Aesthetic: {trend_aes['aesthetic']} ({trend_aes['percentage']}%)\n"
            
            if "popular_vibe" in insights:
                pop_vibe = insights["popular_vibe"]
                report += f"- Popular Vibe: {pop_vibe['vibe']} ({pop_vibe['percentage']}%)\n"
            
            if "brand_diversity" in insights:
                brand_div = insights["brand_diversity"]
                report += f"- Brand Diversity: {brand_div['unique_brands']} unique brands\n"
                report += f"- Most Common Brand: {brand_div['most_common_brand']}\n"
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Streetwear Batch Processing")
    parser.add_argument("input_path", help="Path to image file or directory")
    parser.add_argument("--output", "-o", help="Output directory for results")
    parser.add_argument("--recursive", "-r", action="store_true", help="Process directories recursively")
    parser.add_argument("--workers", "-w", type=int, default=2, help="Number of worker threads")
    parser.add_argument("--no-gpu-opt", action="store_true", help="Disable GPU optimizations")
    parser.add_argument("--report", action="store_true", help="Generate detailed streetwear report")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Path not found: {args.input_path}")
        return
    
    # Initialize processor
    use_gpu_opt = not args.no_gpu_opt
    processor = StreetwearBatchProcessor(use_gpu_optimization=use_gpu_opt, max_workers=args.workers)
    
    # Get image paths
    if os.path.isfile(args.input_path):
        image_paths = [args.input_path]
    else:
        image_paths = processor.find_images(args.input_path, recursive=args.recursive)
    
    if not image_paths:
        print("No streetwear images found!")
        return
    
    print(f"Found {len(image_paths)} streetwear images to process")
    
    # Process batch
    try:
        summary = processor.process_batch(image_paths, output_dir=args.output)
        
        # Display results
        batch_info = summary["batch_info"]
        analytics = summary["streetwear_analytics"]
        
        print(f"\nStreetware Batch Processing Complete!")
        print(f"Successful: {batch_info['successful']}/{batch_info['total_images']}")
        print(f"Processing Time: {batch_info['processing_time']}s")
        
        # Show quick analytics
        if "insights" in analytics:
            insights = analytics["insights"]
            print(f"\nQuick Insights:")
            if "dominant_category" in insights:
                dom_cat = insights["dominant_category"]
                print(f"- Dominant Category: {dom_cat['category']} ({dom_cat['percentage']}%)")
            if "trending_aesthetic" in insights:
                trend_aes = insights["trending_aesthetic"]
                print(f"- Trending Aesthetic: {trend_aes['aesthetic']} ({trend_aes['percentage']}%)")
        
        # Generate full report
        if args.report:
            report = processor.generate_streetwear_report(summary)
            print(report)
            
            if args.output:
                report_file = os.path.join(args.output, "streetwear_report.txt")
                with open(report_file, 'w') as f:
                    f.write(report)
                print(f"Report saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()