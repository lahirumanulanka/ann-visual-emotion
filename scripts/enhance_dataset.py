#!/usr/bin/env python3
"""
Script to enhance the emotion recognition dataset from 48x48 to 224x224 pixels
using generative AI and advanced interpolation techniques.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from genai.synth_data import create_enhanced_dataset, get_available_methods


def main():
    parser = argparse.ArgumentParser(
        description="Enhance emotion dataset from 48x48 to 224x224 pixels"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="../data/raw",
        help="Path to original dataset directory"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="../data/enhanced",
        help="Path to save enhanced dataset"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="enhanced_bicubic",
        choices=get_available_methods(),
        help="Enhancement method to use"
    )
    
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Target image size (width height)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Convert paths to absolute
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Check if input directory exists
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    logger.info("Starting dataset enhancement...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Enhancement method: {args.method}")
    logger.info(f"Target size: {args.target_size[0]}x{args.target_size[1]}")
    
    try:
        create_enhanced_dataset(
            original_data_dir=str(input_dir),
            enhanced_data_dir=str(output_dir),
            method=args.method,
            target_size=tuple(args.target_size)
        )
        
        logger.info("Dataset enhancement completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during enhancement: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()