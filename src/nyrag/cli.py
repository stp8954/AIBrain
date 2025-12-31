import argparse
import sys

from nyrag.config import Config
from nyrag.logger import logger
from nyrag.pipeline.validation import PipelineValidationError, validate_pipeline_config
from nyrag.process import process_from_config


def main():
    """Main CLI entry point for nyrag."""
    parser = argparse.ArgumentParser(
        description="nyrag - Web crawler and document processor for RAG applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing crawl/processing data (skip already processed URLs/files)",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate pipeline configuration and exit without processing",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = Config.from_yaml(args.config)

        logger.info(f"Project: {config.name}")
        logger.info(f"Mode: {config.mode}")
        logger.info(f"Output directory: {config.get_output_path()}")

        # Validate-only mode
        if args.validate:
            logger.info("Validation mode enabled - checking configuration...")
            result = validate_pipeline_config(config)
            if result.valid:
                logger.success("Configuration validation passed!")
                for warning in result.warnings:
                    logger.warning(f"Warning: {warning}")
                sys.exit(0)
            else:
                logger.error(f"Configuration validation failed: {result.message}")
                sys.exit(1)

        if args.resume:
            logger.info("Resume mode enabled - will skip already processed items")

        logger.info("Vespa feeding enabled - documents will be fed to Vespa as they are processed")

        # Process based on config
        process_from_config(config, resume=args.resume)

        logger.success(f"Processing complete! Output saved to {config.get_output_path()}")

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
