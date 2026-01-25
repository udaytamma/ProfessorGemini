#!/usr/bin/env python3
"""Generate a single topic using Professor Gemini pipeline.

Usage:
    python generate_topic.py "Topic Name"
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from core.pipeline import Pipeline
from utils.file_utils import FileManager
from config.settings import get_settings


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_topic.py 'Topic Name'")
        sys.exit(1)

    topic = sys.argv[1]
    settings = get_settings()

    print(f"Generating content for: {topic}")
    print(f"Model: {settings.gemini_model}")
    print(f"Critique: {'ON' if settings.enable_critique else 'OFF'}")
    print(f"Local Synthesis: {'ON' if settings.local_synthesis else 'OFF'}")
    print("-" * 50)

    # Run pipeline
    start = datetime.now()
    pipeline = Pipeline()
    result = pipeline.execute(topic)
    duration = (datetime.now() - start).total_seconds()

    print("-" * 50)
    print(f"Duration: {duration:.1f}s")
    print(f"Success: {result.success}")
    print(f"Sections: {result.total_sections}")
    print(f"Low confidence: {result.low_confidence_sections}")

    if not result.success:
        print(f"Error: {result.error}")
        sys.exit(1)

    # Print step details
    print("\nPipeline Steps:")
    for step in result.steps:
        method = step.metadata.get("method", "N/A")
        model = step.metadata.get("model", "N/A")
        print(f"  - {step.name}: {step.duration_ms}ms (method={method}, model={model})")

    # Save to Cyrus
    file_manager = FileManager(settings.cyrus_root_path)
    success, filepath, message = file_manager.save_guide(
        content=result.master_guide,
        low_confidence_count=result.low_confidence_sections,
    )

    if success:
        print(f"\nSaved to: {filepath}")
    else:
        print(f"\nFailed to save: {message}")
        # Print content to stdout as fallback
        print("\n--- CONTENT ---")
        print(result.master_guide[:500] + "...")


if __name__ == "__main__":
    main()
