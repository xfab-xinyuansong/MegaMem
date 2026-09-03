"""
megamem.browser CLI

Command-line front end for the memory browser tooling.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from .interactive_browser import InteractiveMemoryBrowser
from .memory_viewer import MemoryViewer
from ..utils.log import configure_logging

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI ``argparse`` parser."""
    parser = argparse.ArgumentParser(
        description="megamem Browser - Interactive exploration of ChromaDB memory stores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive browser
  python -m megamem.browser /path/to/memory_store
  
  # Browse specific collection
  python -m megamem.browser /path/to/memory_store -c megamem
  
  # Generate analysis report
  python -m megamem.browser /path/to/memory_store --analyze --output report.json
  
  # Quick statistics
  python -m megamem.browser /path/to/memory_store --stats
  
  # Search and analyze
  python -m megamem.browser /path/to/memory_store --search "incident management"
        """
    )

    parser.add_argument('db_path', help='Path to ChromaDB database directory')
    parser.add_argument('--collection', '-c',
                        help='Name of collection to browse (if not specified, uses first available)')
    parser.add_argument('--interactive', '-i', action='store_true', default=True,
                        help='Launch interactive browser (default)')
    parser.add_argument('--stats', action='store_true',
                        help='Show quick statistics and exit')
    parser.add_argument('--analyze', action='store_true',
                        help='Generate comprehensive analysis report')
    parser.add_argument('--search', '-s', help='Search query to analyze')
    parser.add_argument('--output', '-o',
                        help='Output file path for analysis/search results')
    parser.add_argument('--limit', '-l', type=int, default=10,
                        help='Limit number of results (default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--no-interactive', action='store_true',
                        help='Disable interactive mode (for scripting)')
    return parser


def _dump_json(payload, path: str) -> None:
    """Persist ``payload`` as pretty-printed JSON at ``path``."""
    import json
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _print_search_results(payload: dict) -> None:
    """Render a brief, console-friendly view of a ``search_and_analyze`` payload."""
    print(f"\n📊 Search Results:")
    print(f"Query: {payload['query']}")
    print(f"Total Results: {payload['total_results']}")

    if 'distance_stats' in payload:
        dstats = payload['distance_stats']
        print(f"Distance Range: {dstats['min']:.3f} - {dstats['max']:.3f} (avg: {dstats['average']:.3f})")

    print("\n📋 Top Results:")
    for rank, hit in enumerate(payload.get('results', [])[:5], 1):
        score_suffix = f" (distance: {hit['distance']:.3f})" if hit['distance'] else ""
        print(f"{rank}. ID: {hit['id']}{score_suffix}")
        print(f"   Content: {hit['content_preview'][:100]}...")
        print()


def _print_summary(summary: dict) -> None:
    """Render a brief textual summary of the collection."""
    print("📊 Memory Summary:")
    print(f"Collection: {summary['collection_name']}")
    coll_stats = summary['statistics']
    print(f"Documents: {coll_stats['total_documents']:,}")
    print(f"Metadata Fields: {len(coll_stats['metadata_keys'])}")
    if coll_stats['content_stats']:
        print(f"Total Characters: {coll_stats['content_stats']['total_characters']:,}")
        print(f"Average Length: {coll_stats['content_stats']['average_length']:.1f}")


def main():
    """Entry point for the memory browser CLI."""
    opts = _build_arg_parser().parse_args()

    configure_logging()
    if opts.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.exists(opts.db_path):
        print(f"❌ Database path does not exist: {opts.db_path}")
        sys.exit(1)

    try:
        viewer = MemoryViewer(
            db_path=opts.db_path,
            collection_name=opts.collection,
        )

        if opts.stats:
            print(viewer.quick_stats())

        elif opts.search:
            print(f"🔍 Searching for: '{opts.search}'")
            results = viewer.search_and_analyze(
                query=opts.search,
                n_results=opts.limit,
            )

            if opts.output:
                _dump_json(results, opts.output)
                print(f"✅ Results saved to: {opts.output}")
            else:
                _print_search_results(results)

        elif opts.analyze:
            print("📊 Generating comprehensive analysis...")
            output_path = opts.output or f"memory_analysis_{viewer.chroma_browser.collection_name}.json"
            viewer.export_analysis_report(
                output_path=output_path,
                include_search_examples=True,
            )
            print(f"✅ Analysis report saved to: {output_path}")

        elif not opts.no_interactive:
            browser = InteractiveMemoryBrowser(
                db_path=opts.db_path,
                collection_name=opts.collection,
            )
            browser.run()

        else:
            summary = viewer.get_memory_summary(include_samples=True, sample_count=3)
            if opts.output:
                _dump_json(summary, opts.output)
                print(f"✅ Summary saved to: {opts.output}")
            else:
                _print_summary(summary)

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"CLI error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
