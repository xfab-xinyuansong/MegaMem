"""
Interactive memory browser.

REPL-style command loop on top of :class:`ChromaBrowser` for exploring
megamem-managed memory stores from the terminal.
"""

import os
import sys
import json
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

from .chroma_browser import ChromaBrowser, ChromaDocument
from ..utils.log import configure_logging

logger = logging.getLogger(__name__)


def _coerce_filter_value(raw: str):
    """Best-effort conversion of a string filter value to its native type."""
    lowered = raw.lower()
    if lowered in ('true', 'false'):
        return lowered == 'true'
    if raw.isdigit():
        return int(raw)
    if raw.replace('.', '').isdigit():
        return float(raw)
    return raw


class InteractiveMemoryBrowser:
    """
    Curses-free interactive browser for ChromaDB-backed memory stores.

    Wraps :class:`ChromaBrowser` and dispatches typed commands to inspect,
    search, filter, view and export documents.
    """

    def __init__(self, db_path: str, collection_name: str = None):
        """
        Build an interactive browser instance.

        Args:
            db_path: Path to ChromaDB database directory
            collection_name: Name of collection to browse (optional)
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            self.chroma_browser = ChromaBrowser(
                db_path=str(self.db_path),
                collection_name=collection_name,
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB browser: {str(e)}")
            raise

        self.current_documents: List[ChromaDocument] = []
        self.current_filter: Optional[Dict[str, Any]] = None

    def run(self) -> None:
        """Enter the REPL command loop."""
        print("\n" + "=" * 70)
        print("MEGAMEM INTERACTIVE MEMORY BROWSER")
        print("=" * 70)
        print(f"📁 Database: {self.db_path}")
        print(f"📊 Collection: {self.chroma_browser.collection_name}")
        print("Type 'help' for available commands, 'quit' to exit")

        self._show_quick_stats()

        # Map commands to handlers; argless handlers are wrapped to ignore args.
        argful = {
            'switch': self._switch_collection,
            'list': self._list_documents,
            'search': self._search_documents,
            'filter': self._filter_documents,
            'show': self._show_document_detail,
            'export': self._export_documents,
            'metadata': self._show_metadata_info,
        }
        argless = {
            'help': self._show_help,
            'stats': self._show_detailed_stats,
            'collections': self._list_collections,
            'clear': self._clear_filter,
            'count': self._count_documents,
        }
        quit_aliases = {'quit', 'exit', 'q'}

        while True:
            try:
                command = input("\n💭 Memory Browser > ").strip()
                if not command:
                    continue

                parts = command.split()
                cmd = parts[0].lower()
                cmd_args = parts[1:]

                if cmd in quit_aliases:
                    print("Goodbye! 👋")
                    break

                if cmd in argful:
                    argful[cmd](cmd_args)
                elif cmd in argless:
                    argless[cmd]()
                else:
                    print(f"❌ Unknown command: '{cmd}'. Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                self.logger.error(f"Interactive browser error: {str(e)}")

    def _show_help(self) -> None:
        """Print the in-app command cheat-sheet."""
        print("\n📚 AVAILABLE COMMANDS:")
        print("-" * 50)
        print("🔍 Collection Management:")
        print("  collections           - List all collections")
        print("  switch <name>         - Switch to different collection")
        print("  stats                 - Show detailed collection statistics")
        print()
        print("📋 Document Browsing:")
        print("  list [limit]          - List documents (default: 10)")
        print("  count                 - Count total documents")
        print("  show <id|index>       - Show detailed document view")
        print()
        print("🔎 Search & Filter:")
        print("  search <query>        - Semantic search in documents")
        print("  filter <key>=<value>  - Filter by metadata")
        print("  clear                 - Clear current filters")
        print()
        print("📊 Analysis:")
        print("  metadata [key]        - Show metadata information")
        print("  stats                 - Detailed collection statistics")
        print()
        print("💾 Export:")
        print("  export <format> <path> - Export current documents")
        print("                          (formats: json, csv, txt)")
        print()
        print("ℹ️  System:")
        print("  help                  - Show this help message")
        print("  quit/exit/q           - Exit the browser")
        print()

        suffix = " (filtered)" if self.current_filter else ""
        print(f"📊 Current Status: {len(self.current_documents)} documents loaded{suffix}")

    def _show_quick_stats(self) -> None:
        """Print a one-line summary of the active collection."""
        try:
            stats = self.chroma_browser.get_collection_stats()
            print(f"📈 Quick Stats: {stats.total_documents:,} documents, {len(stats.metadata_keys)} metadata fields")
        except Exception as e:
            print(f"❌ Failed to load stats: {str(e)}")

    def _show_detailed_stats(self) -> None:
        """Print a multi-section view of collection statistics."""
        print("\n📊 COLLECTION STATISTICS")
        print("-" * 50)

        try:
            stats = self.chroma_browser.get_collection_stats()

            print(f"📋 Collection: {stats.collection_name}")
            print(f"📈 Total Documents: {stats.total_documents:,}")

            if stats.content_stats:
                cs = stats.content_stats
                print(f"\n📝 Content Statistics:")
                print(f"  Total Characters: {cs.get('total_characters', 0):,}")
                print(f"  Average Length: {cs.get('average_length', 0):.1f}")
                print(f"  Min Length: {cs.get('min_length', 0)}")
                print(f"  Max Length: {cs.get('max_length', 0)}")

            if stats.metadata_keys:
                print(f"\n🏷️  Metadata Fields ({len(stats.metadata_keys)}):")
                for key in sorted(stats.metadata_keys):
                    n_unique = stats.unique_metadata_values.get(key, 0)
                    print(f"  • {key}: {n_unique} unique values")

        except Exception as e:
            print(f"❌ Failed to load detailed stats: {str(e)}")

    def _list_collections(self) -> None:
        """Print every collection in the persistent store."""
        print("\n📚 AVAILABLE COLLECTIONS")
        print("-" * 50)

        try:
            names = self.chroma_browser.list_collections()
            if not names:
                print("📭 No collections found")
                return

            current = self.chroma_browser.collection_name
            for pos, name in enumerate(names, 1):
                marker = "👉" if name == current else "  "
                print(f"{marker} {pos}. {name}")

        except Exception as e:
            print(f"❌ Failed to list collections: {str(e)}")

    def _switch_collection(self, args: List[str]) -> None:
        """Switch to the collection named in ``args``."""
        if not args:
            print("❌ Usage: switch <collection_name>")
            return

        target = args[0]

        try:
            if not self.chroma_browser.switch_collection(target):
                print(f"❌ Failed to switch to collection: {target}")
                return

            self.current_documents = []  # drop cached docs from prev. collection
            self.current_filter = None
            print(f"✅ Switched to collection: {target}")
            self._show_quick_stats()

        except Exception as e:
            print(f"❌ Switch failed: {str(e)}")

    def _list_documents(self, args: List[str]) -> None:
        """Print the first N documents in the active collection."""
        limit = 10

        if args:
            try:
                limit = int(args[0])
            except ValueError:
                print("❌ Invalid limit. Using default of 10.")

        print(f"\n📋 DOCUMENTS (showing up to {limit})")
        print("-" * 70)

        try:
            documents = self.chroma_browser.get_all_documents(limit=limit)

            if not documents:
                print("📭 No documents found")
                return

            for pos, doc in enumerate(documents, 1):
                metadata_preview = ""
                if doc.metadata:
                    snippets = [f"{k}={v}" for k, v in list(doc.metadata.items())[:2]]
                    metadata_preview = " | " + ", ".join(snippets)
                    if len(doc.metadata) > 2:
                        metadata_preview += "..."

                preview = doc.content[:60] + "..." if len(doc.content) > 60 else doc.content

                print(f"{pos:2d}. 🆔 {doc.id}")
                print(f"    📏 {len(doc.content)} chars{metadata_preview}")
                print(f"    💬 {preview}")
                print()

            self.current_documents = documents

        except Exception as e:
            print(f"❌ Failed to load documents: {str(e)}")

    def _search_documents(self, args: List[str]) -> None:
        """Run a semantic search over ``args`` joined into a query."""
        if not args:
            print("❌ Usage: search <query>")
            return

        query = " ".join(args)
        print(f"\n🔍 SEARCH RESULTS for: '{query}'")
        print("-" * 70)

        try:
            documents = self.chroma_browser.search_documents(query=query, n_results=10)

            if not documents:
                print("📭 No matching documents found")
                return

            for pos, doc in enumerate(documents, 1):
                score_info = f" (distance: {doc.distance:.3f})" if doc.distance is not None else ""
                preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content

                print(f"{pos:2d}. 🆔 {doc.id[:20]}...{score_info}")
                print(f"    📏 {len(doc.content)} chars")
                print(f"    💬 {preview}")
                print()

            self.current_documents = documents

        except Exception as e:
            print(f"❌ Search failed: {str(e)}")

    def _filter_documents(self, args: List[str]) -> None:
        """Apply a metadata equality filter parsed from ``key=value``."""
        if not args:
            print("❌ Usage: filter <key>=<value>")
            print("   Example: filter source_file=handbook.md")
            return

        filter_expr = " ".join(args)

        try:
            if "=" not in filter_expr:
                print("❌ Invalid filter format. Use: key=value")
                return

            key, raw_value = filter_expr.split("=", 1)
            key = key.strip()
            value = _coerce_filter_value(raw_value.strip())

            where_filter = {key: value}

            print(f"\n🔎 FILTERED RESULTS for: {key}={value}")
            print("-" * 70)

            documents = self.chroma_browser.filter_documents(where_filter=where_filter, limit=50)

            if not documents:
                print("📭 No documents match the filter")
                return

            for pos, doc in enumerate(documents, 1):
                metadata_value = doc.metadata.get(key, "N/A")
                preview = doc.content[:60] + "..." if len(doc.content) > 60 else doc.content

                print(f"{pos:2d}. 🆔 {doc.id}")
                print(f"    🏷️  {key}: {metadata_value}")
                print(f"    💬 {preview}")
                print()

            self.current_documents = documents
            self.current_filter = where_filter

        except Exception as e:
            print(f"❌ Filter failed: {str(e)}")

    def _clear_filter(self) -> None:
        """Drop the active filter and any cached documents."""
        self.current_filter = None
        self.current_documents = []
        print("✅ Filter cleared")

    def _show_document_detail(self, args: List[str]) -> None:
        """Show the full record for a numeric index or document id."""
        if not args:
            print("❌ Usage: show <document_id_or_index>")
            return

        identifier = args[0]
        document: Optional[ChromaDocument] = None

        try:
            if identifier.isdigit():
                pos = int(identifier) - 1
                if 0 <= pos < len(self.current_documents):
                    document = self.current_documents[pos]
                else:
                    print(f"❌ Invalid index. Use 1-{len(self.current_documents)}")
                    return
            else:
                # Fall back to scanning all docs by id / id-prefix.
                for doc in self.chroma_browser.get_all_documents():
                    if doc.id == identifier or doc.id.startswith(identifier):
                        document = doc
                        break

                if document is None:
                    print(f"❌ Document not found: {identifier}")
                    return

            print(f"\n🔍 DOCUMENT DETAILS")
            print("=" * 70)
            print(f"🆔 ID: {document.id}")
            print(f"📏 Content Length: {len(document.content)} characters")

            if document.distance is not None:
                print(f"📐 Distance: {document.distance:.6f}")

            if document.metadata:
                print(f"\n🏷️  METADATA:")
                print("-" * 40)
                for key, value in document.metadata.items():
                    print(f"  {key}: {value}")

            print(f"\n💬 CONTENT:")
            print("-" * 70)
            print(f"[Index] {document.content}")
            print(f"[Value] {document.metadata.get('value', 'N/A')}")
            print("-" * 70)

        except Exception as e:
            print(f"❌ Failed to show document details: {str(e)}")

    def _export_documents(self, args: List[str]) -> None:
        """Persist the currently-loaded documents in the requested format."""
        if len(args) < 2:
            print("❌ Usage: export <format> <filepath>")
            print("   Formats: json, csv, txt")
            print("   Example: export json ./documents.json")
            return

        format_type, file_path = args[0], args[1]

        if format_type not in ('json', 'csv', 'txt'):
            print("❌ Supported formats: json, csv, txt")
            return

        if not self.current_documents:
            print("❌ No documents to export. Load some documents first.")
            return

        try:
            self.chroma_browser.export_documents(
                documents=self.current_documents,
                file_path=file_path,
                format=format_type,
            )
            print(f"✅ Exported {len(self.current_documents)} documents to: {file_path}")

        except Exception as e:
            print(f"❌ Export failed: {str(e)}")

    def _count_documents(self) -> None:
        """Print the document count for the active collection."""
        try:
            stats = self.chroma_browser.get_collection_stats()
            print(f"📊 Total documents in collection: {stats.total_documents:,}")
        except Exception as e:
            print(f"❌ Failed to count documents: {str(e)}")

    def _show_metadata_info(self, args: List[str]) -> None:
        """Print metadata field summary, optionally drilling into one key."""
        print("\n🏷️  METADATA INFORMATION")
        print("-" * 50)

        try:
            stats = self.chroma_browser.get_collection_stats()

            if not stats.metadata_keys:
                print("📭 No metadata fields found")
                return

            if not args:
                print(f"📊 Total metadata fields: {len(stats.metadata_keys)}")
                print()
                for field in sorted(stats.metadata_keys):
                    count = stats.unique_metadata_values.get(field, 0)
                    print(f"  • {field}: {count} unique values")
                return

            field_name = args[0]
            if field_name not in stats.unique_metadata_values:
                print(f"❌ Metadata field not found: {field_name}")
                return

            n_unique = stats.unique_metadata_values[field_name]
            print(f"📊 Field: {field_name}")
            print(f"   Unique values: {n_unique}")

            sample_values: set = set()
            for doc in self.chroma_browser.get_all_documents(limit=50):
                if field_name in doc.metadata:
                    sample_values.add(str(doc.metadata[field_name]))
                    if len(sample_values) >= 10:
                        break

            if sample_values:
                print(f"   Sample values: {', '.join(list(sample_values)[:10])}")

        except Exception as e:
            print(f"❌ Failed to load metadata info: {str(e)}")


def main():
    """CLI entry point: launch the interactive memory browser."""
    import argparse

    parser = argparse.ArgumentParser(
        description="megamem Interactive Browser - Explore ChromaDB collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Browse default collection in database
  python -m megamem.browser ./memory_store
  
  # Browse specific collection
  python -m megamem.browser ./memory_store --collection megamem
  
  # Browse with verbose logging
  python -m megamem.browser ./memory_store --verbose
        """
    )

    parser.add_argument('db_path', help='Path to ChromaDB database directory')
    parser.add_argument('--collection', '-c',
                        help='Name of collection to browse (if not specified, uses first available)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    configure_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.exists(args.db_path):
        print(f"❌ Database path does not exist: {args.db_path}")
        sys.exit(1)

    try:
        browser = InteractiveMemoryBrowser(
            db_path=args.db_path,
            collection_name=args.collection,
        )
        browser.run()

    except Exception as e:
        print(f"❌ Failed to start memory browser: {str(e)}")
        logger.error(f"Memory browser startup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
