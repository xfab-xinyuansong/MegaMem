"""
Memory viewer.

Headless companion to :class:`InteractiveMemoryBrowser` — provides
programmatic summaries, search analyses, metadata pattern analysis and
report export, all without a REPL.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

from .chroma_browser import ChromaBrowser, ChromaDocument, ChromaStats

logger = logging.getLogger(__name__)


class MemoryViewer:
    """
    Programmatic facade over a ChromaDB-backed memory store.

    Wraps :class:`ChromaBrowser` with helpers tailored for analysis and
    reporting; no interactive UI involved.
    """

    def __init__(self, db_path: str, collection_name: str = None):
        """
        Build the viewer.

        Args:
            db_path: Path to ChromaDB database directory
            collection_name: Name of collection to analyze (optional)
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

    def get_memory_summary(
        self,
        include_samples: bool = True,
        sample_count: int = 5,
    ) -> Dict[str, Any]:
        """
        Build a high-level summary of the active collection.

        Args:
            include_samples: Whether to include sample documents
            sample_count: Number of sample documents to include

        Returns:
            Dict[str, Any]: Memory summary
        """
        try:
            stats = self.chroma_browser.get_collection_stats()

            summary: Dict[str, Any] = {
                'database_path': str(self.db_path),
                'collection_name': stats.collection_name,
                'analysis_timestamp': datetime.now().isoformat(),
                'statistics': asdict(stats),
            }

            if include_samples and stats.total_documents > 0:
                docs = self.chroma_browser.get_all_documents(limit=sample_count)
                summary['sample_documents'] = [
                    {
                        'id': doc.id,
                        'content_length': len(doc.content),
                        'content_preview': doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        'metadata': doc.metadata,
                    }
                    for doc in docs
                ]

            return summary

        except Exception as e:
            self.logger.error(f"Failed to generate memory summary: {str(e)}")
            return {
                'database_path': str(self.db_path),
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat(),
            }

    def search_and_analyze(
        self,
        query: str,
        n_results: int = 10,
    ) -> Dict[str, Any]:
        """
        Run a semantic search and bundle aggregate stats with the hits.

        Args:
            query: Search query
            n_results: Number of results to analyze

        Returns:
            Dict[str, Any]: Search results and analysis
        """
        try:
            docs = self.chroma_browser.search_documents(query=query, n_results=n_results)

            analysis: Dict[str, Any] = {
                'query': query,
                'total_results': len(docs),
                'analysis_timestamp': datetime.now().isoformat(),
                'results': [],
            }

            if docs:
                distances = [d.distance for d in docs if d.distance is not None]
                lengths = [len(d.content) for d in docs]

                if distances:
                    analysis['distance_stats'] = {
                        'min': min(distances),
                        'max': max(distances),
                        'average': sum(distances) / len(distances),
                    }

                analysis['content_stats'] = {
                    'min_length': min(lengths),
                    'max_length': max(lengths),
                    'average_length': sum(lengths) / len(lengths),
                }

                for d in docs:
                    analysis['results'].append({
                        'id': d.id,
                        'distance': d.distance,
                        'content_length': len(d.content),
                        'metadata': d.metadata,
                        'content_preview': d.content[:300] + "..." if len(d.content) > 300 else d.content,
                    })

            return analysis

        except Exception as e:
            self.logger.error(f"Search and analysis failed: {str(e)}")
            return {
                'query': query,
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat(),
            }

    def analyze_metadata_patterns(self) -> Dict[str, Any]:
        """
        Tally the value distribution of every metadata field.

        Returns:
            Dict[str, Any]: Metadata pattern analysis
        """
        try:
            documents = self.chroma_browser.get_all_documents()

            analysis: Dict[str, Any] = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_documents': len(documents),
                'metadata_analysis': {},
            }

            if not documents:
                return analysis

            field_stats: Dict[str, Dict[str, Any]] = {}

            for doc in documents:
                for key, value in doc.metadata.items():
                    bucket = field_stats.setdefault(
                        key,
                        {'values': {}, 'total_occurrences': 0, 'data_types': set()},
                    )

                    str_value = str(value)
                    bucket['values'][str_value] = bucket['values'].get(str_value, 0) + 1
                    bucket['total_occurrences'] += 1
                    bucket['data_types'].add(type(value).__name__)

            for field, data in field_stats.items():
                # Sets are not JSON-serialisable; freeze to a list.
                data['data_types'] = list(data['data_types'])

                unique_values = len(data['values'])
                top_value, top_count = max(data['values'].items(), key=lambda x: x[1])
                total_occ = data['total_occurrences']

                analysis['metadata_analysis'][field] = {
                    'unique_values': unique_values,
                    'total_occurrences': total_occ,
                    'data_types': data['data_types'],
                    'most_common_value': {
                        'value': top_value,
                        'count': top_count,
                        'percentage': (top_count / total_occ) * 100,
                    },
                    'coverage': (total_occ / len(documents)) * 100,
                }

                if unique_values > 5:
                    top_pairs = sorted(data['values'].items(), key=lambda x: x[1], reverse=True)[:5]
                    analysis['metadata_analysis'][field]['top_values'] = [
                        {'value': v, 'count': c} for v, c in top_pairs
                    ]

            return analysis

        except Exception as e:
            self.logger.error(f"Metadata analysis failed: {str(e)}")
            return {
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat(),
            }

    def export_analysis_report(
        self,
        output_path: str,
        include_search_examples: bool = True,
    ) -> None:
        """
        Build a comprehensive JSON report and write it to ``output_path``.

        Args:
            output_path: Path to save the report
            include_search_examples: Whether to include search examples
        """
        try:
            report: Dict[str, Any] = {
                'report_type': 'comprehensive_memory_analysis',
                'generated_at': datetime.now().isoformat(),
                'database_info': {
                    'path': str(self.db_path),
                    'collection': self.chroma_browser.collection_name,
                },
                'memory_summary': self.get_memory_summary(include_samples=True, sample_count=3),
                'metadata_patterns': self.analyze_metadata_patterns(),
            }

            if include_search_examples:
                seed_queries = [
                    "incident management",
                    "procedures",
                    "contact",
                    "emergency",
                    "escalation",
                ]

                gathered: Dict[str, Any] = {}
                for q in seed_queries:
                    try:
                        result = self.search_and_analyze(q, n_results=3)
                        if result.get('total_results', 0) > 0:
                            gathered[q] = result
                    except Exception as e:
                        self.logger.warning(f"Search example failed for '{q}': {str(e)}")

                if gathered:
                    report['search_examples'] = gathered

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)

            self.logger.info(f"Analysis report exported to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to export analysis report: {str(e)}")
            raise

    def quick_stats(self) -> str:
        """
        Render compact, multi-line statistics text for the active collection.

        Returns:
            str: Formatted statistics
        """
        try:
            stats = self.chroma_browser.get_collection_stats()

            lines = [
                f"📊 Memory Statistics",
                f"Collection: {stats.collection_name}",
                f"Documents: {stats.total_documents:,}",
                f"Metadata Fields: {len(stats.metadata_keys)}",
            ]

            if stats.content_stats:
                lines.append(f"Total Characters: {stats.content_stats.get('total_characters', 0):,}")
                lines.append(f"Average Length: {stats.content_stats.get('average_length', 0):.1f}")

            return "\n".join(lines)

        except Exception as e:
            return f"❌ Failed to get statistics: {str(e)}"
