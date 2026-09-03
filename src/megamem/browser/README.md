# megamem Browser

Professional interactive browser for exploring megamem stores and ChromaDB collections.

## Overview

The megamem browser provides comprehensive tools for exploring, analyzing, and managing memory data stored in ChromaDB collections. It offers both interactive and programmatic interfaces for memory analysis.

## Features

### 🔍 **Interactive Browser**
- Real-time exploration of ChromaDB collections
- Semantic search with similarity scoring
- Metadata filtering and analysis
- Document detail views with full content
- Export capabilities (JSON, CSV, TXT)
- Collection switching and management

### 📊 **Memory Viewer**
- Programmatic memory analysis
- Comprehensive statistics generation
- Metadata pattern analysis
- Search result analysis
- Report generation and export

### 🛠️ **ChromaDB Integration**
- Direct ChromaDB access without intermediate layers
- Support for multiple collections
- Persistent client connections
- Efficient data retrieval and caching

## Installation

The browser is included with the megamem package. Ensure you have ChromaDB installed:

```bash
pip install chromadb
```

## Usage

### Command Line Interface

#### Interactive Browser (Default)
```bash
# Browse memory store
python -m megamem.browser /path/to/memory_store

# Browse specific collection
python -m megamem.browser /path/to/memory_store -c megamem

# Verbose output
python -m megamem.browser /path/to/memory_store --verbose
```

#### Quick Statistics
```bash
# Show quick stats
python -m megamem.browser /path/to/memory_store --stats
```

#### Search and Analysis
```bash
# Search with results
python -m megamem.browser /path/to/memory_store --search "incident management"

# Search with export
python -m megamem.browser /path/to/memory_store --search "procedures" --output results.json
```

#### Comprehensive Analysis
```bash
# Generate analysis report
python -m megamem.browser /path/to/memory_store --analyze --output report.json
```

### Interactive Commands

Once in the interactive browser:

#### Basic Navigation
```
help                  - Show available commands
collections           - List all collections
switch <name>         - Switch to different collection
stats                 - Show detailed statistics
count                 - Count total documents
```

#### Document Browsing
```
list [limit]          - List documents (default: 10)
show <id|index>       - Show detailed document view
```

#### Search and Filtering
```
search <query>        - Semantic search
filter <key>=<value>  - Filter by metadata
clear                 - Clear current filters
```

#### Analysis and Export
```
metadata [key]        - Show metadata information
export <format> <path> - Export documents (json/csv/txt)
```

### Programmatic Usage

#### Memory Viewer
```python
from megamem.browser import MemoryViewer

# Initialize viewer
viewer = MemoryViewer(db_path="/path/to/memory_store")

# Get quick statistics
stats = viewer.quick_stats()
print(stats)

# Generate comprehensive summary
summary = viewer.get_memory_summary(include_samples=True)

# Search and analyze
results = viewer.search_and_analyze("incident management", n_results=10)

# Export analysis report
viewer.export_analysis_report("analysis_report.json")
```

#### ChromaDB Browser
```python
from megamem.browser import ChromaBrowser

# Initialize browser
browser = ChromaBrowser(db_path="/path/to/memory_store", collection_name="megamem")

# Get all documents
documents = browser.get_all_documents(limit=100)

# Search documents
results = browser.search_documents("emergency procedures", n_results=5)

# Filter by metadata
filtered = browser.filter_documents({"source_file": "handbook.md"})

# Get statistics
stats = browser.get_collection_stats()

# Export documents
browser.export_documents(documents, "export.json", format="json")
```

#### Interactive Browser
```python
from megamem.browser import InteractiveMemoryBrowser

# Launch interactive session
browser = InteractiveMemoryBrowser(db_path="/path/to/memory_store")
browser.run()
```

## Data Structures

### ChromaDocument
```python
@dataclass
class ChromaDocument:
    id: str
    content: str
    metadata: Dict[str, Any]
    embeddings: Optional[List[float]] = None
    distance: Optional[float] = None
```

### ChromaStats
```python
@dataclass
class ChromaStats:
    total_documents: int
    collection_name: str
    metadata_keys: List[str]
    unique_metadata_values: Dict[str, int]
    content_stats: Dict[str, Any]
```

## Output Formats

### JSON Export
```json
{
  "export_timestamp": "2025-10-13T10:30:45",
  "collection_name": "megamem",
  "total_documents": 156,
  "documents": [
    {
      "id": "doc_12345",
      "content": "Document content...",
      "metadata": {
        "source_file": "handbook.md",
        "chunk_index": 3
      }
    }
  ]
}
```

### Analysis Report
```json
{
  "report_type": "comprehensive_memory_analysis",
  "generated_at": "2025-10-13T10:30:45",
  "memory_summary": {
    "statistics": {
      "total_documents": 156,
      "content_stats": {
        "total_characters": 45123,
        "average_length": 289.25
      }
    }
  },
  "metadata_patterns": {
    "source_file": {
      "unique_values": 8,
      "most_common_value": {
        "value": "handbook.md",
        "count": 89,
        "percentage": 57.1
      }
    }
  }
}
```

## Examples

### Browse VOM Memory Store
```bash
# Browse VOM system memory
python -m megamem.browser ./vom_memory_store

💭 Memory Browser > collections
💭 Memory Browser > list 5
Memory Browser > search "service outage"
💭 Memory Browser > show 1
💭 Memory Browser > export json ./documents.json
```

### Analyze Memory Patterns
```bash
# Generate comprehensive analysis
python -m megamem.browser ./memory_store --analyze --output analysis.json

# Search specific topics
python -m megamem.browser ./memory_store --search "incident procedures" --limit 20
```

### Filter and Export
```bash
# Interactive filtering
python -m megamem.browser ./memory_store

💭 Memory Browser > filter source_file=handbook.md
💭 Memory Browser > list 10
💭 Memory Browser > export csv ./handbook_docs.csv
```

## Error Handling

The browser includes comprehensive error handling:

- **Database Not Found**: Clear error message with path validation
- **Collection Errors**: Graceful handling of missing/invalid collections
- **Search Failures**: Fallback behavior for failed semantic searches
- **Export Errors**: Directory creation and permission validation
- **Connection Issues**: Retry logic and connection status reporting

## Performance Considerations

- **Caching**: Document caching for improved performance
- **Lazy Loading**: Documents loaded on-demand
- **Batch Operations**: Efficient bulk document retrieval
- **Memory Management**: Controlled memory usage for large collections

## Development

### Testing
```bash
# Run test suite
python test_browser.py

# Test with specific memory store
python -m megamem.browser /path/to/test/store --stats
```

### Adding New Features
1. Extend `ChromaBrowser` for new ChromaDB operations
2. Add commands to `InteractiveMemoryBrowser` for new interactive features
3. Update `MemoryViewer` for new analysis capabilities
4. Add CLI options in `__main__.py` for new command-line features

## Integration

### With VOM System
```python
# In VOM interactive mode
from megamem.browser import InteractiveMemoryBrowser

# Launch browser for VOM memory store
browser = InteractiveMemoryBrowser(db_path="./vom_memory_store")
browser.run()
```

### With Other megamem Components
```python
from megamem.browser import MemoryViewer
from megamem.client import MemoryClient

# Analyze memory after building
client = MemoryClient(config)
# ... build memory ...

# Analyze results
viewer = MemoryViewer(db_path=config['persist_path'])
summary = viewer.get_memory_summary()
```

## Troubleshooting

### Common Issues

1. **ChromaDB Not Found**
   ```bash
   pip install chromadb
   ```

2. **Collection Not Found**
   - Check collection name spelling
   - Use `collections` command to list available collections

3. **Permission Errors**
   - Ensure read access to database directory
   - Check write permissions for export operations

4. **Large Collections**
   - Use `limit` parameter to reduce memory usage
   - Consider filtering before export operations

### Debug Mode
```bash
python -m megamem.browser /path/to/store --verbose
```

This provides detailed logging for troubleshooting connection and operation issues.

---

The megamem browser provides a comprehensive, professional interface for exploring and analyzing memory data, making it easy to understand and debug memory system behavior.
