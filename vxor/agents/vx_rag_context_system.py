#!/usr/bin/env python3
"""
VX-RAG-CONTEXT-SYSTEM
====================

Retrieval-Augmented Generation system for embedding-based context selection.
Implements CTO-recommended approach for Issue + relevant commits + tests.

Features:
- Embedding-based context selection (→ ~2x Accuracy Boost)
- Semantic similarity search for relevant code
- Integration with VX-PATCH-REPAIR-SYSTEM
- Few-shot learning with successful patch templates
"""

import json
import logging
import numpy as np
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import hashlib
import sqlite3
from datetime import datetime

# Embedding imports (fallback to simple implementations if not available)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class CodeContext:
    """Context object for code snippets with embeddings."""
    id: str
    content: str
    file_path: str
    function_name: Optional[str]
    class_name: Optional[str]
    commit_hash: Optional[str]
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PatchTemplate:
    """Template for successful patches with context."""
    id: str
    issue_type: str
    original_code: str
    patched_code: str
    success_metrics: Dict[str, float]
    embedding: Optional[np.ndarray] = None
    usage_count: int = 0
    last_used: Optional[datetime] = None


class EmbeddingEngine:
    """Handles text embeddings for semantic similarity."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger("VX-RAG.EmbeddingEngine")
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Default for MiniLM
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize embedding model with fallbacks."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.logger.info(f"Initialized SentenceTransformer: {self.model_name}")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load SentenceTransformer: {e}")
        
        # Fallback to simple hash-based embeddings
        self.logger.info("Using fallback hash-based embeddings")
        self.embedding_dim = 256
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) to embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model is not None:
            return self.model.encode(texts)
        else:
            # Fallback: hash-based embeddings
            embeddings = []
            for text in texts:
                # Simple hash-based embedding
                hash_obj = hashlib.sha256(text.encode())
                hash_bytes = hash_obj.digest()
                # Convert to float array and normalize
                embedding = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
                embedding = embedding[:self.embedding_dim]  # Truncate to desired size
                if len(embedding) < self.embedding_dim:
                    # Pad with zeros if needed
                    embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
                # Normalize
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            
            return np.array(embeddings)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


class VectorStore:
    """Vector database for storing and retrieving embeddings."""
    
    def __init__(self, db_path: str = "vx_rag_vectors.db", use_faiss: bool = True):
        self.logger = logging.getLogger("VX-RAG.VectorStore")
        self.db_path = Path(db_path)
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.embedding_dim = 384
        
        # Initialize storage
        self._init_sqlite()
        if self.use_faiss:
            self._init_faiss()
        
        self.contexts = {}  # In-memory cache
        self.templates = {}  # Patch templates cache
    
    def _init_sqlite(self):
        """Initialize SQLite database for metadata."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS code_contexts (
                id TEXT PRIMARY KEY,
                content TEXT,
                file_path TEXT,
                function_name TEXT,
                class_name TEXT,
                commit_hash TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS patch_templates (
                id TEXT PRIMARY KEY,
                issue_type TEXT,
                original_code TEXT,
                patched_code TEXT,
                success_metrics TEXT,
                usage_count INTEGER DEFAULT 0,
                last_used TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def _init_faiss(self):
        """Initialize FAISS index for fast similarity search."""
        try:
            import faiss
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            self.faiss_ids = []
            self.logger.info("Initialized FAISS index")
        except Exception as e:
            self.logger.warning(f"Failed to initialize FAISS: {e}")
            self.use_faiss = False
    
    def add_context(self, context: CodeContext):
        """Add code context to vector store."""
        # Store in SQLite
        self.conn.execute('''
            INSERT OR REPLACE INTO code_contexts 
            (id, content, file_path, function_name, class_name, commit_hash, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            context.id,
            context.content,
            context.file_path,
            context.function_name,
            context.class_name,
            context.commit_hash,
            json.dumps(context.metadata)
        ))
        self.conn.commit()
        
        # Add to FAISS index if available
        if self.use_faiss and context.embedding is not None:
            # Normalize embedding for cosine similarity
            normalized_embedding = context.embedding / np.linalg.norm(context.embedding)
            self.faiss_index.add(normalized_embedding.reshape(1, -1))
            self.faiss_ids.append(context.id)
        
        # Cache in memory
        self.contexts[context.id] = context
        
        self.logger.debug(f"Added context: {context.id}")
    
    def add_template(self, template: PatchTemplate):
        """Add patch template to store."""
        self.conn.execute('''
            INSERT OR REPLACE INTO patch_templates
            (id, issue_type, original_code, patched_code, success_metrics, usage_count, last_used)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            template.id,
            template.issue_type,
            template.original_code,
            template.patched_code,
            json.dumps(template.success_metrics),
            template.usage_count,
            template.last_used.isoformat() if template.last_used else None
        ))
        self.conn.commit()
        
        self.templates[template.id] = template
        self.logger.debug(f"Added template: {template.id}")
    
    def search_similar_contexts(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[CodeContext, float]]:
        """Search for similar code contexts."""
        if self.use_faiss and len(self.faiss_ids) > 0:
            return self._faiss_search(query_embedding, top_k)
        else:
            return self._linear_search(query_embedding, top_k)
    
    def _faiss_search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[CodeContext, float]]:
        """Search using FAISS index."""
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        scores, indices = self.faiss_index.search(normalized_query.reshape(1, -1), min(top_k, len(self.faiss_ids)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.faiss_ids):
                context_id = self.faiss_ids[idx]
                context = self.contexts.get(context_id)
                if context:
                    results.append((context, float(score)))
        
        return results
    
    def _linear_search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[CodeContext, float]]:
        """Fallback linear search."""
        similarities = []
        
        for context in self.contexts.values():
            if context.embedding is not None:
                similarity = np.dot(query_embedding, context.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(context.embedding)
                )
                similarities.append((context, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_patch_templates(self, issue_type: str = None, top_k: int = 10) -> List[PatchTemplate]:
        """Get relevant patch templates."""
        query = "SELECT * FROM patch_templates"
        params = []
        
        if issue_type:
            query += " WHERE issue_type = ?"
            params.append(issue_type)
        
        query += " ORDER BY usage_count DESC, created_at DESC LIMIT ?"
        params.append(top_k)
        
        cursor = self.conn.execute(query, params)
        templates = []
        
        for row in cursor.fetchall():
            template = PatchTemplate(
                id=row[0],
                issue_type=row[1],
                original_code=row[2],
                patched_code=row[3],
                success_metrics=json.loads(row[4]) if row[4] else {},
                usage_count=row[5],
                last_used=datetime.fromisoformat(row[6]) if row[6] else None
            )
            templates.append(template)
        
        return templates


class RAGContextSelector:
    """Main RAG system for context selection."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("VX-RAG.ContextSelector")
        
        # Initialize components
        self.embedding_engine = EmbeddingEngine(
            model_name=self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        self.vector_store = VectorStore(
            db_path=self.config.get('db_path', 'vx_rag_vectors.db'),
            use_faiss=self.config.get('use_faiss', True)
        )
        
        # Performance tracking
        self.search_history = []
        
    def index_codebase(self, codebase_path: str):
        """Index entire codebase for RAG."""
        start_time = time.time()
        indexed_count = 0
        
        codebase = Path(codebase_path)
        
        for py_file in codebase.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create context for entire file
                file_context = CodeContext(
                    id=f"file_{py_file.stem}_{hash(str(py_file))}",
                    content=content,
                    file_path=str(py_file.relative_to(codebase)),
                    function_name=None,
                    class_name=None,
                    commit_hash=None,
                    metadata={'type': 'file', 'size': len(content)}
                )
                
                # Generate embedding
                file_context.embedding = self.embedding_engine.encode(content)[0]
                
                # Add to vector store
                self.vector_store.add_context(file_context)
                indexed_count += 1
                
                # Also index individual functions and classes
                indexed_count += self._index_code_elements(content, py_file, codebase)
                
            except Exception as e:
                self.logger.warning(f"Failed to index {py_file}: {e}")
        
        duration = time.time() - start_time
        self.logger.info(f"Indexed {indexed_count} code elements in {duration:.2f}s")
    
    def _index_code_elements(self, content: str, file_path: Path, codebase: Path) -> int:
        """Index individual functions and classes."""
        indexed_count = 0
        
        try:
            import ast
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_content = ast.get_source_segment(content, node)
                    if func_content:
                        context = CodeContext(
                            id=f"func_{node.name}_{hash(func_content)}",
                            content=func_content,
                            file_path=str(file_path.relative_to(codebase)),
                            function_name=node.name,
                            class_name=None,
                            commit_hash=None,
                            metadata={'type': 'function', 'line': node.lineno}
                        )
                        context.embedding = self.embedding_engine.encode(func_content)[0]
                        self.vector_store.add_context(context)
                        indexed_count += 1
                
                elif isinstance(node, ast.ClassDef):
                    class_content = ast.get_source_segment(content, node)
                    if class_content:
                        context = CodeContext(
                            id=f"class_{node.name}_{hash(class_content)}",
                            content=class_content,
                            file_path=str(file_path.relative_to(codebase)),
                            function_name=None,
                            class_name=node.name,
                            commit_hash=None,
                            metadata={'type': 'class', 'line': node.lineno}
                        )
                        context.embedding = self.embedding_engine.encode(class_content)[0]
                        self.vector_store.add_context(context)
                        indexed_count += 1
        
        except Exception as e:
            self.logger.warning(f"Failed to parse AST for {file_path}: {e}")
        
        return indexed_count
    
    def select_relevant_context(self, issue_description: str, top_k: int = 5) -> List[Tuple[CodeContext, float]]:
        """Select relevant context for an issue using RAG."""
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedding_engine.encode(issue_description)[0]
        
        # Search for similar contexts
        similar_contexts = self.vector_store.search_similar_contexts(query_embedding, top_k)
        
        # Track performance
        search_time = time.time() - start_time
        self.search_history.append({
            'query_length': len(issue_description),
            'results_count': len(similar_contexts),
            'search_time': search_time,
            'timestamp': time.time()
        })
        
        self.logger.info(f"Found {len(similar_contexts)} relevant contexts in {search_time:.3f}s")
        return similar_contexts
    
    def get_few_shot_templates(self, issue_type: str, top_k: int = 3) -> List[PatchTemplate]:
        """Get few-shot learning templates for issue type."""
        templates = self.vector_store.get_patch_templates(issue_type, top_k)
        
        # Update usage statistics
        for template in templates:
            template.usage_count += 1
            template.last_used = datetime.now()
            self.vector_store.add_template(template)  # Update in store
        
        return templates
    
    def add_successful_patch(self, issue_type: str, original_code: str, patched_code: str, 
                           success_metrics: Dict[str, float]):
        """Add successful patch as template for future use."""
        template = PatchTemplate(
            id=f"patch_{issue_type}_{hash(patched_code)}",
            issue_type=issue_type,
            original_code=original_code,
            patched_code=patched_code,
            success_metrics=success_metrics,
            usage_count=0,
            last_used=None
        )
        
        # Generate embedding for template
        template_text = f"{issue_type} {original_code} {patched_code}"
        template.embedding = self.embedding_engine.encode(template_text)[0]
        
        self.vector_store.add_template(template)
        self.logger.info(f"Added successful patch template: {template.id}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get RAG system performance metrics."""
        if not self.search_history:
            return {}
        
        recent_searches = self.search_history[-100:]  # Last 100 searches
        
        return {
            'total_searches': len(self.search_history),
            'average_search_time': sum(s['search_time'] for s in recent_searches) / len(recent_searches),
            'average_results_count': sum(s['results_count'] for s in recent_searches) / len(recent_searches),
            'contexts_indexed': len(self.vector_store.contexts),
            'templates_available': len(self.vector_store.templates)
        }


# Integration with VX-PATCH-REPAIR-SYSTEM
class EnhancedIssueReaderAgent:
    """Enhanced Issue Reader with RAG context selection."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("VX-RAG.EnhancedIssueReader")
        
        # Initialize RAG system
        self.rag_system = RAGContextSelector(config.get('rag_config', {}))
        
        # Initialize base issue reader
        from vx_patch_repair_system import IssueReaderAgent
        self.base_reader = IssueReaderAgent(config)
    
    def process_with_rag(self, issue_data: Dict[str, Any]) -> Tuple[Any, List[Tuple[CodeContext, float]]]:
        """Process issue with RAG-enhanced context selection."""
        # Get base issue context
        issue_context = self.base_reader.process(issue_data)
        
        # Select relevant context using RAG
        query_text = f"{issue_context.title} {issue_context.description}"
        relevant_contexts = self.rag_system.select_relevant_context(
            query_text, 
            top_k=self.config.get('context_top_k', 5)
        )
        
        # Enhance issue context with RAG results
        issue_context.metadata = issue_context.metadata or {}
        issue_context.metadata['rag_contexts'] = [
            {
                'id': ctx.id,
                'file_path': ctx.file_path,
                'similarity': score,
                'type': ctx.metadata.get('type', 'unknown')
            }
            for ctx, score in relevant_contexts
        ]
        
        return issue_context, relevant_contexts


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize RAG system
    rag_config = {
        'embedding_model': 'all-MiniLM-L6-v2',
        'db_path': 'vx_rag_test.db',
        'use_faiss': True
    }
    
    rag_system = RAGContextSelector(rag_config)
    
    # Index a small codebase (current directory)
    rag_system.index_codebase(".")
    
    # Test context selection
    test_issue = "NoneType object has no attribute 'strip' error in data processing"
    contexts = rag_system.select_relevant_context(test_issue, top_k=3)
    
    print(f"Found {len(contexts)} relevant contexts:")
    for ctx, score in contexts:
        print(f"- {ctx.file_path}:{ctx.function_name or ctx.class_name or 'file'} (similarity: {score:.3f})")
    
    # Show performance metrics
    metrics = rag_system.get_performance_metrics()
    print(f"RAG Performance: {metrics}")
