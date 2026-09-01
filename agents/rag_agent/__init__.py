import os
import time
import logging
from typing import List, Optional, Dict, Any

from .doc_parser import MedicalDocParser
from .content_processor import ContentProcessor
from .vectorstore_qdrant import VectorStore
from .reranker import Reranker
from .query_expander import QueryExpander
from .query_rewriter import QueryRewriter
from .response_generator import ResponseGenerator

class MedicalRAG:
    """
    Medical Retrieval-Augmented Generation system that integrates all components.
    """
    def __init__(self, config):
        """
        Initialize the RAG Agent.
        
        Args:
            config: Configuration object with RAG settings
        """
        # Set up logging
        self.logger = logging.getLogger(f"{self.__module__}")
        self.logger.info("Initializing Medical RAG system")
        self.config = config
        self.doc_parser = MedicalDocParser()
        self.content_processor = ContentProcessor(config)
        self.vector_store = VectorStore(config)
        self.reranker = Reranker(config)
        self.query_expander = QueryExpander(config)
        self.query_rewriter = QueryRewriter(config)
        self.response_generator = ResponseGenerator(config)
        self.parsed_content_dir = self.config.rag.parsed_content_dir

        # 会话级语义缓存：{session_id: [{"query":..., "embedding":..., "response":..., "sources":..., "confidence":...}]}
        self._semantic_cache: Dict[str, List[Dict[str, Any]]] = {}
    
    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Ingest all files in a directory into the RAG system.
        
        Args:
            directory_path: Path to the directory containing files to ingest
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        self.logger.info(f"Ingesting files from directory: {directory_path}")
        
        try:
            # Check if directory exists
            if not os.path.isdir(directory_path):
                raise ValueError(f"Directory not found: {directory_path}")
            
            # Get all files in the directory
            files = [os.path.join(directory_path + '/', f) for f in os.listdir(directory_path) 
                     if os.path.isfile(os.path.join(directory_path, f))]
            
            if not files:
                self.logger.warning(f"No files found in directory: {directory_path}")
                return {
                    "success": True,
                    "documents_ingested": 0,
                    "chunks_processed": 0,
                    "processing_time": time.time() - start_time
                }
            
            # Track statistics
            total_chunks_processed = 0
            successful_ingestions = 0
            failed_ingestions = 0
            failed_files = []
            
            # Process each file
            for file_path in files:
                self.logger.info(f"Processing file {successful_ingestions + failed_ingestions + 1}/{len(files)}: {file_path}")
                
                try:
                    result = self.ingest_file(file_path)
                    if result["success"]:
                        successful_ingestions += 1
                        total_chunks_processed += result.get("chunks_processed", 0)
                    else:
                        failed_ingestions += 1
                        failed_files.append({"file": file_path, "error": result.get("error", "Unknown error")})
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {e}")
                    failed_ingestions += 1
                    failed_files.append({"file": file_path, "error": str(e)})
            
            return {
                "success": True,
                "documents_ingested": successful_ingestions,
                "failed_documents": failed_ingestions,
                "failed_files": failed_files,
                "chunks_processed": total_chunks_processed,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error ingesting directory: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def ingest_file(self, document_path: str) -> Dict[str, Any]:
        """
        Ingest a single file into the RAG system.
        
        Args:
            document_path: Path to the file to ingest
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        self.logger.info(f"Ingesting file: {document_path}")

        try:
            # Step 1: Parse document
            self.logger.info("1. Parsing document and extracting images...")
            #parsed_document, images = self.doc_parser.parse_document(document_path, self.parsed_content_dir)
            parsed_document, images = self.doc_parser.parse_document(
                document_path, self.parsed_content_dir, do_ocr=False
            )
            self.logger.info(f"   Parsed document and extracted {len(images)} images")

            # Step 2: Summarize images
            self.logger.info("2. Summarizing images...")
            #image_summaries = self.content_processor.summarize_images(images)
            image_summaries = []
            self.logger.info(f"   Generated {len(image_summaries)} image summaries")

            # Step 3: Format document with image summaries
            self.logger.info("3. Formatting document with image summaries...")
            formatted_document = self.content_processor.format_document_with_images(parsed_document, image_summaries)

            # Step 4: Chunk document into semantic sections
            self.logger.info("4. Chunking document into semantic sections...")
            document_chunks = self.content_processor.chunk_document(formatted_document)
            self.logger.info(f"   Document split into {len(document_chunks)} chunks")

            # Step 5: Create vector store and document store
            self.logger.info("5. Creating vector store knowledge base...")
            self.vector_store.create_vectorstore(
                document_chunks=document_chunks, 
                document_path=document_path
                )
            
            return {
                "success": True,
                "documents_ingested": 1,
                "chunks_processed": len(document_chunks),
                "processing_time": time.time() - start_time
            }
        
        except Exception as e:
            self.logger.error(f"Error ingesting file: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        
    def process_query(self, query: str, chat_history=None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a query with the RAG system.
        
        Args:
            query: The query string
            chat_history: Optional chat history (raw text) for multi-turn context
            session_id: Optional session id for per-session semantic cache

        Returns:
            Response dictionary with keys: response, sources, confidence, processing_time, cached
        """
        start_time = time.time()
        self.logger.info(f"RAG Agent processing query: {query}")
        
        # Process query and return result, passing chat_history
        try:
            # Step 0: Rewrite the multi-turn query into a standalone question
            search_query = query
            if self.config.rag.rewrite_query and chat_history:
                self.logger.info("0. Rewriting query with conversation history...")
                search_query = self.query_rewriter.rewrite(query, chat_history)
                self.logger.info(f"   Standalone query: '{search_query}'")

            # Step 0.5: Session-level semantic cache lookup (reuse results for similar questions)
            query_vector = None
            if self.config.rag.semantic_cache_enabled and session_id:
                try:
                    query_vector = self._embed_query(search_query)
                except Exception as exc:
                    self.logger.warning(f"   Query embedding failed, semantic cache disabled for this request: {exc}")
                if query_vector is not None:
                    cached_response = self._lookup_semantic_cache(session_id, query_vector)
                    if cached_response is not None:
                        cached_response["processing_time"] = time.time() - start_time
                        cached_response["cached"] = True
                        self.logger.info(f"   Semantic cache hit for session {session_id}")
                        return cached_response

            # Step 1: Expand query
            self.logger.info(f"1. Expanding query: '{search_query}'")
            expansion_result = self.query_expander.expand_query(search_query)
            expanded_query = expansion_result["expanded_query"]
            self.logger.info(f"   Original: '{search_query}'")
            self.logger.info(f"   Expanded: '{expanded_query}'")
            query = expanded_query

            # Step 2: Retrieval
            self.logger.info(f"2. Retrieving relevant documents for the query: '{query}'")
            vectorstore, docstore = self.vector_store.load_vectorstore()
            retrieved_documents = self.vector_store.retrieve_relevant_chunks(
                query=query,
                vectorstore=vectorstore,
                docstore=docstore,
                )

            self.logger.info(f"   Retrieved {len(retrieved_documents)} relevant document chunks")

            # Step 3: Rerank the retrieved documents if we have a reranker and enough documents
            self.logger.info(f"3. Reranking the retrieved documents")
            if self.reranker and len(retrieved_documents) > 1:
                reranked_documents, reranked_top_k_picture_paths = self.reranker.rerank(query, retrieved_documents, self.parsed_content_dir)
                self.logger.info(f"   Reranked retrieved documents and chose top {len(reranked_documents)}")
                self.logger.info(f"   Found {len(reranked_top_k_picture_paths)} referenced images")
            else:
                self.logger.info(f"   Could not rerank the retrieved documents, falling back to original scores")
                reranked_documents = retrieved_documents
                reranked_top_k_picture_paths = []

            # Step 4: Generate response
            self.logger.info("4. Generating response...")
            response = self.response_generator.generate_response(
                query=query,
                retrieved_docs=reranked_documents,
                picture_paths=reranked_top_k_picture_paths,
                chat_history=chat_history
                )
            
            # Store the successful response in the session-level semantic cache
            if self.config.rag.semantic_cache_enabled and session_id and response.get("response"):
                self._write_semantic_cache(session_id, search_query, query_vector, response)

            # Add timing information
            processing_time = time.time() - start_time
            response["processing_time"] = processing_time
            response["cached"] = False

            return response
        
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Return error response
            return {
                "response": f"I encountered an error while processing your query: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "processing_time": time.time() - start_time
            }

    def _embed_query(self, text: str) -> List[float]:
        """Embed a query string into a vector using the configured embedding model."""
        embedding_model = getattr(self.config.rag, "embedding_model", None)
        if embedding_model is None:
            raise ValueError("embedding_model is not configured")
        embeddings = embedding_model.embed_query(text)
        if not embeddings:
            raise ValueError("embedding model returned an empty vector")
        return list(embeddings)

    def _cosine_similarity(self, vector_a: List[float], vector_b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        try:
            import numpy as np
        except ImportError:
            return 0.0
        array_a = np.asarray(vector_a, dtype=np.float32)
        array_b = np.asarray(vector_b, dtype=np.float32)
        norm_a = float(np.linalg.norm(array_a))
        norm_b = float(np.linalg.norm(array_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(array_a, array_b) / (norm_a * norm_b))

    def _lookup_semantic_cache(self, session_id: str, query_vector: List[float]) -> Optional[Dict[str, Any]]:
        """Return a cached response if a semantically similar query exists for the session."""
        cached_entries = self._semantic_cache.get(session_id, [])
        if not cached_entries:
            return None
        best_entry, best_score = None, 0.0
        for entry in cached_entries:
            score = self._cosine_similarity(query_vector, entry.get("embedding", []))
            if score > best_score:
                best_score, best_entry = score, entry
        threshold = self.config.rag.semantic_cache_threshold
        if best_entry is not None and best_score >= threshold:
            self.logger.info(f"   Cache similarity: {best_score:.4f} (threshold {threshold})")
            return {
                "response": best_entry["response"],
                "sources": list(best_entry.get("sources", [])),
                "confidence": best_entry.get("confidence", 0.0),
            }
        return None

    def _write_semantic_cache(self, session_id: str, query: str, query_vector: Optional[List[float]], response: Dict[str, Any]) -> None:
        """Store a generated response in the session-level semantic cache (FIFO eviction)."""
        if query_vector is None:
            return
        entries = self._semantic_cache.setdefault(session_id, [])
        capacity = self.config.rag.semantic_cache_capacity
        entries.append({
            "query": query,
            "embedding": list(query_vector),
            "response": response.get("response", ""),
            "sources": list(response.get("sources", [])),
            "confidence": response.get("confidence", 0.0),
        })
        while len(entries) > capacity:
            entries.pop(0)
