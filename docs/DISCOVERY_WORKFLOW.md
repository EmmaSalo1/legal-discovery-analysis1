# Discovery Review Workflow

## Overview
This document outlines the standard workflow for reviewing opposing counsel discovery productions using the Legal Discovery Analysis System.

## Workflow Steps

### 1. Case Setup
1. Create new case in the system
2. Set up case metadata (parties, court, case type)
3. Define document categories and privilege protocols

### 2. Document Import
1. Receive discovery production from opposing counsel
2. Organize documents by production set
3. Bulk import using provided scripts
4. Verify Bates numbering and metadata

### 3. Initial Processing
1. System extracts text content from all documents
2. Documents are categorized by type (email, deposition, contract, etc.)
3. Initial privilege screening is performed
4. Documents are indexed for search and analysis

### 4. Review and Analysis
1. **Privilege Review**: Review flagged documents for privilege issues
2. **Content Analysis**: Use AI chat to explore document contents
3. **Contradiction Detection**: Run automated contradiction analysis
4. **Timeline Construction**: Build chronological timeline of events
5. **Evidence Identification**: Flag key evidence and admissions

### 5. Strategic Analysis
1. Identify case strengths and weaknesses
2. Develop examination strategies based on contradictions
3. Prepare evidence summaries for trial team
4. Generate reports for case strategy meetings

### 6. Ongoing Management
1. Monitor for new         self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collections = {}

    def _get_collection(self, case_id: str):
        """Get or create a collection for a specific case"""
        collection_name = f"case_{case_id}"
        
        if collection_name not in self.collections:
            try:
                self.collections[collection_name] = self.client.get_collection(name=collection_name)
            except:
                self.collections[collection_name] = self.client.create_collection(
                    name=collection_name,
                    metadata={"case_id": case_id, "created_at": str(datetime.now())}
                )
        
        return self.collections[collection_name]

    async def add_document(self, document_data: Dict[str, Any], case_id: str) -> str:
        """Add a document to the vector store"""
        try:
            collection = self._get_collection(case_id)
            
            # Prepare document chunks for better retrieval
            chunks = self._chunk_document(document_data["content"])
            
            doc_ids = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_data['id']}_chunk_{i}"
                
                # Prepare metadata
                metadata = {
                    "document_id": document_data["id"],
                    "filename": document_data["filename"],
                    "case_id": case_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_path": document_data["file_path"],
                    "created_at": str(document_data["created_at"]),
                    **document_data.get("metadata", {})
                }
                
                # Convert non-string metadata values to strings
                for key, value in metadata.items():
                    if not isinstance(value, (str, int, float, bool)):
                        metadata[key] = json.dumps(value)
                
                collection.add(
                    documents=[chunk],
                    metadatas=[metadata],
                    ids=[chunk_id]
                )
                
                doc_ids.append(chunk_id)
            
            logger.info(f"Added document {document_data['filename']} with {len(chunks)} chunks to case {case_id}")
            return document_data["id"]
            
        except Exception as e:
            logger.error(f"Error adding document to vector store: {e}")
            raise

    def _chunk_document(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split document into overlapping chunks"""
        if not content:
            return [""]
        
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks if chunks else [content]

    async def search_documents(self, case_id: str, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search documents in a case"""
        try:
            collection = self._get_collection(case_id)
            
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            search_results = []
            for i in range(len(results["documents"][0])):
                search_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    async def get_case_documents(self, case_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a case"""
        try:
            collection = self._get_collection(case_id)
            
            # Get all documents (limiting to 1000 for performance)
            results = collection.get(limit=1000)
            
            # Group by document_id to avoid duplicates from chunks
            documents = {}
            for i, doc_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                document_id = metadata.get("document_id", doc_id)
                
                if document_id not in documents:
                    documents[document_id] = {
                        "id": document_id,
                        "filename": metadata.get("filename", "Unknown"),
                        "case_id": case_id,
                        "created_at": metadata.get("created_at"),
                        "metadata": metadata
                    }
            
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"Error getting case documents: {e}")
            return []