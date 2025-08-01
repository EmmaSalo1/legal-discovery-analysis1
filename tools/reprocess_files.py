import asyncio
import sys
import os
sys.path.append('.')

from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore
from app.config import settings

async def reprocess_all_files():
    """Reprocess all files in a case to fix vector store issues"""
    
    case_id = "Test"  # Change to your case ID
    case_dir = f"./discovery_sets/{case_id}"
    
    if not os.path.exists(case_dir):
        print(f"❌ Case directory doesn't exist: {case_dir}")
        return
    
    print(f"🔄 REPROCESSING FILES for case: {case_id}")
    print("=" * 50)
    
    # Initialize processors
    document_processor = DocumentProcessor()
    vector_store = VectorStore()
    
    # Clear existing vector store for this case (optional)
    print("🗑️  Clearing existing vector store...")
    try:
        collection_name = f"case_{case_id}"
        collection = vector_store.client.get_collection(collection_name)
        vector_store.client.delete_collection(collection_name)
        print("✅ Cleared existing collection")
    except:
        print("ℹ️  No existing collection to clear")
    
    # Find all files
    all_files = []
    for root, dirs, files in os.walk(case_dir):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    
    print(f"📁 Found {len(all_files)} files to process")
    
    # Process each file
    processed = 0
    failed = 0
    
    for i, file_path in enumerate(all_files):
        filename = os.path.basename(file_path)
        print(f"\n📄 Processing {i+1}/{len(all_files)}: {filename}")
        
        try:
            # Process the document
            result = await document_processor.process_document(file_path, case_id)
            
            if result.get('processing_status') == 'completed':
                # Add to vector store
                doc_id = await vector_store.add_document(result, case_id)
                print(f"   ✅ Successfully processed and indexed")
                processed += 1
            else:
                print(f"   ⚠️  Processing completed with issues")
                print(f"   📝 Error: {result.get('error', 'Unknown error')}")
                failed += 1
                
        except Exception as e:
            print(f"   ❌ Failed to process: {str(e)}")
            failed += 1
    
    print(f"\n📊 PROCESSING COMPLETE")
    print(f"✅ Successfully processed: {processed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success rate: {processed/(processed+failed)*100:.1f}%")
    
    # Test search after reprocessing
    print(f"\n🔎 Testing search after reprocessing...")
    try:
        results = await vector_store.search_documents(case_id, "phone", limit=5)
        print(f"✅ Search test successful: {len(results)} results found")
        
        if results:
            print("📋 Sample results:")
            for i, result in enumerate(results[:3]):
                filename = result.get('metadata', {}).get('filename', 'unknown')
                score = result.get('similarity_score', 0)
                print(f"   {i+1}. {filename} (score: {score:.3f})")
    except Exception as e:
        print(f"❌ Search test failed: {e}")

if __name__ == "__main__":
    asyncio.run(reprocess_all_files())