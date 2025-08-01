import asyncio
import sys
import os
sys.path.append('.')

from app.services.vector_store import VectorStore
from app.config import settings

async def comprehensive_debug():
    """Comprehensive debug of vector store and documents"""
    
    print("🔍 COMPREHENSIVE VECTOR STORE DEBUG")
    print("=" * 50)
    
    vector_store = VectorStore()
    case_id = "Test"  # Change to your case ID
    
    try:
        # 1. Check if collection exists
        collection_name = f"case_{case_id}"
        print(f"📂 Checking collection: {collection_name}")
        
        try:
            collection = vector_store.client.get_collection(collection_name)
            print(f"✅ Collection exists")
        except Exception as e:
            print(f"❌ Collection doesn't exist: {e}")
            print("💡 This means no files have been properly processed yet")
            return
        
        # 2. Get all documents in collection
        all_docs = collection.get()
        total_docs = len(all_docs['ids'])
        print(f"📊 Total documents in collection: {total_docs}")
        
        if total_docs == 0:
            print("❌ No documents found in vector store!")
            print("💡 Files were uploaded but not processed into vector store")
            return
        
        # 3. Analyze document types
        print(f"\n📄 Document Analysis:")
        doc_types = {}
        full_docs = 0
        
        for i, (doc_id, metadata) in enumerate(zip(all_docs['ids'], all_docs['metadatas'])):
            if metadata.get('type') == 'full_document':
                full_docs += 1
            else:
                file_type = metadata.get('file_type', 'unknown')
                doc_types[file_type] = doc_types.get(file_type, 0) + 1
        
        print(f"   📈 Searchable documents: {total_docs - full_docs}")
        print(f"   🗂️  Full document copies: {full_docs}")
        print(f"   📋 Document types: {doc_types}")
        
        # 4. Show sample documents
        print(f"\n🔍 Sample Documents (first 5):")
        for i in range(min(5, len(all_docs['ids']))):
            doc_id = all_docs['ids'][i]
            metadata = all_docs['metadatas'][i]
            content_preview = all_docs['documents'][i][:100] + "..." if len(all_docs['documents'][i]) > 100 else all_docs['documents'][i]
            
            print(f"   {i+1}. ID: {doc_id}")
            print(f"      Type: {metadata.get('file_type', 'unknown')}")
            print(f"      File: {metadata.get('filename', 'unknown')}")
            print(f"      Content: {content_preview}")
            print()
        
        # 5. Test search functionality
        print(f"🔎 Testing Search Functionality:")
        test_queries = ["phone", "report", "Bunner", "LG"]
        
        for query in test_queries:
            print(f"\n   Testing query: '{query}'")
            try:
                results = await vector_store.search_documents(case_id, query, limit=3)
                print(f"   ✅ Found {len(results)} results")
                
                for j, result in enumerate(results):
                    score = result.get('similarity_score', 0)
                    filename = result.get('metadata', {}).get('filename', 'unknown')
                    print(f"      {j+1}. {filename} (score: {score:.3f})")
                    
            except Exception as e:
                print(f"   ❌ Search failed: {e}")
        
        # 6. Check specific file
        print(f"\n🎯 Looking for specific file: 'Bunner report re_ LG phone on Sept 15, 2017.docx'")
        bunner_found = False
        for i, metadata in enumerate(all_docs['metadatas']):
            filename = metadata.get('filename', '')
            if 'Bunner' in filename and 'LG phone' in filename:
                print(f"   ✅ Found: {filename}")
                print(f"   📄 Document ID: {all_docs['ids'][i]}")
                print(f"   🏷️  File type: {metadata.get('file_type', 'unknown')}")
                bunner_found = True
                break
        
        if not bunner_found:
            print(f"   ❌ Bunner report not found in vector store")
            print(f"   💡 Check if file was processed correctly")
    
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()

# Also check file system
def check_file_system():
    """Check what files exist on disk"""
    print(f"\n📁 FILE SYSTEM CHECK")
    print("=" * 30)
    
    case_dir = "./discovery_sets/Test"  # Change to your case ID
    
    if not os.path.exists(case_dir):
        print(f"❌ Case directory doesn't exist: {case_dir}")
        return
    
    total_files = 0
    for root, dirs, files in os.walk(case_dir):
        if files:
            rel_path = os.path.relpath(root, case_dir)
            print(f"📂 {rel_path}: {len(files)} files")
            for file in files[:5]:  # Show first 5 files
                print(f"   - {file}")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more")
            total_files += len(files)
    
    print(f"📊 Total files on disk: {total_files}")

if __name__ == "__main__":
    check_file_system()
    asyncio.run(comprehensive_debug())