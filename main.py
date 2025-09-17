import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import logging
from fastapi.responses import JSONResponse, RedirectResponse
from App.create_vector_db import ImageIndexer 
from App.semantic_image_search import SemanticImageSearch  

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("VectorDBAPI")

app = FastAPI(title="Semantic Image Search API")

# Request models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

# Startup Event
@app.on_event("startup")
async def startup_event():
    """
    Build vector DB on startup only if missing and initialize searcher.
    """
    global searcher
    index_path = os.path.join("Data/VectorDB", "image_index.faiss")
    metadata_path = os.path.join("Data/VectorDB", "metadata.json")
    image_folder = "Data/Images"

    # Check if index and metadata exist
    if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
        logger.info("Index or metadata missing. Running ImageIndexer...")
        indexer = ImageIndexer(image_folder=image_folder, output_dir="Data/VectorDB")
        await indexer.build_index()
    else:
        logger.info("Index and metadata already exist. Skipping indexing.")

    # Initialize SemanticImageSearch
    searcher = SemanticImageSearch(
        index_path=index_path,
        metadata_path=metadata_path,
        image_folder=image_folder
    )
    logger.info("SemanticImageSearch ready")

# Search Endpoint
@app.post("/search_images", response_model=List[Dict])
async def search_images(req: SearchRequest):
    """
    Perform semantic search on images using the query
    """
    try:
        results = await searcher.search(req.query, top_k=req.top_k)
        return results
    except Exception as e:
        logger.error(f"Error searching images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running.
    """
    return {"status": "healthy"}

@app.get("/list_images", response_model=List[str])
async def list_indexed_images():
    """
    List all images that are currently indexed in the vector DB.
    """
    try:
        return [item["image_name"] for item in searcher.metadata_list]
        
    except Exception as e:
        logger.error(f"Error listing images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/streamlit")
def open_streamlit():
    return RedirectResponse(url="http://localhost:8501")


# POST http://127.0.0.1:8000/search_images
# {"query": "mountains at sunset", "top_k": 5 }