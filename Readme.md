## File Structure 
project-root/
│
├── App/                         # Core application logic
│   ├── create_vector_db.py      # Builds FAISS index from images (embeddings + metadata)
│   └── semantic_image_search.py # Performs semantic search using FAISS index
│
├── Data/                        # Data storage
│   ├── Images/                  # Source images to be indexed
│   └── VectorDB/                # Vector database files
│       ├── image_index.faiss    # FAISS index storing embeddings
│       └── metadata.json        # Metadata (image names + descriptions)
│
├── Streamlit/                   # Streamlit frontend
│   └── streamlit_ui.py          # Streamlit app for user interface
│
├── .env                         # Environment variables (API keys, endpoints, etc.)
├── Dockerfile                   # Docker config for containerizing the app
├── download_images.py           # Script to download images from URLs
├── main.py                      # FastAPI backend service
│                                 # - Startup: builds FAISS index if missing
│                                 # - Initializes SemanticImageSearch
│                                 # - Exposes REST API endpoints:
│                                 #   • /search_images → semantic search
│                                 #   • /list_images   → list all indexed images
│                                 #   • /health        → service health check
│                                 #   • /streamlit     → redirects to Streamlit frontend
│
├── photos_url.csv               # CSV file with image URLs
├── requirements.txt             # Python dependencies
└── run.py                       # Entry script: launches both FastAPI (port 8000)
                                 # and Streamlit UI (port 8501) together



## How It Works

On startup ('main.py'), the system loads images from a local folder and initializes the Azure OpenAI client with embedding and chat models. For each image, it generates a concise textual description using the multimodal GPT-4o-mini model and creates embeddings for these descriptions using the text-embedding-3-small model.

These embeddings are aggregated to build a FAISS vector index for fast similarity search. Both the FAISS index and image metadata (descriptions and filenames) are saved to disk for later use.

The FastAPI backend exposes semantic search endpoints that accept a user query, embed it, search the FAISS index, and optionally re-rank results using Azure OpenAI. The Streamlit frontend interacts with these endpoints, providing an interactive UI for searching images. The backend can be used directly via FastAPI or through the Streamlit UI.


## Getting Started

1. Set Azure OpenAI credentials and model names in '.env'.
2. Run 'download_images.py' if needed, to populate images.
3. Launch the app using 'python run.py', which starts FastAPI and Streamlit together.
4. Use the Streamlit UI at http://localhost:8501 or the FastAPI endpoints.



## Dependencies

Managed in 'requirements.txt'. Python 3.8+ recommended.



## Deployment Strategy

- Single Port Handling: Azure Web App exposes only one external port (WEBSITES_PORT), which is used by FastAPI.

- Streamlit Access: Streamlit runs internally in headless mode and is accessed via FastAPI through the /streamlit endpoint.

- Docker Configuration: The Dockerfile exposes the Azure port dynamically, installs dependencies, and runs the combined FastAPI + Streamlit service.


## License & Contribution

Feel free to open issues or pull requests for improvements.



