import os
import base64
import json
import faiss
import numpy as np
import logging
import asyncio
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ImageIndexer")


class ImageIndexer:
    def __init__(self, image_folder: str = "Data/Images", output_dir: str = "Data/VectorDB"):
        load_dotenv()

        # Azure OpenAI config
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.chat_model = os.getenv("AZURE_OPENAI_CHAT_MODEL_NAME")
        self.embed_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME")

        if not all([self.api_key, self.endpoint, self.api_version, self.chat_model, self.embed_model]):
            logger.error("Missing one or more required environment variables")
            raise ValueError("Missing environment variables for Azure OpenAI configuration")

        # Async Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )

        # Paths
        self.image_folder = image_folder
        self.output_dir = output_dir
        self.index_path = os.path.join(output_dir, "image_index.faiss")
        self.metadata_path = os.path.join(output_dir, "metadata.json")

        os.makedirs(self.output_dir, exist_ok=True)

    async def build_index(self):
        logger.info("Starting image indexing...")

        metadata_list = []
        embeddings = []

        for img_file in os.listdir(self.image_folder):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(self.image_folder, img_file)
            logger.info(f"Processing image: {img_file}")

            # Convert image to base64
            with open(img_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")

            # Generate description
            try:
                
                description_response = await self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant that generates image descriptions for semantic search. "
                            "Create a concise yet vivid description (2-3 sentences max). "
                            "Preserve key visual elements, attributes, and context within the image "
                            "(e.g., colors, lighting, scene, objects, mood, and style). "
                            "The description should be business-appropriate, natural, and precise."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Generate a concise description for this image."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                        ]
                    }
                ]
                     )
                description = description_response.choices[0].message.content.strip()

            
            except Exception as e:
                logger.error(f"Failed to generate description for {img_file}: {e}", exc_info=True)
                continue

            # Create embedding
            try:
                embed_response = await self.client.embeddings.create(
                    model=self.embed_model,
                    input=description
                )
                embedding = embed_response.data[0].embedding
            except Exception as e:
                logger.error(f"Failed to generate embedding for {img_file}: {e}", exc_info=True)
                continue

            metadata_list.append({
                "image_name": img_file,
                "description": description
            })
            embeddings.append(embedding)

        if not embeddings:
            logger.warning("No valid images processed. FAISS index not created.")
            return

        # Convert embeddings → numpy
        embeddings_np = np.array(embeddings).astype("float32")

        # Build FAISS index
        d = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings_np)

        # Save outputs
        faiss.write_index(index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, indent=2)

        logger.info(f"FAISS index and metadata created in: {self.output_dir}")

