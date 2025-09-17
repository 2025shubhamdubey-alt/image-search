import math
import os
import json
import faiss
import logging
import numpy as np
import asyncio
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from typing import List, Dict

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SemanticImageSearch")

class SemanticImageSearch:
    def __init__(
        self,
        index_path: str = "Data/VectorDB/image_index.faiss",
        metadata_path: str = "Data/VectorDB/metadata.json",
        image_folder: str = "Data/Images",
    ):
        load_dotenv()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.embed_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME")
        self.model = os.getenv("AZURE_OPENAI_COMPLETION_MODEL_NAME")  # Correct env var name

        if not all([self.api_key, self.endpoint, self.api_version, self.embed_model, self.model]):
            raise ValueError("Missing Azure OpenAI environment variables")

        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )

        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata_list = json.load(f)

        self.image_folder = image_folder
        
        logger.info("SemanticImageSearch initialized successfully")

    def clean_ellipsis(self, text: str) -> str:
        # Replace ellipsis literals with placeholder string
        if text is None:
            return ""
        # Defensive: convert non-string to string
        if not isinstance(text, str):
            text = str(text)
        return text.replace("...", "[ellipsis]")
    
    async def generate_hyde_for_query(self, query: str) -> str:
        prompt = f"Explain the following input query in a simple and concise manner, clearly stating the main intent:\n\n\"{query}\""
        #prompt = f"Given the query, generate a detailed hypothetical answer or document that best addresses the query:\n\n\"{query}\""
        response = await self.client.chat.completions.create(
            model=self.model,

            
            messages=[
                {"role": "system", "content": "You are a helpful assistant who converts input query into simple, concise query."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=50,
        )
        hypothetical_doc = response.choices[0].message.content.strip()
        return hypothetical_doc

    async def re_rank_chunks(self, query: str, retrieved_chunks: List[Dict]) -> List[Dict]:
        half_len = math.ceil(len(retrieved_chunks) / 2)
        try:
            if len(retrieved_chunks) == 0:
                return []

            # Clean explanations to avoid ellipsis serialization error
            cleaned_chunks = []
            for chunk in retrieved_chunks:
                cleaned_exp = self.clean_ellipsis(chunk.get("explanation", ""))
                cleaned_chunks.append({**chunk, "explanation": cleaned_exp})

            # Build prompt with cleaned explanations
            prompt_chunks = "\n".join(
                [f"{i+1}. {chunk['explanation']}" for i, chunk in enumerate(cleaned_chunks)]
            )
            prompt = (
                f"Given the following query:\n\"{query}\"\n\n"
                f"Here are several descriptions:\n{prompt_chunks}\n\n"
                "Please rank these descriptions from most relevant (1) to least relevant "
                f"({len(cleaned_chunks)}) according to how well they match the query. "
                "Respond only with a comma-separated list of numbers representing the order.\n\n"
                "Example response: 3,1,2"
            )

            # Call Azure OpenAI chat completions with correct await and create method
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an assistant that ranks text relevance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200,
            )

            ranking_text = response.choices[0].message.content.strip()
            ranking_indices = [int(x.strip()) - 1 for x in ranking_text.split(",") if x.strip().isdigit()]
            if len(ranking_indices) != len(cleaned_chunks):
                raise ValueError("Ranking length mismatch")

            top_indices = ranking_indices[:half_len]
            re_ranked_chunks = [retrieved_chunks[i] for i in top_indices]

            for i, chunk in enumerate(re_ranked_chunks):
                chunk['rank'] = i + 1

            logger.info(f"Re-ranked to top {len(re_ranked_chunks)} chunks")
            return re_ranked_chunks

        except Exception as e:
            logger.error(f"Re-ranker failed: {e}")
            fallback_chunks = retrieved_chunks[:half_len]
            for i, chunk in enumerate(fallback_chunks):
                chunk['rank'] = i + 1
            logger.info(f"Returned fallback top {len(fallback_chunks)} chunks")
            return fallback_chunks

    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        logger.info(f"Searching for: '{query}'")

        #hyde_doc = await self.generate_hyde_for_query(query)

        embed_response = await self.client.embeddings.create(
            model=self.embed_model,
            input=query
        )
        query_embedding = np.array(embed_response.data[0].embedding).astype("float32").reshape(1, -1)

        distances, indices = self.index.search(query_embedding, 2 * top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            image_name = self.metadata_list[idx]["image_name"]
            results.append({
                "rank": i + 1,
                "image_name": image_name,
                "image_url": os.path.join(self.image_folder, image_name).replace("\\", "/"),
                "explanation": self.metadata_list[idx].get("description", ""),
                "distance": float(distances[0][i])
            })

        final_results = await self.re_rank_chunks(query, results)
        return final_results
