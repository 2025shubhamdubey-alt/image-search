import streamlit as st
import asyncio
import httpx
import time

# Search function
async def async_search(query: str, top_k: int = 5):
    """
    Call FastAPI async endpoint to get semantic search results.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://127.0.0.1:8000/search_images",
            json={"query": query, "top_k": top_k},
            timeout=60
        )
        response.raise_for_status()
        return response.json()

# Streamlit UI
st.set_page_config(page_title="Semantic Image Search", layout="wide")
st.title("🌄 Semantic Image Search Chatbot")

query = st.text_input("Enter your search query:")
top_k = st.slider("Number of results:", min_value=1, max_value=10, value=5)

if st.button("Search") and query:
    with st.spinner("Searching images..."):
        try:
            start_time = time.time() 

            # Run the async function inside Streamlit
            results = asyncio.run(async_search(query, top_k))
            
            #st.write(results)  # Debug: print raw results

            end_time = time.time() 
            total_time = end_time - start_time

            if not results:
                st.info("No matching images found.")
            else:
                st.success(f"Found {len(results)} images in {total_time:.2f} seconds")

                for res in results:
                    # Create two columns: left = image, right = text
                    col1, col2 = st.columns([1.2, 2])
                    with col1:
                        st.image(res.get("image_url"), width=350)
                    with col2:
                        st.markdown(f"**Rank:** {res['rank']}")
                        st.write("")
                        st.markdown(f"**Explanation:** {res['explanation']}")
                        #st.markdown(f"**Distance:** {res['distance']:.4f}")
                    st.markdown("---")

        except httpx.HTTPStatusError as e:
            st.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            st.error(f"Error: {e}")
