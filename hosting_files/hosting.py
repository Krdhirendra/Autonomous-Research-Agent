import streamlit as st
import os
import json
import requests
from io import BytesIO
from markdown_pdf import MarkdownPdf, Section
from hosting_V2_RAG_classes import Chunks, EmbeddingManager, VectorStore, RAGRetriever
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from readability import Document
from langchain_core.documents import Document as LCDocument

# Use Streamlit Secrets for Hosting
GROQ_TOKEN = st.secrets["GROQ_TOKEN"]
TAVILY_TOKEN = st.secrets["TAVILY_API_KEY"]
os.environ["TAVILY_API_KEY"] = TAVILY_TOKEN

st.title("ARA: Autonomus Research Agent")
st.markdown(
    "<p style='text-align:center; color:grey; font-size:20px;'>"
    "© Created by <b>Dhirendra</b>"
    "</p>",
    unsafe_allow_html=True
)
st.markdown("Enter a research topic, and I will find papers and write a report for you.")

user_input = st.text_input("Enter your research prompt:", placeholder="e.g., Compare Transformers and LSTMs for time-series")

if st.button("Run Research Agent"):
    if not user_input:
        st.warning("Please enter a prompt first.")
    else:
        with st.status("🤖 Agent is working...", expanded=True) as status:
            # --- 1. TASK UNDERSTANDING ---
            st.write("Understanding task...")
            # (System prompt remains the same as your file)
            llm = ChatGroq(
                model = 'llama-3.3-70b-versatile',
                api_key=GROQ_TOKEN,
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
            system_prompt = """
                You are a helpful assistant. Your task is to analyze the user's question and produce a structured classification.

                Steps:
                1. Determine the task type(s).
                Possible categories:
                - RESEARCH
                - SURVEY
                - COMPARISON
                If multiple categories apply, include all of them.

                2. Extract key information from the question, including (when applicable):
                - topic
                - time_range
                - methods / approaches
                - constraints or assumptions

                Output format:
                - Return ONLY a valid JSON object.
                - Do NOT include explanations, markdown, or extra text.
                - Use lowercase values for task_type.

                Example:

                User question:
                "Compare Transformers and LSTMs for time-series forecasting"

                Output:
                {
                "user_prompt" : ["Compare Transformers and LSTMs for time-series forecasting"]
                "task": {
                "task_type": ["comparison"],
                "topic": "time-series forecasting",
                "methods": ["transformers", "lstms"]}
                }

                Some Rules:
                1. if you are saying it is classification task type you should provide the atleast 2 diffrent methods in comparision from the user input
                you cant just provide one method in comparison and say it is a comparision task.
                """

            messages = [SystemMessage(content = system_prompt),HumanMessage(content=f'{user_input}')]

            response = llm.invoke(messages)
            task_n_prompt = response.content
            
            # Generate Search queries for web searching
            query_system_prompt = """
            You are a search query generator.

            Your task is to generate effective ACADEMIC search queries
            based on the user's original question and a structured Python dictionary.

            Rules:
            - Generate 3 to 5 distinct academic search queries
            - Queries should be suitable for Google Scholar / arXiv
            - Return ONLY a valid JSON object
            - No explanations, no markdown

            Sample INPUT:
            {
            "user_prompt": "Compare Transformers and LSTMs for time-series forecasting",
            "task": {
                "task_type": ["comparison"],
                "topic": "time-series forecasting",
                "methods": ["transformers", "lstms"]
            }
            }

            Sample OUTPUT:
            {
            "search_queries": [
                "transformer vs lstm time series forecasting",
                "benchmark transformer lstm time series forecasting",
                "deep learning time series forecasting comparison paper"
            ]
            }
            """

            # LLM for Query Generation
            query_llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=GROQ_TOKEN,
                temperature=0.2,       
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )

            query_messages = [SystemMessage(content=query_system_prompt),HumanMessage(content=task_n_prompt)]
            query_response = query_llm.invoke(query_messages)

            # Parse & Validate Output
            try:
                parsed_output = json.loads(query_response.content)
                search_queries = parsed_output["search_queries"]
                if not (3 <= len(search_queries) <= 5):
                    raise ValueError("Expected 3–5 search queries")

            except Exception as e:
                raise ValueError(f"Invalid query generator output: {e}")
            

            # --- 2. WEB SEARCHING ---
            st.write("Searching academic sources...")
            TAVILY_TOKEN = os.environ['TAVILY_API_KEY']

            def search_infos(queries,max_results=3):
                search = TavilySearchResults(max_results=max_results)
                all_results = []

                for query in queries:
                    pdf_query = f"{query} research paper OR arxiv"
                    web_query = f"{query} blog OR article OR analysis"

                    pdf_results = search.invoke(pdf_query)
                    web_results = search.invoke(web_query)

                    results = pdf_results + web_results
                    all_results.extend(results)

                seen_urls = set()
                unique_results = []

                for item in all_results:
                    if item['url'] not in seen_urls:
                        unique_results.append(item)
                        seen_urls.add(item['url'])
                return unique_results
            
            urls = search_infos(search_queries,max_results=3)


            # --- 3. RAG PROCESSING ---
            st.write("Reading and Embedding documents...")

            def extract_text_from_url(url):
                headers = {"User-Agent":"Mozilla/5.0"}
                response = requests.get(url, headers=headers, timeout=10)

                if response.status_code != 200:
                    raise Exception(f"Failed to Fetch {url}\n")
                
                content_type = response.headers.get("content-Type","")

                if "application/pdf" in content_type or url.endswith('.pdf'):
                    return extract_pdf(response.content)
                elif "text/html" in content_type:
                    return extract_html(response.text)
                else:
                    return response.text
                

            def extract_html(html_content):
                doc = Document(html_content)
                cleaned_html = doc.summary()

                soup = BeautifulSoup(cleaned_html, "html.parser")

                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                return soup.get_text(separator=" ")


            def extract_pdf(binary_content):
                try:
                    reader = PdfReader(BytesIO(binary_content))
                    text_parts = []

                    for i, page in enumerate(reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                clean_text = str(page_text).replace('\x00','')
                                if clean_text:
                                    text_parts.append(clean_text)

                        except Exception as page_error:
                            print(f"  > Warning: Could not extract page {i+1}. Skipping stream. Error: {page_error}")
                            continue
                            
                    return " ".join(text_parts)
                
                except Exception as e:
                    print(f"Critical error reading PDF structure: {e}")
                    return ""

            def read_documents(urls):
                embed_manager = EmbeddingManager()
                vectorstore=VectorStore(persist_directory='data/2_new_vector_store')
                for url in urls:
                    try:
                        text = extract_text_from_url(url['url'])

                        chunker = Chunks(text)
                        chunks_text = chunker.split_documents()
                        chunk_docs = [LCDocument(page_content=t, metadata={"source": url['url']}) for t in chunks_text]            

                        embeddings = embed_manager.generate_embeddings(texts=chunks_text)
                       
                        vectorstore.add_documents(documents=chunk_docs, embeddings=embeddings)

                    except Exception as e:
                        print(f"Skipping the {url['url']}\nERROR: {e}")

                return embed_manager, vectorstore
            
            embed_manager, vectorstore = read_documents(urls)
            #a llm for generating some retrieval queries
            RETRIEVAL_QUERY_PROMPT = """
            You are a retrieval query generator for a research assistant.

            Your task is to generate 5 to 7 focused retrieval queries
            that help retrieve information from research papers.

            Rules:
            - DO NOT answer the question
            - DO NOT explain anything
            - DO NOT include markdown
            - DO NOT include bullet points
            - Output ONLY a valid JSON object
            - The response must start with '{' and end with '}'

            Output format:
            {
            "queries": [
                "query 1",
                "query 2",
                "query 3",
                "query 4",
                "query 5"
            ]
            }
            """
            def generate_retrieval_queries(user_query: str, api_key: str) -> list[str]:
                query_expansion_llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    api_key=api_key,
                    temperature=0.2,   # low = stable retrieval
                )

                messages = [
                    SystemMessage(content=RETRIEVAL_QUERY_PROMPT),
                    HumanMessage(content=user_query)
                ]

                response = query_expansion_llm.invoke(messages)
                # print("RAW LLM OUTPUT:")
                # print(response.content)
                # print("------")
                try:
                    parsed = json.loads(response.content)
                    queries = parsed["queries"]

                    if not (5 <= len(queries) <= 7):
                        raise ValueError("Expected 5–7 retrieval queries")

                    return queries

                except Exception as e:
                    raise RuntimeError(f"Invalid retrieval-query output: {e}")


            def multi_query_retrieve(
                retriever,
                queries: list[str],
                top_k: int = 10
            ) -> list[dict]:
                """
                Runs RAG retrieval for each query and merges results.
                """
                all_chunks = []

                for q in queries:
                    chunks = retriever.retrieve(q, top_k=top_k)
                    all_chunks.extend(chunks)

                return all_chunks


            def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
                seen = set()
                deduped = []

                for chunk in chunks:
                    key = (
                        chunk["content"][:200],  # content fingerprint
                        chunk["metadata"].get("source")
                    )

                    if key not in seen:
                        seen.add(key)
                        deduped.append(chunk)

                return deduped



            def extract_key_infos(vectorstore,embed_manager):
                user_prompt = json.loads(task_n_prompt)['user_prompt'][0]
                ragretriver = RAGRetriever(vector_store=vectorstore, embedding_manager=embed_manager)
                # Step 1: expand query
                retrieval_queries = generate_retrieval_queries(
                    user_query=user_prompt,
                    api_key=GROQ_TOKEN
                )

                # Step 2: retrieve per query
                raw_chunks = multi_query_retrieve(
                    retriever=ragretriver,
                    queries=retrieval_queries,
                    top_k=10
                )

                # Step 3: deduplicate
                clean_chunks = deduplicate_chunks(raw_chunks)

                # Step 4: promote to papers (you already implemented this)
                papers = ragretriver.group_chunks_by_paper(clean_chunks)
                return papers
            
            papers = extract_key_infos(vectorstore=vectorstore,embed_manager=embed_manager)
            
            # --- 4. REPORT GENERATION ---
            st.write("Verifying acquired informations")

            verify_llm = ChatGroq(
                model = 'meta-llama/llama-4-scout-17b-16e-instruct',
                # model = 'llama-3.3-70b-versatile',
                api_key=GROQ_TOKEN,
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )

            verify_system_prompt = """
            You are an academic paper relevance evaluator.

            Your task is to evaluate a SINGLE paper context against a user query.

            You must decide:
            1. Whether this paper is relevant enough to help answer the user query.
            2. If relevant, classify the paper’s role.

            Definitions:
            - A paper is RELEVANT if it:
            • directly compares methods mentioned in the query, OR
            • provides a survey/review of the topic, OR
            • provides guidelines, benchmarks, or empirical insights related to the query.

            - A paper is NOT RELEVANT if it:
            • only briefly mentions the topic without analysis, OR
            • focuses on a single narrow method without comparison or broader insight, OR
            • is unrelated to answering the user query.

            If the paper is NOT relevant:
            → Output exactly:
            NO

            If the paper IS relevant:
            → Classify it into ONE of the following categories:
            survey
            comparison
            method-specific
            tutorial

            Output rules:
            - Output ONLY one token.
            - Do NOT include explanations.
            - Do NOT include punctuation or formatting.

            """
            def verify_papers(papers):
                verified_papers = []
                for paper in papers:
                    # is_verified = False
                    verify_human_prompt = f"""
                        here is the User's query:
                        {task_n_prompt}

                        here is the Context:
                        {paper['best_chunk']}
                    """
                    verify_messages = [
                    SystemMessage(content=verify_system_prompt),
                    HumanMessage(content=verify_human_prompt)
                    ]
                    response = verify_llm.invoke(verify_messages)
                    if response.content in ['survey','comparison','method-specific','tutorial']:
                        verified_papers.append(paper)
                        # is_verified = True
                    
                    # if is_verified:
                    #   print(f"{paper['source']} got verified")

                return verified_papers
            
            verified_papers = verify_papers(papers)
            st.write("preparing verified data...........")
            #this is what that will be feeded into the the write llm.
            def prepare_input(verified_papers):
                req_info = []
                for papera in verified_papers:
                    url = papera['source']
                    for chunka in papera['chunks']:
                        chunkas = chunka['content']
                        req_info.append({
                            "content":chunkas,
                            "url": url
                        })
                return req_info
            
            req_info = prepare_input(verified_papers=verified_papers)

            st.write("Generating final report...")
            writer_llm = ChatGroq(
                model='openai/gpt-oss-20b',
                # model='llama-3.3-70b-versatile',
                api_key=GROQ_TOKEN,
                temperature=0.3,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )

            writer_system_prompt = """
                Based on provided verified information in this format:
                [{
                    "content":content,
                    "url": url
                }]

                against a given user's QUERY

                Using ONLY the provided verified information, generate a structured
                and Detailed research-style report answering the user query.

                Include the urls for refrences

                If the provided information is insufficient to confidently answer
                any part of the query, explicitly state the missing information
                instead of guessing.

                Output format:
                    {"Title":small but concise title for the report,
                    "Report":"Report content in proper formatting"}
                Output rules:
                - Report should start with a proper title with proper formatting
                - Report should follow a proper format of report writing
                - Output ONLY the final report in proper docs formatting so that later it can be saved in form PDF.
                - Do NOT include extra tokens other than the report.
                
            """

            def write_report(req_info):
                x = json.loads(task_n_prompt)['user_prompt'][0]
                trimmed_info = str(req_info)[:25000]
                writer_human_prompt = f"""
                Here is the 
                    provided verified information: {trimmed_info},
                    User's Query: {x}
                """

                writer_messages = [
                    SystemMessage(writer_system_prompt),
                    HumanMessage(writer_human_prompt)
                ]

                response = writer_llm.invoke(writer_messages)
                return response.content
            
            final_report = write_report(req_info=req_info)
            
        status.update(label="✅ Research Complete!", state="complete")


        # DISPLAY REPORT
        fr = json.loads(final_report)
        st.header(fr['Title'])
        st.markdown(fr['Report'])

        # DOWNLOAD PDF
        pdf = MarkdownPdf(toc_level=2)
        pdf.add_section(Section(fr['Report']))
        
        # Save to a buffer instead of a local path for hosting
        pdf_buffer = BytesIO()
        pdf.save(pdf_buffer)
        
        st.download_button(
            label="Download Report as PDF",
            data=pdf_buffer.getvalue(),
            file_name="ARA_Research_Report.pdf",
            mime="application/pdf"

        )

