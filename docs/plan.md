### Complete Implementation Guide: Job Searcher + Automated Applications Agent

This is the end-to-end blueprint to build your MVP in 3-5 days, leveraging Llama 3.1 70B Q4 (pulled via Ollama for optimal RTX 5070 performance: ~25-35 tokens/sec on agent chains, 128k context for full JD/resume handling). It integrates FastAPI (backend), Ollama (LLM), LangChain (agents/RAG/tools for tailoring/scraping), and ethical data sources (e.g., Adzuna/USAJobs APIs). Focus on modularity: Start with core components, test incrementally. All runs locally; total code ~200-400 lines. Use a Git repo for versioning.

#### Prerequisites (Day 0: 30-60 mins)
1. **Hardware/Environment Check**: Confirm RTX 5070 setup—run `nvidia-smi` to verify 12GB VRAM free, CUDA 13.0. Install Python 3.10+ if needed (via pyenv). Create a virtual env: `python -m venv job-agent && source job-agent/bin/activate` (Linux/Mac) or `job-agent\Scripts\activate` (Windows).
2. **Install Core Packages**: Run `pip install fastapi uvicorn langchain langchain-community langchain-ollama chromadb faiss-cpu playwright python-docx pypdf2 requests pandas streamlit crewai` (for multi-agent). For scraping: `playwright install` (handles browser automation ethically).
3. **Ollama Setup**: Install Ollama (download from ollama.com). Run `ollama serve` in a terminal. Pull model: `ollama pull llama3.1:70b` (downloads ~40GB; quantizes to Q4 automatically for ~9GB VRAM fit). Test: `ollama run llama3.1:70b` and prompt "Hello" to confirm ~30t/s inference.
4. **API Keys Setup**: Sign up for free tiers—Adzuna (app_id/app_key), USAJobs (no key), Scrapingdog (free credits). Store in `.env` file: `ADZUNA_APP_ID=your_id` etc. Use `python-dotenv` to load.

#### Step 1: Project Structure & Config (30 mins)
5. **Scaffold Project**: Create folders: `job-agent/` with `app/` (FastAPI), `agents/` (LangChain logic), `tools/` (scrapers/APIs), `utils/` (RAG/resume handlers), `tests/` (samples). Add `requirements.txt` with packages above. Main file: `app/main.py` for FastAPI app.
6. **Config LLM in LangChain**: In `utils/llm.py`, define:
   ```python
   from langchain_ollama import ChatOllama
   from dotenv import load_dotenv
   load_dotenv()
   llm = ChatOllama(model="llama3.1:70b", temperature=0.1)  # Low temp for structured outputs
   ```
   This enables tool-calling (e.g., JSON for JD extraction) with 128k context—perfect for RAG on 5-10 page resumes.

#### Step 2: Build RAG for Resume/JD Handling (1-2 hours)
7. **Document Loaders**: In `utils/rag.py`, use LangChain loaders:
   ```python
   from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
   from langchain_text_splitters import RecursiveCharacterTextSplitter
   from langchain_ollama import OllamaEmbeddings
   from chromadb import PersistentClient
   loader = PyPDFLoader("resume.pdf")  # Or Docx2txtLoader
   docs = loader.load()
   splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
   chunks = splitter.split_documents(docs)
   embeddings = OllamaEmbeddings(model="llama3.1:70b")  # Uses same model for embeds
   vectorstore = PersistentClient(path="./chroma_db").get_or_create_collection("resumes")
   vectorstore.add(documents=chunks, embeddings=embeddings.embed_documents([c.page_content for c in chunks]))
   retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
   ```
   Test: Query "Extract skills" → retrieves relevant chunks for tailoring.

#### Step 3: Implement Job Search Tools (2-3 hours)
8. **API-Based Scrapers**: In `tools/search_tools.py`, create LangChain tools for legal sources (no scraping bans):
   - Adzuna Tool:
     ```python
     from langchain.tools import tool
     import requests
     @tool
     def adzuna_search(what: str, where: str = "remote") -> str:
         url = f"https://api.adzuna.com/v1/api/jobs/us/search/1?app_id={os.getenv('ADZUNA_APP_ID')}&app_key={os.getenv('ADZUNA_APP_KEY')}&what={what}&where={where}&results_per_page=20"
         resp = requests.get(url).json()
         return [{"title": j["title"], "company": j["company"]["display_name"], "jd": j["description"]} for j in resp["results"]]
     ```
   - USAJobs Tool: Similar OData query: `https://data.usajobs.gov/api/search?Keyword={what}` (parse JSON for JDs).
   - Fallback: Playwright for career pages (e.g., Greenhouse):
     ```python
     from langchain_community.tools.playwright.utils import create_sync_playwright_browser
     @tool
     def scrape_career_page(url: str) -> str:
         with create_sync_playwright_browser() as p:
             page = p.new_page()
             page.goto(url)
             page.wait_for_load_state("networkidle")
             return page.content()  # Extract JDs via selector (e.g., '.job-description')
     ```
     Add delays: `page.wait_for_timeout(2000)` for ethics.
9. **Agent for Search**: In `agents/search_agent.py`, use ReAct:
   ```python
   from langchain.agents import create_react_agent, AgentExecutor
   from langchain.prompts import PromptTemplate
   prompt = PromptTemplate.from_template("Search for {query} jobs. Use tools: {tools}. Respond with list of 10 matches.")
   agent = create_react_agent(llm, [adzuna_search, usajobs_search, scrape_career_page], prompt)
   executor = AgentExecutor(agent=agent, tools=[...], verbose=True)
   ```
   Test: `executor.invoke({"query": "Python developer remote"})` → JSON list of jobs.

#### Step 4: Tailoring & Application Agents (3-4 hours)
10. **Tailoring Agent**: In `agents/tailor_agent.py`, chain RAG + LLM:
    ```python
    from langchain.chains import RetrievalQA
    from langchain.schema import StrOutputParser
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type="stuff")
    def tailor_resume(jd: str, resume_path: str) -> str:
        context = qa_chain.run(f"Extract key skills from JD: {jd}")
        prompt = f"Using resume context: {context}, tailor bullets for JD: {jd}. Output DOCX-ready Markdown."
        tailored = llm.invoke(prompt)  # Llama's reasoning shines here
        from docx import Document
        doc = Document()
        doc.add_paragraph(tailored.content)
        doc.save("tailored_resume.docx")
        return "Tailored resume saved."
    ```
    Use CrewAI for multi-role: `TailorCrew = Crew(agents=[researcher, writer], tasks=[extract_task, rewrite_task])`.
11. **Application Automator**: In `agents/apply_agent.py`, extend with Playwright:
    ```python
    @tool
    def auto_apply(job_url: str, resume_path: str) -> str:
        with create_sync_playwright_browser() as p:
            page = p.new_page()
            page.goto(job_url)
            # Ethical: Pause for user approval
            input("Approve and press Enter to proceed...")
            page.fill('input[name="resume"]', resume_path)  # Adapt selectors
            page.click('button[type="submit"]')
            return "Applied successfully."
    ```
    Agent: ReAct chain calling tailor → apply, with memory (LangGraph state for tracking apps).

#### Step 5: FastAPI Backend Integration (1-2 hours)
12. **Endpoints**: In `app/main.py`:
    ```python
    from fastapi import FastAPI, UploadFile, File
    from agents.search_agent import executor as search_exec
    app = FastAPI()
    @app.post("/search")
    def search_jobs(query: str):
        return search_exec.invoke({"query": query})
    @app.post("/tailor")
    def tailor( jd: str, resume: UploadFile = File(...)):
        # Save upload, call tailor_resume
        return {"status": "Tailored"}
    @app.post("/apply")
    def apply(job_url: str, resume_path: str):
        return auto_apply(job_url, resume_path)
    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    ```
    Add Pydantic for validation: e.g., `class JobQuery(BaseModel): query: str`.
13. **UI Layer**: In `ui/dashboard.py`, use Streamlit:
    ```python
    import streamlit as st
    import requests
    st.title("Job Agent")
    query = st.text_input("Job search")
    if st.button("Search"):
        resp = requests.post("http://localhost:8000/search", json={"query": query})
        st.json(resp.json())
    # Add file upload for resume, tailor/apply buttons
    ```
    Run: `streamlit run ui/dashboard.py`.

#### Step 6: Testing, Security, & Deployment (1 day)
14. **Unit/End-to-End Tests**: In `tests/test_agent.py`, use pytest: Mock APIs (`responses` lib), test tailoring on sample JD (from USAJobs). E2E: Search → Tailor → Simulate apply. Edge cases: API fails (fallback to cache), long JDs (chunk via RAG).
15. **Security/Privacy**: Encrypt DB (`cryptography` lib for Chroma). No cloud storage—local SQLite for logs: `job_logs.db` with applied URLs. Add rate-limiting (FastAPI middleware). Ethics: Log warnings for scraped sites; user consent prompts.
16. **Optimization for RTX 5070**: Monitor VRAM with `nvidia-smi -l 1` during runs. If >10GB used, drop to Llama 3.1 8B. For speed: Use vLLM wrapper in LangChain (`from langchain_vllm import VLLM`—CUDA-accelerated batching for 2x parallel searches).
17. **Local Deployment**: Run Ollama server, then `uvicorn app.main:app --reload`. Access UI at `localhost:8501`. Persist DBs in `./data/`.
18. **Expansion Hooks**: Add LangGraph for stateful memory (track rejections). Monitor with logging (`structlog`). Scale: Dockerize for VPS if needed.

Run `python -m pytest` to validate, then fire up the dashboard—upload your resume, search "FastAPI dev", and watch it tailor/apply. This delivers a usable, privacy-safe tool saving hours weekly. Debug via Ollama's verbose mode if chains drift. Fork from AIHawk GitHub for extras. You're set!
