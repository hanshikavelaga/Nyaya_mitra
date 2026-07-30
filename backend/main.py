from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json
import logging
import google.generativeai as genai
import threading
import time
from contextlib import asynccontextmanager

from sqlalchemy.orm import Session
from database import engine, get_db, SessionLocal
import models

# Import API Routers
from routers.upload import router as upload_router
from services.rag_retriever import retrieve_matching_laws
from services.gemini_client import generate_text_completion
from config import GEMINI_API_KEY, OPENROUTER_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure standard Gemini if key is present
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize SQLite database tables on server startup
models.Base.metadata.create_all(bind=engine)

# Seed database with laws if empty
def seed_database_laws():
    db = next(get_db())
    try:
        count = db.query(models.Law).count()
        if count == 0:
            logger.info("Database laws table is empty. Seeding from laws_kb.json...")
            import os
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(BASE_DIR, "data", "laws_kb.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    laws = data.get("laws", [])
                    for item in laws:
                        db_law = models.Law(
                            act=item["act"],
                            category=item["category"],
                            scope=item["scope"],
                            summary=item["summary"],
                            details=item["details"],
                            remedies_json=json.dumps(item["remedies"]),
                            keywords_json=json.dumps(item["keywords"])
                        )
                        db.add(db_law)
                    db.commit()
                    logger.info(f"Successfully seeded {len(laws)} laws into SQLite database.")
            else:
                logger.error(f"Seeding failed: {json_path} not found.")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to seed database: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Launch background translation loop
    print(">>> [NyayaMitra] Background translation loop starting up... <<<", flush=True)
    translation_thread = threading.Thread(target=translate_laws_in_background, daemon=True)
    translation_thread.start()
    yield
    # Shutdown

seed_database_laws()

app = FastAPI(title="NyayaMitra AI API", version="1.0.0", lifespan=lifespan)

# Enable CORS for frontend cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(upload_router)

class ChatMessage(BaseModel):
    role: str # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    message: str
    document_id: int
    history: List[ChatMessage] = []
    language: Optional[str] = "english"

class TranslateRequest(BaseModel):
    text: str
    target_language: str # e.g. 'telugu'

@app.get("/")
def read_root():
    return {"status": "running", "app": "NyayaMitra AI Backend (SQLite Active)"}

@app.post("/api/chat")
async def chat_interaction(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        doc_id = request.document_id
        user_query = request.message
        lang = (request.language or "english").lower()

        # 1. Verify document exists in database
        doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Active document context not found in database.")

        # 2. Write User Message to Database logs
        user_log = models.ChatLog(document_id=doc_id, role="user", content=user_query)
        db.add(user_log)

        # 3. Retrieve matched laws dynamically (RAG)
        search_query = f"{doc.filename} {doc.raw_text} {user_query}"
        matched_laws = retrieve_matching_laws(search_query)
        laws_context_str = json.dumps(matched_laws, indent=2)

        # 4. Generate grounded reply (OpenRouter, Gemini, or Mock Fallback)
        disclaimer = "\n\n---\n*Disclaimer: NyayaMitra provides simplified legal translations based on matched public statutes. This is not formal legal advice. Please consult an advocate before filing appeals.*"
        disclaimer_telugu = "\n\n---\n*గమనిక: న్యాయమిత్ర సాధారణ చట్టాల సమాచారాన్ని మాత్రమే అందిస్తుంది. ఇది అధికారిక న్యాయ సలహా కాదు. ఏవైనా చర్యలు తీసుకునే ముందు లాయర్‌ను సంప్రదించండి.*"
        
        if not GEMINI_API_KEY and not OPENROUTER_API_KEY:
            logger.warning("No API keys configured. Loading matched mock RAG response.")
            matched_law_name = matched_laws[0].get("act") if matched_laws else "General Civil Code"
            matched_law_summary = matched_laws[0].get("summary") if matched_laws else "Verify notices in writing."
            
            if lang == "telugu":
                assistant_reply = (
                    f"మీ పత్రం '{doc.filename}' మరియు చట్టపరమైన మార్గదర్శకాల ఆధారంగా: **{matched_law_name}**,\n"
                    f"మీ ప్రశ్నకు ఇక్కడ సాధారణ వివరణ ఉంది:\n\n"
                    f"* నియమం: {matched_law_summary}\n"
                    f"* సిఫార్సు: మీ నోటీసులో ఈ వివరాలు ఉన్నాయో లేదో చూసుకోండి మరియు గడువు లోపల సమాధానం ఇవ్వండి.\n"
                    f"మీరు కోరితే నేను మీ కోసం ప్రత్యుత్తర లేఖను సిద్ధం చేయగలను!"
                    f"{disclaimer_telugu}"
                )
            else:
                assistant_reply = (
                    f"Based on your document '{doc.filename}' and matched legal guidelines: **{matched_law_name}**,\n"
                    f"here is a helpful response to your query '{user_query}':\n\n"
                    f"* Statute: {matched_law_summary}\n"
                    f"* Recommendation: Check if your notice lists these parameters and reply within the given deadline.\n"
                    f"Let me know if you would like me to draft a dispute letter for you!"
                    f"{disclaimer}"
                )
        else:
            # Build grounded RAG chat prompt
            system_prompt = f"""
You are NyayaMitra AI, a citizen-friendly legal translation assistant.
You are helping a citizen understand a legal notice they uploaded.
Provide a simple, clear, and reassuring reply to their question.

DOCUMENT CONTEXT:
Filename: {doc.filename}
Classified Category: {doc.doc_type}
Raw text: {doc.raw_text}

GROUNDING LAWS CONTEXT (RAG):
{laws_context_str}

Instructions:
1. Settle their queries clearly, avoiding complex legalese.
2. Ground your facts solely in the provided document context and matched laws. Do not make up external legal proceedings.
3. Keep it brief (under 150 words).
"""
            if lang == "telugu":
                system_prompt += """
4. IMPORTANT: The user has selected Telugu. You MUST respond in simple, clear, everyday, citizen-friendly Telugu. Do NOT use complex Sanskritized Telugu legal words (e.g. use 'నోటీసు' instead of 'నోటిఫికేషన్', 'లీజు' or 'అద్దె' instead of 'కౌలు', 'కోర్టు' instead of 'న్యాయస్థానం', 'లాయర్' instead of 'న్యాయవాది', 'కేసు' instead of 'అభియోగం'). Write in a comforting, conversational tone so it is easy for a regular person to understand.
"""
            # Format history for prompt wrapper
            history_list = []
            for msg in request.history:
                history_list.append({
                    "role": "user" if msg.role == "user" else "assistant",
                    "content": msg.content
                })

            try:
                reply = generate_text_completion(
                    system_prompt=system_prompt,
                    user_prompt=user_query,
                    history_messages=history_list
                )
                assistant_reply = reply + (disclaimer_telugu if lang == "telugu" else disclaimer)
            except Exception as e:
                logger.error(f"Chat completion call failed: {str(e)}")
                if lang == "telugu":
                    assistant_reply = f"క్షమించండి, ఏఐ ఇంజిన్‌తో కమ్యూనికేషన్ సమస్య ఏర్పడింది. చట్టబద్ధమైన గైడ్: {matched_laws[0].get('act') if matched_laws else 'General Civil Code'}" + disclaimer_telugu
                else:
                    assistant_reply = f"I apologize, I encountered a communication error with our AI engine. Fallback Act match: {matched_laws[0].get('act') if matched_laws else 'General Civil Code'}" + disclaimer
        
        # 5. Write Assistant Message to Database logs
        assistant_log = models.ChatLog(document_id=doc_id, role="assistant", content=assistant_reply)
        db.add(assistant_log)
        
        db.commit()

        return {
            "document_id": doc_id,
            "reply": assistant_reply
        }
    except Exception as e:
        db.rollback()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Chat database logging failed: {str(e)}")

@app.post("/api/translate")
async def translate_text(request: TranslateRequest):
    text_to_translate = request.text
    lang = request.target_language.lower()
    
    if not GEMINI_API_KEY and not OPENROUTER_API_KEY:
        # Static English-to-Telugu translation dictionary for demo fallback
        telugu_dictionary = {
            "1. Document Ingestion": "1. పత్రం అప్‌లోడ్",
            "2. AI Analysis Workspace": "2. ఏఐ విశ్లేషణ విభాగం",
            "Drag & drop your notice file here": "మీ నోటీసు పత్రాన్ని ఇక్కడ డ్రాప్ చేయండి",
            "Supports PDF, PNG, JPEG up to 10MB": "PDF, PNG, JPEG ఫార్మాట్లు (గరిష్టంగా 10MB)",
            "Browse File": "ఫైల్ ఎంచుకోండి",
            "Analyzing legal document...": "పత్రాన్ని విశ్లేషిస్తోంది...",
            "Extracting text with Gemini AI": "జెమిని ఏఐ ద్వారా సమాచారాన్ని సేకరిస్తోంది",
            "Remove": "తొలగించు",
            "Notice File Detail Preview": "పత్రం పాఠ్య వివరణ ప్రివ్యూ",
            "Listen Summary": "సారాంశం వినండి",
            "Pause Audio": "ఆడియో నిలిపివేయండి",
            "Plain Language Explanation": "సాధారణ భాషా వివరణ",
            "Critical Milestones Timeline": "కీలక గడువుల కాలక్రమం",
            "Relevant Legal Citations": "సంబంధిత చట్టపరమైన ఆధారాలు",
            "Recommended Next Steps": "సిఫార్సు చేయబడిన తదుపరి చర్యలు",
            "Autogenerated Response Template": "స్వయంచాలక ప్రత్యుత్తర నమూనా",
            "Your landlord, Greenwood Management, claims you defaulted on July rent of INR 25,000.": "మీ యజమాని గ్రీన్‌వుడ్ మేనేజ్‌మెంట్ మీరు రూ. 25,000 అద్దె చెల్లించలేదని నోటీసు పంపారు.",
            "Verify receipts.": "రశీదులను సరిచూసుకోండి.",
            "Draft legal dispute reply.": "ప్రత్యుత్తర లేఖను సిద్ధం చేయండి."
        }
        translated = telugu_dictionary.get(text_to_translate, f"[తెలుగు అనువాదం] {text_to_translate}")
    else:
        try:
            translation_prompt = (
                "Translate the following text into simple, conversational, everyday Telugu (\"వాడుక భాష\").\n"
                "Make it extremely clear and easy for a common citizen to read. Avoid complex Sanskritized legal words.\n"
                "If a word is commonly used in English, write it in Telugu script (transliterate) rather than using a complex Telugu word.\n"
                "For example, use 'నోటీసు' (notice) instead of 'నోటిఫికేషన్/హెచ్చరిక', 'కోర్టు' (court) instead of 'న్యాయస్థానం', "
                "'లాయర్' (lawyer) instead of 'న్యాయవాది', 'లీజు' (lease) or 'అద్దె' instead of 'కౌలు', 'కేసు' (case) instead of 'అభియోగం'.\n"
                "Preserve spacing, punctuation, names, and key numbers. Only output the translated text.\n\n"
                f"Text to translate:\n{text_to_translate}"
            )
            translated = generate_text_completion(
                system_prompt="You are a translation assistant that translates English legal notices into simple, comforting, everyday Telugu for ordinary citizens.",
                user_prompt=translation_prompt
            )
        except Exception:
            translated = f"[తెలుగు అనువాదం] {text_to_translate}"
            
    return {
        "original_text": text_to_translate,
        "target_language": lang,
        "translated_text": translated
    }

@app.get("/api/calendar")
async def generate_calendar_event(date: str, title: str):
    ics_content = (
        "BEGIN:VCALENDAR\n"
        "VERSION:2.0\n"
        "PRODID:-//NyayaMitra//Milestone Calendar//EN\n"
        f"BEGIN:VEVENT\n"
        f"SUMMARY:{title}\n"
        f"DTSTART;VALUE=DATE:{date.replace('-', '')}\n"
        f"DTEND;VALUE=DATE:{date.replace('-', '')}\n"
        "DESCRIPTION:NyayaMitra AI Legal Notice Deadline Reminder.\n"
        "END:VEVENT\n"
        "END:VCALENDAR\n"
    )
    return {
        "message": f"Calendar event file generated for {title}",
        "ics_file_content": ics_content
    }

@app.get("/api/laws/categories")
async def get_laws_categories(db: Session = Depends(get_db)):
    try:
        categories = db.query(models.Law.category).distinct().all()
        category_list = [c[0] for c in categories if c[0]]
        category_list.sort()
        return category_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch categories: {str(e)}")

# Static mapping for category translations
TELUGU_CATEGORIES = {
    "Tenancy & Housing": "అద్దె & గృహనిర్మాణం",
    "Finance & Banking": "ఆర్థికం & బ్యాంకింగ్",
    "Consumer Rights": "వినియోగదారుల హక్కులు",
    "Utilities & Services": "సౌకర్యాలు & సేవలు",
    "Civil, Personal & Cyber": "సివిల్, వ్యక్తిగత & సైబర్",
    "Bank Statements & Finance": "బ్యాంక్ స్టేట్‌మెంట్లు & ఆర్థికం",
    "Employee Disputes": "ఉద్యోగుల వివాదాలు",
    "Family Law Matters": "కుటుంబ చట్ట వ్యవహారాలు",
    "Inheritance & Heir Disputes": "వారసత్వ & వారసుల వివాదాలు",
    "Land Disputes": "భూమి వివాదాలు"
}

@app.get("/api/laws")
async def get_laws(category: Optional[str] = None, language: Optional[str] = "english", db: Session = Depends(get_db)):
    try:
        query = db.query(models.Law)
        is_telugu = language and language.lower() == "telugu"
        
        # Always map Telugu category name back to English for querying database
        english_category = category
        if category:
            for eng_cat, tel_cat in TELUGU_CATEGORIES.items():
                if category == tel_cat or category == eng_cat:
                    english_category = eng_cat
                    break
        
        if english_category and english_category.lower() != "all":
            query = query.filter(models.Law.category == english_category)
        laws = query.all()
        
        formatted_laws = []
        
        # Fetch translations if telugu
        translations_map = {}
        if is_telugu:
            translations = db.query(models.LawTranslation).all()
            translations_map = {t.law_id: t for t in translations}
        
        for l in laws:
            try:
                remedies = json.loads(l.remedies_json)
            except Exception:
                remedies = []
            try:
                keywords = json.loads(l.keywords_json)
            except Exception:
                keywords = []
                
            # Handle Telugu details falling back to English if not ready
            act_text = l.act
            scope_text = l.scope
            summary_text = l.summary
            details_text = l.details
            remedies_list = remedies
            
            if is_telugu:
                t = translations_map.get(l.id)
                if t:
                    act_text = t.act_telugu or l.act
                    scope_text = t.scope_telugu or l.scope
                    summary_text = t.summary_telugu or l.summary
                    details_text = t.details_telugu or l.details
                    if t.remedies_telugu_json:
                        try:
                            remedies_list = json.loads(t.remedies_telugu_json)
                        except Exception:
                            pass
            
            category_text = l.category
            if is_telugu:
                category_text = TELUGU_CATEGORIES.get(l.category, l.category)

            formatted_laws.append({
                "id": l.id,
                "act": act_text,
                "category": category_text,
                "scope": scope_text,
                "summary": summary_text,
                "details": details_text,
                "remedies": remedies_list,
                "keywords": keywords
            })
        return formatted_laws
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch laws: {str(e)}")

# Background Telugu Translation Loop
def translate_laws_in_background():
    print(">>> [NyayaMitra] Initializing database session for translation daemon... <<<", flush=True)
    # Wait 2 seconds for server boot-up
    time.sleep(2)
    db = SessionLocal()
    try:
        print(">>> [NyayaMitra] Starting translation loop... <<<", flush=True)
        
        # Prioritize visible category laws first (4 for All, 2 per category)
        all_visible_ids = [l.id for l in db.query(models.Law.id).limit(4).all()]
        
        categories = [c[0] for c in db.query(models.Law.category).distinct().all()]
        for cat in categories:
            cat_ids = [l.id for l in db.query(models.Law.id).filter(models.Law.category == cat).limit(2).all()]
            all_visible_ids.extend(cat_ids)
            
        # De-duplicate while preserving prioritization order
        visible_ids = []
        for vid in all_visible_ids:
            if vid not in visible_ids:
                visible_ids.append(vid)
        
        logger.info(f"Targeting {len(visible_ids)} laws for translation (prioritizing visible ones first).")
        
        for law_id in visible_ids:
            # Check if already translated
            translated = db.query(models.LawTranslation).filter(models.LawTranslation.law_id == law_id).first()
            if translated:
                continue
                
            law = db.query(models.Law).filter(models.Law.id == law_id).first()
            if not law:
                continue
                
            try:
                remedies = []
                try:
                    remedies = json.loads(law.remedies_json)
                except Exception:
                    pass
                
                # Single call to translate all text fields to valid JSON
                translation_prompt = f"""
                You are a legal translator translating technical English notices into simple, everyday, conversational Telugu ("వాడుక భాష") for ordinary citizens.
                Do NOT use complex Sanskritized legal words. Use common everyday words that a regular citizen uses in daily conversation.
                If a term is common in English, write it in Telugu script (transliterate) rather than using a complex Telugu word (e.g. use "నోటీసు" instead of "నోటిఫికేషన్", "కోర్టు" instead of "న్యాయస్థానం", "లాయర్" instead of "న్యాయవాది", "లీజు" or "అద్దె" instead of "కౌలు", "కేసు" instead of "అభియోగం").
                
                Translate the following law entry:
                Act: {law.act}
                Scope: {law.scope}
                Summary: {law.summary}
                Details: {law.details}
                Remedies: {json.dumps(remedies)}
                
                You MUST output your response as a valid JSON object with the exact keys: "act", "scope", "summary", "details", "remedies". Do not add markdown backticks or any extra text around the JSON object. Just return the raw JSON object string.
                """
                
                print(f">>> [NyayaMitra] Translating law ID {law.id} ({law.act})... <<<", flush=True)
                res = generate_text_completion(
                    system_prompt="You are a legal translation assistant that outputs raw JSON format.",
                    user_prompt=translation_prompt
                )
                
                # Parse response safely using regex to extract JSON block
                res_clean = res.strip()
                import re
                try:
                    data = json.loads(res_clean)
                except Exception:
                    # Fallback: find anything between the first '{' and last '}'
                    match = re.search(r'\{.*\}', res_clean, re.DOTALL)
                    if match:
                        try:
                            data = json.loads(match.group(0))
                        except Exception as parse_err:
                            raise ValueError(f"Found JSON block but failed to parse: {str(parse_err)}. Raw: {res_clean[:200]}")
                    else:
                        raise ValueError(f"No JSON block found in response. Raw: {res_clean[:200]}")
                
                # Update DB record by creating a LawTranslation entry
                db_trans = models.LawTranslation(
                    law_id=law.id,
                    act_telugu=data.get("act"),
                    scope_telugu=data.get("scope"),
                    summary_telugu=data.get("summary"),
                    details_telugu=data.get("details"),
                    remedies_telugu_json=json.dumps(data.get("remedies", []))
                )
                db.add(db_trans)
                try:
                    db.commit()
                    print(f">>> [NyayaMitra] Successfully translated law ID {law.id}. <<<", flush=True)
                except Exception as db_err:
                    db.rollback()
                    logger.warning(f"Translation conflict or duplicate for law id {law.id}, skipped. Details: {str(db_err)}")
                
                # Sleep for 2.5 seconds to keep translation speed high while avoiding massive bursts
                time.sleep(2.5)
            except Exception as e:
                db.rollback()
                print(f">>> [NyayaMitra] Failed to translate law ID {law.id}: {str(e)} <<<", flush=True)
                # If we hit quota (429), sleep for 30 seconds
                if "429" in str(e) or "quota" in str(e).lower() or "limit" in str(e).lower():
                    print(">>> [NyayaMitra] Rate limit reached. Cooling down for 30 seconds... <<<", flush=True)
                    time.sleep(30.0)
                else:
                    time.sleep(4.0)
        
        print(">>> [NyayaMitra] All targeted visible laws have been successfully translated! <<<", flush=True)
    except Exception as err:
        print(f">>> [NyayaMitra] Error in background translator: {str(err)} <<<", flush=True)
    finally:
        db.close()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
