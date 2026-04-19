import streamlit as st
import pandas as pd
import sqlite3
from google import genai
from google.genai import types
import json
import time

# =========================
# CONFIG
# =========================
gemini_api_key = st.secrets["gemini_api_key"]
gmn_client = genai.Client(api_key=gemini_api_key)

db_name = "test_database.db"
data_table = "transactions"

data_dict_text = """
- trx_date: วันที่ทำธุรกรรม
- trx_no: หมายเลขธุรกรรม
- member_code: รหัสสมาชิกของลูกค้า
- branch_code: รหัสสาขา
- branch_region: ภูมิภาคที่สาขาตั้งอยู่
- branch_province: จังหวัดที่สาขาตั้งอยู่
- product_code: รหัสสินค้า
- product_category: หมวดหมู่หลักของสินค้า
- product_group: กลุ่มของสินค้า
- product_type: ประเภทของสินค้า
- order_qty: จำนวนชิ้น/หน่วย ที่ลูกค้าสั่งซื้อ
- unit_price: ราคาขายของสินค้าต่อ 1 หน่วย
- cost: ต้นทุนของสินค้าต่อ 1 หน่วย
- item_discount: ส่วนลดเฉพาะรายการสินค้านั้น ๆ
- customer_discount: ส่วนลดจากสิทธิของลูกค้า
- net_amount: ยอดขายสุทธิของรายการนั้น
- cost_amount: ต้นทุนรวมของรายการนั้น
"""

# =========================
# HELPER FUNCTIONS
# =========================
def detect_language(text: str) -> str:
    """Detect main language from user question. Returns 'th' or 'en'."""
    thai_chars = sum("\u0E00" <= ch <= "\u0E7F" for ch in text)
    english_chars = sum(("a" <= ch.lower() <= "z") for ch in text)

    if thai_chars > english_chars:
        return "th"
    return "en"


def get_localized_message(key: str, language: str) -> str:
    """Return system/fallback messages in the same language as user question."""
    messages = {
        "th": {
            "sql_error": "ขออภัย ไม่สามารถสร้างคำสั่ง SQL ได้",
            "no_data": "ไม่พบข้อมูลที่ตรงกับคำถาม",
            "quota_error": "ขออภัย ขณะนี้ Gemini API เกินโควต้าชั่วคราว กรุณาลองใหม่อีกครั้งในอีกสักครู่",
            "processing": "กำลังหาคำตอบ...",
            "chat_placeholder": "พิมพ์คำถามที่นี่...",
            "db_error_prefix": "เกิดข้อผิดพลาดจากฐานข้อมูล",
            "ai_error_prefix": "เกิดข้อผิดพลาดจาก AI",
        },
        "en": {
            "sql_error": "Sorry, I could not generate the SQL query.",
            "no_data": "No data matched your question.",
            "quota_error": "Sorry, the Gemini API quota has been temporarily exceeded. Please try again shortly.",
            "processing": "Finding the answer...",
            "chat_placeholder": "Type your question here...",
            "db_error_prefix": "Database error",
            "ai_error_prefix": "AI error",
        }
    }
    return messages.get(language, messages["en"]).get(key, key)


def query_to_dataframe(sql_query: str, database_name: str):
    """Run SQL query and return result as DataFrame."""
    connection = None
    try:
        connection = sqlite3.connect(database_name)
        result_df = pd.read_sql_query(sql_query, connection)
        return result_df
    except Exception as e:
        return f"Database Error: {e}"
    finally:
        if connection is not None:
            connection.close()


def generate_gemini_answer(prompt: str, is_json: bool = False, language: str = "en") -> str:
    """Call Gemini API and return text response with simple retry for quota issues."""
    config = types.GenerateContentConfig(
        response_mime_type="application/json" if is_json else "text/plain"
    )

    max_retries = 3
    wait_seconds = 15

    for attempt in range(max_retries):
        try:
            response = gmn_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=config
            )
            return response.text

        except Exception as e:
            error_text = str(e)

            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                if attempt < max_retries - 1:
                    time.sleep(wait_seconds)
                    continue
                return get_localized_message("quota_error", language)

            return f"{get_localized_message('ai_error_prefix', language)}: {e}"

    return get_localized_message("quota_error", language)


# =========================
# PROMPT TEMPLATES
# =========================
script_prompt = """
### Goal
Create the shortest and most correct SQLite script to answer the user's question from the available data.
Return JSON only.

### Context
You are an SQLite Master working inside an automated system (Strict JSON API).
Do not answer with explanations.
Return only executable SQL wrapped in JSON.

### Input
- User question: <Question> {question} </Question>
- Table name: <Table_Name> {table_name} </Table_Name>
- Column schema: <Schema>
{data_dict}
</Schema>

### Process
1. Analyze the query from <Question> and <Schema>
2. If any date column is involved, always use SQLite date functions such as `date()` or `strftime()`
3. Write concise SQL focused only on the requested answer
4. Do not guess the result; generate only the SQL needed to retrieve the answer from the real database

### Output
Return exactly one JSON object in this format only:
{{"script": "SELECT ... FROM ..."}}

Do not return markdown.
Do not return explanation text.
"""

answer_prompt = """
### Goal
Summarize the result from the data and answer the user's question accurately, concisely, and naturally.

### Context
You are a Data Analyst summarizing a DataFrame result for the user.
Keep the answer short and precise.
Focus on correct numeric interpretation.

### Input
- User question: <Question> {question} </Question>
- Response language: <Language> {language} </Language>
- Data from DataFrame: <Raw_Data>
{raw_data}
</Raw_Data>

### Language Rule
1. You must answer only in the language specified in <Language>
2. If <Language> is "th", answer in Thai only
3. If <Language> is "en", answer in English only
4. Do not mix languages unless a column name, proper noun, code, or raw value must remain unchanged
5. If the user question is mixed-language, follow the main language indicated by <Language>

### Process
1. Analyze <Raw_Data> so it answers <Question>
2. Summarize only the most relevant result
3. Format numbers with comma separators for thousands
4. Use no more than 2 decimal places when needed
5. Add a suitable unit when the context clearly implies one
6. If there is only one value, answer directly without unnecessary explanation

### Output
Return only the final answer text.
Do not return markdown.
Do not return bullet points.
Do not return bilingual text.
"""

# =========================
# CORE LOGIC
# =========================
def generate_summary_answer(user_question: str) -> str:
    language = detect_language(user_question)

    # 1) Generate SQL from user question
    script_prompt_input = script_prompt.format(
        question=user_question,
        table_name=data_table,
        data_dict=data_dict_text
    )

    sql_json_text = generate_gemini_answer(
        script_prompt_input,
        is_json=True,
        language=language
    )

    if sql_json_text.startswith("AI Error:") or sql_json_text.startswith("เกิดข้อผิดพลาดจาก AI"):
        return sql_json_text

    if sql_json_text == get_localized_message("quota_error", language):
        return sql_json_text

    try:
        sql_script = json.loads(sql_json_text)["script"]
    except Exception:
        if language == "th":
            return f"{get_localized_message('sql_error', language)}\n\nผลลัพธ์ที่ได้:\n{sql_json_text}"
        return f"{get_localized_message('sql_error', language)}\n\nReturned result:\n{sql_json_text}"

    # 2) Query database
    df_result = query_to_dataframe(sql_script, db_name)

    if isinstance(df_result, str):
        if language == "th":
            return f"{get_localized_message('db_error_prefix', language)}: {df_result}"
        return f"{get_localized_message('db_error_prefix', language)}: {df_result}"

    if df_result.empty:
        return get_localized_message("no_data", language)

    # 3) Generate natural language answer
    answer_prompt_input = answer_prompt.format(
        question=user_question,
        language=language,
        raw_data=df_result.to_string(index=False)
    )

    final_answer = generate_gemini_answer(
        answer_prompt_input,
        is_json=False,
        language=language
    )

    return final_answer


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="Gemini Chat with Database",
    page_icon="💬",
    layout="centered"
)

st.title("Gemini Chat with Database")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

ui_language = detect_language("".join(
    [m["content"] for m in st.session_state.messages if m["role"] == "user"][-3:]
)) if st.session_state.messages else "th"

if prompt := st.chat_input(get_localized_message("chat_placeholder", ui_language)):
    user_language = detect_language(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(get_localized_message("processing", user_language)):
            response = generate_summary_answer(prompt)
            st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
