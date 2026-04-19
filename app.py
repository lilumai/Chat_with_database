import streamlit as st
import pandas as pd
import sqlite3
from google import genai
from google.genai import types
import json

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


def generate_gemini_answer(prompt: str, is_json: bool = False) -> str:
    """Call Gemini API and return text response."""
    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json" if is_json else "text/plain"
        )

        response = gmn_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=config
        )
        return response.text
    except Exception as e:
        return f"AI Error: {e}"


# =========================
# PROMPT TEMPLATES
# =========================
script_prompt = """
### Goal
สร้าง SQLite script ที่สั้นและถูกต้องที่สุด เพื่อตอบคำถามจากข้อมูลที่มี โดยส่งออกเป็น JSON เท่านั้น

### Context
คุณคือ SQLite Master ที่ทำงานในระบบอัตโนมัติ (Strict JSON API) ห้ามตอบเป็นคำพูด ให้ตอบเฉพาะโค้ดที่ใช้งานได้จริง

### Input
- คำถามที่ผู้ใช้ต้องการคำตอบ: <Question> {question} </Question>
- ชื่อ Table ที่ต้องใช้ดึงข้อมูล: <Table_Name> {table_name} </Table_Name>
- คำอธิบายคอลัมน์: <Schema>
{data_dict}
</Schema>

### Process
1. วิเคราะห์ Query จาก <Question> และ <Schema>
2. หากมีคอลัมน์วันที่ ให้ใช้ฟังก์ชัน `date()` หรือ `strftime()` ของ SQLite จัดการเสมอ
3. เขียน SQL ให้กระชับและมุ่งเน้นเฉพาะคำตอบที่ต้องการ

### Output
ตอบกลับเป็น JSON object รูปแบบเดียวเท่านั้น:
{{"script": "SELECT ... FROM ..."}}

(ห้ามมีคำอธิบายประกอบ หรือ Markdown นอกเหนือจาก JSON)
"""

answer_prompt = """
### Goal
สรุปผลลัพธ์จากข้อมูลและตอบคำถามอย่างถูกต้อง แม่นยำ และเป็นธรรมชาติ

### Context
คุณคือ Data Analyst ที่ทำหน้าที่สรุปผลจาก DataFrame และตอบคำถามผู้ใช้แบบเจาะจง ห้ามตอบยาวเกินความจำเป็น และเน้นการวิเคราะห์เชิงตัวเลขที่ถูกต้อง

### Input
- คำถามที่ผู้ใช้ต้องการคำตอบ: <Question> {question} </Question>
- ข้อมูลจาก DataFrame: <Raw_Data>
{raw_data}
</Raw_Data>

### Process
1. วิเคราะห์ข้อมูลจาก <Raw_Data> ให้สอดคล้องกับ <Question>
2. คำนวณและสรุปข้อมูลเชิงสถิติที่สำคัญ
3. จัดรูปแบบตัวเลข: ใส่คอมม่า (,) คั่นหลักพัน และทศนิยมไม่เกิน 2 ตำแหน่ง
4. ระบุหน่วย (เช่น บาท, คน, ครั้ง, %) ต่อท้ายตัวเลขทุกครั้งตามบริบทของข้อมูล

### Output
ตอบเป็นข้อความสั้น ๆ โดยมีโครงสร้างดังนี้:
1. คำเกริ่นนำ: ใช้ประโยคสั้น ๆ เข้าประเด็นทันที
2. เนื้อหา: ระบุผลการวิเคราะห์พร้อมตัวเลขที่ใส่คอมม่าและมีหน่วยลงท้ายเสมอ
"""

# =========================
# CORE LOGIC
# =========================
def generate_summary_answer(user_question: str) -> str:
    # 1) Generate SQL from user question
    script_prompt_input = script_prompt.format(
        question=user_question,
        table_name=data_table,
        data_dict=data_dict_text
    )

    sql_json_text = generate_gemini_answer(script_prompt_input, is_json=True)

    if sql_json_text.startswith("AI Error:"):
        return sql_json_text

    try:
        sql_script = json.loads(sql_json_text)["script"]
    except Exception:
        return f"ขออภัย ไม่สามารถสร้างคำสั่ง SQL ได้\n\nผลลัพธ์ที่ได้:\n{sql_json_text}"

    # 2) Query database
    df_result = query_to_dataframe(sql_script, db_name)

    if isinstance(df_result, str):
        return df_result

    if df_result.empty:
        return "ไม่พบข้อมูลที่ตรงกับคำถาม"

    # 3) Generate natural language answer
    answer_prompt_input = answer_prompt.format(
        question=user_question,
        raw_data=df_result.to_string(index=False)
    )

    final_answer = generate_gemini_answer(answer_prompt_input, is_json=False)

    if final_answer.startswith("AI Error:"):
        return final_answer

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

if prompt := st.chat_input("พิมพ์คำถามที่นี่..."):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("กำลังหาคำตอบ..."):
            response = generate_summary_answer(prompt)
            st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
