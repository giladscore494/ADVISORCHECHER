# -*- coding: utf-8 -*-
# =========================================
# Car Advisor – גרסה סופית (Modern Blue)
# שאלון 5 שלבים • הסברים בעברית • ללא תמונות
# =========================================

import streamlit as st
import pandas as pd
import json, os, uuid, re, ast
from datetime import datetime
import numpy as np
import google.generativeai as genai

# --------------------------------------------------
# עיצוב כללי
# --------------------------------------------------
st.set_page_config(page_title="Car Advisor", page_icon="🚗", layout="wide")

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Rubik:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
:root { --primary:#2259b4; --accent:#ff7a18; --ink:#0f172a; --sub:#64748b; }
html, body, [class*="css"] { font-family: 'Inter','Rubik',system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial !important; }
h1,h2,h3 { color: var(--ink) }
.step { background:#fff; border-radius:16px; box-shadow:0 10px 24px rgba(0,0,0,.06); padding:18px; margin-bottom:20px; }
.pill { display:inline-block; background:#eef2ff; color:#273c75; border-radius:9999px; padding:2px 10px; font-weight:600; margin-right:6px;}
.disclaimer { color:#a16207; background:#fffbeb; border:1px solid #fde68a; padding:8px 12px; border-radius:10px; }
.logo { height: 42px; margin-right:8px; vertical-align:middle; }
.topbar { display:flex; align-items:center; gap:10px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# שכבת Smart JSON Repair (תוספת)
# --------------------------------------------------

# נסיון להשתמש ב-json_repair אם מותקן; אם לא – נשתמש בתיקונים פנימיים
_JSON_REPAIR_AVAILABLE = False
try:
    from json_repair import repair_json as _json_repair_lib
    _JSON_REPAIR_AVAILABLE = True
except Exception:
    _JSON_REPAIR_AVAILABLE = False

def _strip_code_fences(text: str) -> str:
    """מסיר גדרות ```...``` או ```json ...``` אם קיימות."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json|JSON)?\s*", "", t, count=1).strip()
        t = re.sub(r"\s*```$", "", t).strip()
    return t

def _extract_largest_braced_block(text: str) -> str:
    """מחלץ את הבלוק הגדול ביותר שמתחיל ב-{ ומסתיים ב-} עם איזון סוגריים."""
    start_idxs = [m.start() for m in re.finditer(r"\{", text)]
    best = ""
    for s in start_idxs:
        depth = 0
        for i in range(s, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[s:i+1]
                    if len(candidate) > len(best):
                        best = candidate
                    break
    return best or text

def _basic_json_heuristics_fix(s: str) -> str:
    """תיקונים קטנים: פסיקים מיותרים לפני סוגר, החלפת גרש בודד במרכאות היכן שנכון, וכו' – בזהירות."""
    # הסרת פסיקים עודפים לפני } או ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # לפעמים יש מפתחות ללא מרכאות: נעטוף אותם (פשוט וזהיר: לפני : ובין תווים חוקיים)
    s = re.sub(r'(?m)^\s*([A-Za-z_][A-Za-z0-9_\-]*)\s*:', r'"\1":', s)
    s = re.sub(r'(?<!")([A-Za-z_][A-Za-z0-9_\-]*)\s*:', r'"\1":', s)
    # החלפת גרשים בודדים למרכאות כאשר זה נראה כמו מחרוזת (פשוט; עדיין נשמור על מספרים/בוליאנים)
    # נבצע רק בתוך ערכים אחרי נקודתיים כאשר אין מרכאות כבר.
    def _fix_single_quotes(m):
        inner = m.group(1)
        if '"' in inner:
            return f': {inner}'
        # אם הערך נראה כמו true/false/null/מספר – לא להחליף
        low = inner.strip().lower()
        if low in ("true","false","null"):
            return f": {low}"
        try:
            float(inner.strip())
            return f": {inner}"
        except Exception:
            pass
        return f': "{inner.strip()}"'
    s = re.sub(r':\s*\'([^\']*)\'', _fix_single_quotes, s)
    return s

def smart_extract_json_text(raw_text: str) -> str:
    """
    מוציא JSON קריא ככל האפשר מתוך טקסט חופשי:
    - מוריד גדרות קוד
    - מחפש בלוק מאוזן גדול עם {}
    - תיקוני תחביר קטנים
    """
    if not isinstance(raw_text, str):
        return ""
    text = _strip_code_fences(raw_text)
    text = text.strip()
    # לעתים המודל מחזיר הסבר + json. ננסה לחלץ הבלוק הגדול ביותר.
    block = _extract_largest_braced_block(text)
    if not block.strip().startswith("{"):
        # fallback: נסה למצוא json מהרישא עד סוף הבלוק הראשון
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            block = m.group(0)
        else:
            block = text
    # תיקונים קלים
    block = _basic_json_heuristics_fix(block)
    return block.strip()

def repair_json_text(raw_text: str) -> str:
    """
    אם json_repair קיים – נשתמש בו.
    אחרת – נחזיר את התוצאה אחרי smart_extract_json_text (ועוד תיקונים קטנים).
    """
    candidate = smart_extract_json_text(raw_text)
    if _JSON_REPAIR_AVAILABLE:
        try:
            # json_repair מחזיר טקסט JSON תקין.
            return _json_repair_lib(candidate)
        except Exception:
            pass
    # שיפורים זעירים נוספים: איזון סוגריים
    # ננסה לאזן סוגריים מסולסלים בלבד (פשוט)
    open_cnt = candidate.count("{")
    close_cnt = candidate.count("}")
    if close_cnt < open_cnt:
        candidate += "}" * (open_cnt - close_cnt)
    elif open_cnt < close_cnt:
        candidate = ("{" * (close_cnt - open_cnt)) + candidate
    return candidate

def _coerce_numeric(value):
    """ממיר למספר אם אפשר (int/float); אחרת מחזיר None או את הערך המקורי אם הוא בוליאני."""
    if isinstance(value, (int, float, bool)):
        return value
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return None

def _validate_and_coerce_result(parsed: dict) -> dict:
    """
    בדיקות רכות והמרות קלות לשדות מספריים.
    מחזיר את המבנה (יתכן עם coercion).
    """
    if not isinstance(parsed, dict):
        return {}
    cars = parsed.get("recommended_cars", [])
    if not isinstance(cars, list):
        parsed["recommended_cars"] = []
        return parsed

    numeric_fields = [
        "avg_fuel_consumption", "annual_fee", "reliability_score", "maintenance_cost",
        "safety_rating", "insurance_cost", "resale_value", "performance_score",
        "comfort_features", "suitability"
    ]

    for car in cars:
        if not isinstance(car, dict):
            continue
        # ודא מפתחות בסיסיים קיימים גם אם חסרים
        for key in ["brand","model","year","fuel","gear","turbo","engine_cc","price_range_nis","market_supply"]:
            car.setdefault(key, "")
        # המרות לשדות מספריים
        for f in numeric_fields:
            if f in car:
                car[f] = _coerce_numeric(car.get(f))
        # המרת turbo ל-Bool במידת האפשר (אם בא עברית/טקסט)
        if "turbo" in car:
            t = car.get("turbo")
            if isinstance(t, str):
                t_norm = t.strip()
                if t_norm in ("כן","yes","true","True"):
                    car["turbo"] = True
                elif t_norm in ("לא","no","false","False"):
                    car["turbo"] = False
        # year כ-int אם ניתן
        if "year" in car:
            try:
                car["year"] = int(str(car["year"]).strip())
            except Exception:
                pass
    return parsed

def parse_gemini_output(raw_text: str) -> dict:
    """
    שכבת פרסינג עמידה:
    1) נסה json.loads ישיר.
    2) נסה json_repair (אם קיים) או תיקונים הורסיטיים.
    3) נסה ast.literal_eval כגיבוי (אם המודל החזיר מבנה dict-like בפייתון).
    4) החזרת {} אם הכל נכשל.
    כולל ולידציה רכה והמרות.
    """
    if not raw_text:
        return {}
    # ניסיון ראשון – כמו שהוא
    try:
        d = json.loads(raw_text)
        return _validate_and_coerce_result(d)
    except Exception:
        pass

    # ניסיון שני – תיקון
    try:
        fixed_text = repair_json_text(raw_text)
        d2 = json.loads(fixed_text)
        return _validate_and_coerce_result(d2)
    except Exception:
        pass

    # ניסיון שלישי – לעיתים המודל מחזיר dict-like של פייתון
    try:
        candidate = smart_extract_json_text(raw_text)
        d3 = ast.literal_eval(candidate)
        if isinstance(d3, dict):
            return _validate_and_coerce_result(d3)
    except Exception:
        pass

    # נכשל – נחזיר ריק
    return {}

# --------------------------------------------------
# פונקציות עזר (הלוגיקה המקורית נשמרה)
# --------------------------------------------------
def init_state():
    for key in ["user_profile","validated_cars","methods_info","fuel_price","electricity_price","ui_step","results_df","search_info"]:
        if key not in st.session_state:
            st.session_state[key] = None
    if st.session_state.ui_step is None:
        st.session_state.ui_step = 0

def make_user_profile(budget_min, budget_max, years_range, fuels, gears,
                      turbo_required, main_use, annual_km, driver_age,
                      family_size, cargo_need, safety_required,
                      trim_level, weights, body_style, driving_style, excluded_colors):
    return {
        "budget_nis": [float(budget_min), float(budget_max)],
        "years": [int(years_range[0]), int(years_range[1])],
        "fuel": [f.lower() for f in fuels],
        "gear": [g.lower() for g in gears],
        "turbo_required": None if turbo_required == "any" else (turbo_required == "yes"),
        "main_use": main_use.strip(),
        "annual_km": int(annual_km),
        "driver_age": int(driver_age),
        "family_size": family_size,
        "cargo_need": cargo_need,
        "safety_required": safety_required,
        "trim_level": trim_level,
        "weights": weights,
        "body_style": body_style,
        "driving_style": driving_style,
        "excluded_colors": excluded_colors,
    }

def clean_gemini_output(cars_raw):
    records, methods = [], []
    for car in cars_raw:
        if not isinstance(car, dict):
            continue
        record, method = {}, {}
        for k, v in car.items():
            if k.endswith("_method"):
                method[k] = v
            else:
                record[k] = v
        records.append(record)
        methods.append(method)
    return pd.DataFrame(records), methods

def normalize_car_values(df):
    if "fuel" in df.columns:
        df["fuel"] = df["fuel"].replace({
            "בנזין": "gasoline",
            "דיזל": "diesel",
            "היברידי": "hybrid",
            "דיזל היברידי": "hybrid-diesel",
            "חשמלי": "electric"
        })
    if "gear" in df.columns:
        df["gear"] = df["gear"].replace({
            "אוטומטי": "automatic",
            "אוטומטית": "automatic",
            "אוטומטי (DSG7)": "automatic",
            "אוטומטי (TCT)": "automatic",
            "אוטומטי (רובוטי)": "automatic",
            "ידני": "manual",
            "ידנית": "manual"
        })
    if "turbo" in df.columns:
        df["turbo"] = df["turbo"].replace({"כן": True, "לא": False, True: True, False: False})
    return df

# --------------------------------------------------
# מיפויים
# --------------------------------------------------
fuel_map = {"בנזין":"gasoline","היברידי":"hybrid","דיזל היברידי":"hybrid-diesel","דיזל":"diesel","חשמלי":"electric"}
gear_map = {"אוטומטית":"automatic","ידנית":"manual"}
turbo_map = {"לא משנה":"any","כן":"yes","לא":"no"}

fuel_map_he = {v:k for k,v in fuel_map.items()}
gear_map_he = {v:k for k,v in gear_map.items()}
turbo_map_he = {"yes":"כן","no":"לא","any":"לא משנה",True:"כן",False:"לא"}

column_map_he = {
    "brand":"מותג","model":"דגם","year":"שנה","fuel":"דלק","gear":"תיבה","turbo":"טורבו","engine_cc":"נפח מנוע (סמ\"ק)",
    "price_range_nis":"טווח מחיר (₪)","avg_fuel_consumption":"צריכת דלק ממוצעת (ק\"מ/ל')","annual_fee":"אגרה שנתית (₪)",
    "annual_energy_cost":"עלות דלק שנתית (₪)","total_annual_cost":"עלות כוללת שנתית (₪)",
    "reliability_score":"אמינות","maintenance_cost":"עלות אחזקה (₪/שנה)","safety_rating":"בטיחות",
    "insurance_cost":"עלות ביטוח (₪/שנה)","resale_value":"שמירת ערך","performance_score":"ביצועים",
    "comfort_features":"נוחות","suitability":"התאמה","market_supply":"היצע בשוק"
}

method_map_he = {
    "fuel_method":"שיטת חישוב צריכת דלק/חשמל","fee_method":"שיטת חישוב אגרה","reliability_method":"שיטת חישוב אמינות",
    "maintenance_method":"שיטת חישוב עלות אחזקה","safety_method":"שיטת חישוב בטיחות","insurance_method":"שיטת חישוב ביטוח",
    "resale_method":"שיטת חישוב שמירת ערך","performance_method":"שיטת חישוב ביצועים","comfort_method":"שיטת חישוב נוחות",
    "suitability_method":"שיטת חישוב התאמה","supply_method":"שיטת קביעת היצע"
}

# --------------------------------------------------
# Init + Header
# --------------------------------------------------
def topbar():
    st.markdown(
        '<div class="topbar"><img src="https://em-content.zobj.net/source/microsoft-teams/363/automobile_1f697.png" class="logo"/>'
        '<div><div style="font-weight:700;color:#0f172a;font-size:22px;">Car Advisor</div>'
        '<div style="color:#64748b;font-size:13px;">ייעוץ רכב • Smart Wizard</div></div>'
        '<span class="pill">Modern</span><span class="pill">Fast</span><span class="pill">Clean</span></div>',
        unsafe_allow_html=True)
    st.markdown("---")

init_state()
for _k in ["_step1","_step2","_step3","_step4"]:
    if _k not in st.session_state:
        st.session_state[_k] = None
topbar()

# --------------------------------------------------
# פונקציית ניווט (מזהים ייחודיים כדי למנוע DuplicateKey)
# --------------------------------------------------
def nav_buttons(left_label="חזור", right_label="הבא",
                left_action=None, right_action=None,
                show_left=True, show_right=True):
    c1,c2 = st.columns([1,1])
    with c1:
        if show_left:
            st.button(left_label, on_click=left_action, key=f"back_{st.session_state.ui_step}_{uuid.uuid4().hex}")
    with c2:
        if show_right:
            st.button(right_label, on_click=right_action, key=f"next_{st.session_state.ui_step}_{uuid.uuid4().hex}")

# --------------------------------------------------
# שלב 0 – פתיחה
# --------------------------------------------------
if st.session_state.ui_step == 0:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.subheader("ברוך הבא ל-Car Advisor")
    st.write("מצא את הרכב המתאים לך בקלות. לחיצה על 'התחל' תוביל אותך לשאלון קצר.")
    st.markdown('</div>', unsafe_allow_html=True)
    def go_next(): st.session_state.ui_step = 1
    nav_buttons(show_left=False, right_label="התחל", right_action=go_next)

# --------------------------------------------------
# שלב 1 – פרטים בסיסיים
# --------------------------------------------------
elif st.session_state.ui_step == 1:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 1: פרטים בסיסיים")
    c1, c2, c3 = st.columns([1,1,1])
    with c1: budget_min = st.number_input("תקציב מינימום (₪)", min_value=0, step=1000, value=40000)
    with c2: budget_max = st.number_input("תקציב מקסימום (₪)", min_value=0, step=1000, value=65000)
    with c3:
        ymin, ymax = st.columns(2)
        with ymin: year_min = st.number_input("שנתון מינימום", 1990, datetime.now().year, 2015)
        with ymax: year_max = st.number_input("שנתון מקסימום", 1990, datetime.now().year, 2019)
    fuels_he = st.multiselect("סוגי דלק מועדפים", list(fuel_map.keys()), default=["בנזין"])
    if "חשמלי" in fuels_he:
        st.info("נבחר דלק חשמלי — תיבת ההילוכים תוגדר כ'אוטומטית'.")
        gears_he = ["אוטומטית"]
    else:
        gears_he = st.multiselect("תיבת הילוכים", list(gear_map.keys()), default=["אוטומטית"])
    turbo_choice_he = st.selectbox("טורבו?", list(turbo_map.keys()), index=1)
    st.session_state._step1 = dict(
        budget_min=budget_min, budget_max=budget_max,
        year_min=year_min, year_max=year_max,
        fuels_he=fuels_he, gears_he=gears_he, turbo_choice_he=turbo_choice_he
    )
    st.markdown('</div>', unsafe_allow_html=True)
    def back(): st.session_state.ui_step = 0
    def next(): st.session_state.ui_step = 2
    nav_buttons(left_label="חזור", right_label="הבא", left_action=back, right_action=next)

# --------------------------------------------------
# שלב 2 – שימוש וסגנון
# --------------------------------------------------
elif st.session_state.ui_step == 2:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 2: שימוש וסגנון")
    c4, c5, c6 = st.columns([2,1,1])
    with c4:
        main_use = st.text_area("תיאור הרכב והשימוש בו", value="נסיעה יומיומית לעבודה וטיולים קצרים", height=100)
    with c5:
        annual_km = st.number_input("נסועה שנתית (ק״מ)", min_value=0, step=1000, value=15000)
    with c6:
        driver_age = st.number_input("גיל נהג", min_value=16, max_value=100, value=21)
    c6a, c6b = st.columns(2)
    with c6a: license_years = st.number_input("וותק רישיון (שנים)", min_value=0, max_value=50, value=2)
    with c6b: driver_gender = st.selectbox("מין נהג", ["זכר", "נקבה"])
    cstyle1, cstyle2, cseats = st.columns([1,1,1])
    with cstyle1: body_style = st.selectbox("סגנון מרכב מועדף", ["כללי","סדאן","האצ'בק","קרוסאובר/ג'יפון"])
    with cstyle2: driving_style = st.selectbox("סגנון נהיגה", ["רגוע ונינוח","דינמי וספורטיבי"])
    with cseats: seats_choice = st.selectbox("מספר מקומות", ["4","5","5+"] )
    excluded_colors = st.text_input("צבעים לפסילה (מופרדים בפסיק)", value="").split(",")
    st.session_state._step2 = dict(
        main_use=main_use, annual_km=annual_km, driver_age=driver_age,
        license_years=license_years, driver_gender=driver_gender,
        body_style=body_style, driving_style=driving_style,
        seats_choice=seats_choice, excluded_colors=excluded_colors
    )
    st.markdown('</div>', unsafe_allow_html=True)
    def back(): st.session_state.ui_step = 1
    def next(): st.session_state.ui_step = 3
    nav_buttons(left_label="חזור", right_label="הבא", left_action=back, right_action=next)

# --------------------------------------------------
# שלב 3 – סדר עדיפויות
# --------------------------------------------------
elif st.session_state.ui_step == 3:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 3: סדר עדיפויות")
    st.markdown("#### בחר דירוג לכל קטגוריה (1–5)")
    reliability_weight = st.slider("אמינות", 1, 5, 5)
    resale_weight = st.slider("שמירת ערך", 1, 5, 3)
    fuel_weight = st.slider("חיסכון בדלק", 1, 5, 4)
    performance_weight = st.slider("ביצועים", 1, 5, 2)
    comfort_weight = st.slider("נוחות", 1, 5, 3)
    weights = {
        "reliability": reliability_weight,
        "resale": resale_weight,
        "fuel": fuel_weight,
        "performance": performance_weight,
        "comfort": comfort_weight,
    }
    st.session_state._step3 = dict(weights=weights)
    st.markdown('</div>', unsafe_allow_html=True)
    def back(): st.session_state.ui_step = 2
    def next(): st.session_state.ui_step = 4
    nav_buttons(left_label="חזור", right_label="הבא", left_action=back, right_action=next)

# --------------------------------------------------
# שלב 4 – פרטים נוספים
# --------------------------------------------------
elif st.session_state.ui_step == 4:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 4: פרטים נוספים")
    insurance_history = st.text_input("עבר ביטוחי", value="שנתיים ללא תביעות")
    violations = st.selectbox("דוחות/שלילות", ["אין", "שלילה בעבר", "נקודות פעילות"])
    cfam, ccargo, csafety, ctrim = st.columns([1,1,1,1])
    with cfam: family_size = st.selectbox("גודל משפחה", ["1-2","3-4","5+"])
    with ccargo: cargo_need = st.selectbox("צורך בתא מטען", ["קטן","בינוני","גדול"])
    with csafety: safety_required = st.radio("חובה מערכות בטיחות אקטיביות?", ["כן","לא"])
    with ctrim: trim_level = st.selectbox("רמת אבזור", ["בסיסי","סטנדרטי","עשיר"])
    consider_supply = st.radio("האם להתחשב בהיצע בשוק?", ["כן","לא"], index=0)
    cfp, cep = st.columns([1,1])
    with cfp: fuel_price = st.number_input("מחיר ליטר דלק (₪)", min_value=1.0, max_value=20.0, value=7.0, step=0.1)
    with cep: electricity_price = st.number_input("מחיר חשמל לקוט״ש (₪)", min_value=0.1, max_value=5.0, value=0.65, step=0.01)
    st.session_state._step4 = dict(
        insurance_history=insurance_history, violations=violations,
        family_size=family_size, cargo_need=cargo_need, safety_required=safety_required,
        trim_level=trim_level, consider_supply=consider_supply,
        fuel_price=fuel_price, electricity_price=electricity_price
    )
    st.markdown('</div>', unsafe_allow_html=True)
    def back(): st.session_state.ui_step = 3
    def next(): st.session_state.ui_step = 5
    nav_buttons(left_label="חזור", right_label="המשך לייעוץ", left_action=back, right_action=next)

# --------------------------------------------------
# שלב 5 – ייעוץ ותוצאות
# --------------------------------------------------
elif st.session_state.ui_step == 5:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 5: קבלת ייעוץ ותוצאות")

    s1 = st.session_state._step1
    s2 = st.session_state._step2
    s3 = st.session_state._step3
    s4 = st.session_state._step4
    if not all([s1, s2, s3, s4]):
        st.error("חסרים נתונים בשלבים קודמים. חזור אחורה והשלם.")
    else:
        # בניית פרופיל (לוגיקה מקורית 1:1)
        fuels = [fuel_map[f] for f in (s1["fuels_he"] or [])]
        gears = [gear_map[g] for g in (s1["gears_he"] or [])]
        turbo_choice = turbo_map[s1["turbo_choice_he"]]
        weights = s3["weights"]

        profile = make_user_profile(
            s1["budget_min"], s1["budget_max"], [s1["year_min"], s1["year_max"]],
            fuels, gears, turbo_choice, s2["main_use"], s2["annual_km"], s2["driver_age"],
            s4["family_size"], s4["cargo_need"], s4["safety_required"], s4["trim_level"],
            weights, s2["body_style"], s2["driving_style"], s2["excluded_colors"]
        )
        profile["license_years"] = s2["license_years"]
        profile["driver_gender"] = s2["driver_gender"]
        profile["insurance_history"] = s4["insurance_history"]
        profile["violations"] = s4["violations"]
        profile["consider_market_supply"] = (s4["consider_supply"] == "כן")
        profile["fuel_price_nis_per_liter"] = s4["fuel_price"]
        profile["electricity_price_nis_per_kwh"] = s4["electricity_price"]
        profile["seats"] = s2["seats_choice"]  # מידע בלבד

        st.session_state.user_profile = profile
        st.session_state.fuel_price = s4["fuel_price"]
        st.session_state.electricity_price = s4["electricity_price"]

        api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.warning("⚠️ לא נמצא GEMINI_API_KEY בסודות או במשתני סביבה.")
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("models/gemini-2.5-pro")

            if st.button("🚀 בקש המלצות מג׳מיני"):
                with st.spinner("פונה לג׳מיני..."):
                    try:
                        prompt = f"""
Please recommend cars for an Israeli customer. Here is the user profile (JSON):
{json.dumps(profile, ensure_ascii=False, indent=2)}

Output requirements:

1) Return a SINGLE JSON object with fields: "search_performed", "search_queries", "recommended_cars".
2) search_performed: ALWAYS return True. You must use live web search (do not return False).
3) search_queries: ALWAYS return the actual queries you used.
4) recommended_cars: an array of 5–10 cars. EACH car MUST include:
   - brand, model, year, fuel, gear, turbo, engine_cc, price_range_nis
   - avg_fuel_consumption (+ fuel_method):
       * for non-EV: km per liter (number only)
       * for EV: kWh per 100 km (number only)
   - annual_fee (₪ per year, number only) +
**must return methods only in Hebrew all the methods must return in clean and easy to read
 in Hebrew.**
**before you select the chosen cars check very carefully if this specific model sell in israel and got high supply in the market do not return mistakes of models that cannot be found in israel***
 fee_method
   - reliability_score (1–10, number only) + reliability_method
   - maintenance_cost (₪/year, number only) + maintenance_method
   - safety_rating (1–10, number only) + safety_method
   - insurance_cost (₪/year, number only) + insurance_method
   - resale_value (1–10, number only) + resale_method
   - performance_score (1–10, number only) + performance_method
   - comfort_features (1–10, number only) + comfort_method
   - suitability (1–10, number only) + suitability_method
   - market_supply ("גבוה" / "בינוני" / "נמוך") + supply_method
5) IMPORTANT: All scoring fields must be numbers only (except market_supply which is categorical).
6) IMPORTANT: Only return car models that are actually sold in Israel.
"""
                        resp = model.generate_content(prompt)
                        # ----- שינוי מרכזי: שכבת Smart JSON Repair -----
                        text = resp.candidates[0].content.parts[0].text if resp and resp.candidates else ""
                        parsed = parse_gemini_output(text)
                        if not parsed:
                            st.error("⚠️ ג׳מיני החזיר פלט שלא ניתן לתקן ל-JSON תקין.")
                            if text:
                                st.code(text[:4000], language="json")
                    except Exception as e:
                        st.error(f"שגיאה בקריאת הפלט מג׳מיני: {e}")
                        parsed = {}

                if parsed and "recommended_cars" in parsed:
                    search_performed = parsed.get("search_performed", False)
                    search_queries = parsed.get("search_queries", [])
                    st.session_state.search_info = {"search_performed": search_performed, "search_queries": search_queries}
                    if search_performed and search_queries:
                        st.info("✅ בוצע חיפוש אינטרנטי לנתוני שוק עדכניים.")
                    else:
                        st.warning("⚠️ לא ברור אם בוצע חיפוש חי. ייתכן שהנתונים חלקיים.")

                    results_df, methods_info = clean_gemini_output(parsed["recommended_cars"])
                    if not results_df.empty:
                        # Normalize
                        results_df = normalize_car_values(results_df)

                        if "avg_fuel_consumption" not in results_df.columns:
                            st.error("חסר שדה avg_fuel_consumption בפלט.")
                            st.stop()

                        # חישובי אנרגיה
                        is_ev = results_df["fuel"].str.lower().eq("electric")
                        km_per_liter = results_df["avg_fuel_consumption"].where(~is_ev, np.nan).replace(0, np.nan)
                        kwh_per_100km = results_df["avg_fuel_consumption"].where(is_ev, np.nan)
                        annual_km = profile["annual_km"]
                        fuel_price = st.session_state.fuel_price or 7.0
                        elec_price = st.session_state.electricity_price or 0.65
                        fuel_cost = (annual_km / km_per_liter) * fuel_price
                        elec_cost = (annual_km / 100.0) * kwh_per_100km * elec_price
                        results_df["annual_energy_cost"] = np.where(is_ev, elec_cost, fuel_cost)
                        # נשמר לתאימות – לא נציג את annual_fuel_cost באנגלית
                        results_df["annual_fuel_cost"] = results_df["annual_energy_cost"]

                        for col in ["maintenance_cost", "insurance_cost", "annual_fee"]:
                            if col not in results_df.columns:
                                results_df[col] = 0.0

                        results_df["total_annual_cost"] = (
                            results_df["annual_energy_cost"].fillna(0) +
                            results_df["maintenance_cost"].fillna(0) +
                            results_df["insurance_cost"].fillna(0) +
                            results_df["annual_fee"].fillna(0)
                        )

                        # כותרות צריכה בעברית לפי EV/דלק
                        if results_df["fuel"].str.lower().eq("electric").any():
                            column_map_he["avg_fuel_consumption"] = "צריכת חשמל (קוט\"ש/100 ק\"מ)"
                            column_map_he["annual_energy_cost"] = "עלות חשמל שנתית (₪)"
                        else:
                            column_map_he["avg_fuel_consumption"] = "צריכת דלק ממוצעת (ק\"מ/ל')"
                            column_map_he["annual_energy_cost"] = "עלות דלק שנתית (₪)"

                        # טבלת תצוגה בעברית – בלי העמודה האנגלית
                        results_df_display = results_df.copy()
                        if "annual_fuel_cost" in results_df_display.columns:
                            results_df_display = results_df_display.drop(columns=["annual_fuel_cost"])
                        results_df_display["fuel"] = results_df_display["fuel"].map(fuel_map_he).fillna(results_df_display["fuel"])
                        results_df_display["gear"] = results_df_display["gear"].map(gear_map_he).fillna(results_df_display["gear"])
                        results_df_display["turbo"] = results_df_display["turbo"].map(turbo_map_he).fillna(results_df_display["turbo"])
                        results_df_display = results_df_display.rename(columns=column_map_he)

                        st.session_state.results_df = results_df
                        st.session_state.methods_info = methods_info
                        st.success(f"✅ התקבלו {len(results_df)} רכבים מג׳מיני.")
                    else:
                        st.error("⚠️ לא נמצאו רכבים בפלט.")
                else:
                    st.error("⚠️ לא התקבל מפתח 'recommended_cars' בפלט.")

        # הצגת תוצאות אם קיימות
        results_df = st.session_state.results_df
        methods_info = st.session_state.methods_info or []
        if results_df is not None and not results_df.empty:
            results_df_display = results_df.copy()
            # להבטיח שלא מציגים את העמודה האנגלית
            if "annual_fuel_cost" in results_df_display.columns:
                results_df_display = results_df_display.drop(columns=["annual_fuel_cost"])
            results_df_display["fuel"] = results_df_display["fuel"].map(fuel_map_he).fillna(results_df_display["fuel"])
            results_df_display["gear"] = results_df_display["gear"].map(gear_map_he).fillna(results_df_display["gear"])
            results_df_display["turbo"] = results_df_display["turbo"].map(turbo_map_he).fillna(results_df_display["turbo"])
            results_df_display = results_df_display.rename(columns=column_map_he)

            st.markdown('<div class="disclaimer">⚠️ הנתונים הם הערכה גסה של AI; יש לאמת לפני קנייה.</div>', unsafe_allow_html=True)
            st.dataframe(results_df_display.reset_index(drop=True), use_container_width=True)

            # גרף עלות כוללת שנתית
            st.markdown("### 📊 השוואת עלות כוללת שנתית")
            chart_df = results_df_display[["מותג","דגם","שנה","עלות כוללת שנתית (₪)"]].copy()
            chart_df["רכב"] = chart_df["מותג"] + " " + chart_df["דגם"] + " " + chart_df["שנה"].astype(str)
            chart_df = chart_df.set_index("רכב")
            st.bar_chart(chart_df["עלות כוללת שנתית (₪)"])

            # הסברים מפורטים בעברית לכל רכב
            st.markdown("### 📝 הסברים מפורטים לכל רכב")
            for i, row in results_df_display.iterrows():
                car_name = f"{row['מותג']} {row['דגם']} {row['שנה']}"
                with st.expander(f"📝 הסבר מפורט על {car_name}"):
                    st.caption(f"דלק: {row['דלק']} | תיבה: {row['תיבה']} | טורבו: {row['טורבו']}")
                    if "טווח מחיר (₪)" in row:
                        st.write(f"**טווח מחיר:** {row['טווח מחיר (₪)']}")
                    if "עלות כוללת שנתית (₪)" in row:
                        try:
                            st.write(f"**עלות שנתית:** {float(row['עלות כוללת שנתית (₪)']):,.0f} ₪")
                        except Exception:
                            st.write(f"**עלות שנתית:** {row['עלות כוללת שנתית (₪)']}")
                    method = methods_info[i] if i < len(methods_info) else {}
                    if method:
                        for k, v in method.items():
                            name = method_map_he.get(k, k)
                            st.write(f"- **{name}:** {v}")
                    else:
                        st.write("אין הסברים מפורטים זמינים לפריט זה.")

    st.markdown('</div>', unsafe_allow_html=True)
    def back(): st.session_state.ui_step = 4
    nav_buttons(left_label="חזור", right_label="הבא", left_action=back, right_action=None, show_right=False)
