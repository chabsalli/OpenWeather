# app.py
import os
import re
import json
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, date

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")

st.title("📊 AI 습관 트래커")
st.caption("체크인 → 달성률 확인 → 날씨/강아지/AI 코치 리포트까지 한 번에!")

# ----------------------------
# Sidebar: API Keys
# ----------------------------
with st.sidebar:
    st.header("🔑 API 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    owm_api_key = st.text_input("OpenWeatherMap API Key", type="password", placeholder="OWM Key...")

    st.divider()
    st.caption("※ 키는 세션에만 유지됩니다. (배포 시 Secrets 사용 권장)")

# ----------------------------
# Utils
# ----------------------------
HABITS = [
    ("wake", "🌅", "기상 미션"),
    ("water", "💧", "물 마시기"),
    ("study", "📚", "공부/독서"),
    ("workout", "🏃‍♂️", "운동하기"),
    ("sleep", "😴", "수면"),
]

CITIES = [
    "Seoul", "Busan", "Incheon", "Daegu", "Daejeon",
    "Gwangju", "Ulsan", "Suwon", "Jeju", "Sejong"
]

COACH_STYLES = {
    "스파르타 코치": "당신은 엄격하고 직설적인 스파르타 코치다. 칭찬은 짧게, 개선점은 날카롭게. 변명은 허용하지 않는다. 구체적 행동 지시를 준다.",
    "따뜻한 멘토": "당신은 공감과 지지를 잘하는 따뜻한 멘토다. 사용자의 감정을 존중하고, 작은 성취를 인정하며, 부담 없는 다음 कदम을 제안한다.",
    "게임 마스터": "당신은 RPG 세계관의 게임 마스터다. 습관을 퀘스트/스탯/보상으로 표현하고, 몰입감 있는 톤으로 내일의 미션을 제시한다.",
}

def _safe_request_json(url: str, timeout: int = 10):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def get_weather(city: str, api_key: str):
    """
    OpenWeatherMap 현재 날씨 (한국어, 섭씨)
    실패 시 None 반환, timeout=10
    """
    if not api_key:
        return None
    # Current weather endpoint
    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={api_key}&units=metric&lang=kr"
    )
    data = _safe_request_json(url, timeout=10)
    if not data:
        return None
    try:
        weather_main = data["weather"][0]["main"]
        weather_desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        wind = data.get("wind", {}).get("speed", None)
        return {
            "city": city,
            "main": weather_main,
            "description": weather_desc,
            "temp_c": float(temp),
            "feels_like_c": float(feels),
            "humidity": int(humidity),
            "wind_mps": None if wind is None else float(wind),
        }
    except Exception:
        return None

def _parse_dog_breed_from_url(image_url: str):
    """
    Dog CEO 이미지 URL에서 품종 추출.
    예: .../breeds/hound-afghan/n02088094_1003.jpg -> Hound (Afghan)
    """
    try:
        m = re.search(r"/breeds/([^/]+)/", image_url)
        if not m:
            return None
        raw = m.group(1)  # e.g. "hound-afghan" or "retriever-golden"
        parts = raw.split("-")
        if len(parts) == 1:
            return parts[0].replace("_", " ").title()
        base = parts[0].replace("_", " ").title()
        sub = " ".join(p.replace("_", " ").title() for p in parts[1:])
        return f"{base} ({sub})"
    except Exception:
        return None

def get_dog_image():
    """
    Dog CEO 랜덤 강아지 사진 URL과 품종 가져오기
    실패 시 None 반환, timeout=10
    """
    url = "https://dog.ceo/api/breeds/image/random"
    data = _safe_request_json(url, timeout=10)
    if not data or data.get("status") != "success":
        return None
    image_url = data.get("message")
    if not image_url:
        return None
    breed = _parse_dog_breed_from_url(image_url) or "알 수 없음"
    return {"image_url": image_url, "breed": breed}

def _condition_grade(achievement_pct: float, mood: int):
    """
    UI에서도 대략 등급을 보여주기 위한 간단 규칙(참고용).
    AI가 최종 리포트에서 재평가할 수 있음.
    """
    score = achievement_pct * 0.7 + (mood * 10) * 0.3  # 0~100
    if score >= 90: return "S"
    if score >= 80: return "A"
    if score >= 65: return "B"
    if score >= 50: return "C"
    return "D"

def generate_report(
    openai_key: str,
    coach_style: str,
    habits_checked: dict,
    mood: int,
    weather: dict | None,
    dog: dict | None,
):
    """
    습관+기분+날씨+강아지 품종을 모아서 OpenAI에 전달
    - 모델: gpt-5-mini
    - 출력 형식:
      컨디션 등급(S~D), 습관 분석, 날씨 코멘트, 내일 미션, 오늘의 한마디
    """
    if not openai_key:
        return None, "OpenAI API Key가 필요합니다."

    system_prompt = COACH_STYLES.get(coach_style, COACH_STYLES["따뜻한 멘토"])

    payload = {
        "date": date.today().isoformat(),
        "coach_style": coach_style,
        "mood_1_to_10": mood,
        "habits": {k: bool(v) for k, v in habits_checked.items()},
        "weather": weather,
        "dog": {"breed": (dog or {}).get("breed"), "image_url": (dog or {}).get("image_url")},
        "required_format": {
            "컨디션 등급": "S/A/B/C/D 중 1개",
            "습관 분석": "핵심 3~6줄",
            "날씨 코멘트": "1~3줄",
            "내일 미션": "체크박스 습관과 연결된 3개 미션(불릿)",
            "오늘의 한마디": "짧고 강렬하게 1줄",
        },
        "language": "Korean",
        "tone_hint": "간결하지만 구체적으로. 허세 없이 실천 중심.",
    }

    user_prompt = (
        "아래 JSON을 바탕으로 오늘의 코칭 리포트를 작성해줘.\n"
        "반드시 아래 섹션 헤더를 그대로 사용해.\n\n"
        "헤더:\n"
        "1) 컨디션 등급\n"
        "2) 습관 분석\n"
        "3) 날씨 코멘트\n"
        "4) 내일 미션\n"
        "5) 오늘의 한마디\n\n"
        "추가 규칙:\n"
        "- 등급은 S~D 중 하나만.\n"
        "- 내일 미션은 불릿 3개.\n"
        "- 과장 금지, 실행 가능한 문장.\n\n"
        f"입력 JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    try:
        # OpenAI Python SDK (v1+)
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)

        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = resp.choices[0].message.content
        return text, None
    except Exception as e:
        return None, f"OpenAI 호출 실패: {e}"

def _init_sample_history():
    # 데모용 6일 샘플 데이터 (오늘 제외)
    # date: 최근 6일, achievement_pct와 checked_count, mood
    samples = []
    for i, (pct, mood) in enumerate([(60, 6), (80, 7), (40, 5), (100, 8), (70, 7), (90, 9)]):
        d = date.today() - timedelta(days=6 - i)
        checked = round(pct / 20)  # 0~5 근사
        samples.append({
            "date": d.isoformat(),
            "achievement_pct": int(pct),
            "checked_count": int(min(5, max(0, checked))),
            "mood": int(mood),
        })
    return samples

def _upsert_today(history: list[dict], today_row: dict):
    # date 키로 오늘 레코드 upsert
    out = [r for r in history if r.get("date") != today_row.get("date")]
    out.append(today_row)
    out.sort(key=lambda x: x["date"])
    return out

# ----------------------------
# Session state init
# ----------------------------
if "history" not in st.session_state:
    st.session_state["history"] = _init_sample_history()

if "last_report" not in st.session_state:
    st.session_state["last_report"] = None

# ----------------------------
# Check-in UI
# ----------------------------
st.subheader("✅ 오늘의 습관 체크인")

left, right = st.columns([1.2, 1])

with left:
    c1, c2 = st.columns(2)
    habits_checked = {}
    # 2열 배치 (왼쪽 3개, 오른쪽 2개)
    left_keys = HABITS[:3]
    right_keys = HABITS[3:]

    with c1:
        for key, emoji, label in left_keys:
            habits_checked[key] = st.checkbox(f"{emoji} {label}", key=f"hb_{key}")
    with c2:
        for key, emoji, label in right_keys:
            habits_checked[key] = st.checkbox(f"{emoji} {label}", key=f"hb_{key}")

    mood = st.slider("🙂 오늘 기분은 어때요? (1~10)", min_value=1, max_value=10, value=7, key="mood")

with right:
    city = st.selectbox("🏙️ 도시 선택", options=CITIES, index=0, key="city")
    coach_style = st.radio("🎭 코치 스타일", options=list(COACH_STYLES.keys()), index=1, key="coach_style")

# ----------------------------
# Metrics + Achievement
# ----------------------------
checked_count = sum(1 for v in habits_checked.values() if v)
achievement_pct = int(round((checked_count / len(HABITS)) * 100))

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("달성률", f"{achievement_pct}%")
with m2:
    st.metric("달성 습관", f"{checked_count}/{len(HABITS)}")
with m3:
    st.metric("기분", f"{mood}/10")

# ----------------------------
# Chart: 6-day sample + today's live data (7 days)
# ----------------------------
today_row = {
    "date": date.today().isoformat(),
    "achievement_pct": achievement_pct,
    "checked_count": checked_count,
    "mood": mood,
}

# 차트는 "샘플 6일 + 오늘"을 항상 7일로 표시 (오늘은 UI 기준)
history_for_chart = [r for r in st.session_state["history"] if r["date"] != today_row["date"]]
# 최근 6개만 유지 (혹시 사용자가 여러 번 저장했더라도)
history_for_chart = sorted(history_for_chart, key=lambda x: x["date"])[-6:]
history_for_chart.append(today_row)

df = pd.DataFrame(history_for_chart)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

st.subheader("📈 최근 7일 달성률")
st.bar_chart(df.set_index("date")["achievement_pct"])

# ----------------------------
# Actions: APIs + Report
# ----------------------------
st.divider()
st.subheader("🧠 AI 코치 리포트")

btn_col1, btn_col2 = st.columns([1, 2])
with btn_col1:
    generate_btn = st.button("🚀 컨디션 리포트 생성", use_container_width=True)

status_placeholder = st.empty()

weather_data = None
dog_data = None
report_text = None
err_text = None

if generate_btn:
    # 세션 기록 저장(오늘 upsert)
    st.session_state["history"] = _upsert_today(st.session_state["history"], today_row)

    # API 호출
    with status_placeholder:
        st.info("날씨/강아지/AI 리포트를 생성 중...")

    weather_data = get_weather(city, owm_api_key)
    dog_data = get_dog_image()

    report_text, err_text = generate_report(
        openai_key=openai_api_key,
        coach_style=coach_style,
        habits_checked=habits_checked,
        mood=mood,
        weather=weather_data,
        dog=dog_data,
    )

    if report_text:
        st.session_state["last_report"] = report_text

    status_placeholder.empty()

# Display cards + report (use last report if exists)
final_report = report_text or st.session_state.get("last_report")

card1, card2 = st.columns(2)

with card1:
    st.markdown("#### 🌦️ 오늘의 날씨")
    if weather_data is None:
        if owm_api_key:
            st.warning("날씨 정보를 가져오지 못했어요. (도시/키/네트워크를 확인해주세요)")
        else:
            st.info("OpenWeatherMap API Key를 입력하면 날씨를 보여드려요.")
    else:
        wind_txt = "-" if weather_data["wind_mps"] is None else f'{weather_data["wind_mps"]:.1f} m/s'
        st.success(
            f"**{weather_data['city']}**\n\n"
            f"- 상태: {weather_data['description']}\n"
            f"- 기온: {weather_data['temp_c']:.1f}°C (체감 {weather_data['feels_like_c']:.1f}°C)\n"
            f"- 습도: {weather_data['humidity']}%\n"
            f"- 바람: {wind_txt}"
        )

with card2:
    st.markdown("#### 🐶 오늘의 강아지")
    if dog_data is None:
        st.warning("강아지 이미지를 가져오지 못했어요. (네트워크를 확인해주세요)")
    else:
        st.image(dog_data["image_url"], use_container_width=True, caption=f"품종: {dog_data['breed']}")

st.markdown("#### 📝 AI 코치 리포트")
if err_text:
    st.error(err_text)
elif final_report:
    st.markdown(final_report)
else:
    st.info("버튼을 눌러 오늘의 리포트를 생성해보세요!")

# ----------------------------
# Share text
# ----------------------------
st.markdown("#### 📣 공유용 텍스트")
approx_grade = _condition_grade(achievement_pct, mood)
weather_short = (
    f"{weather_data['city']} {weather_data['description']} {weather_data['temp_c']:.0f}°C"
    if weather_data else "날씨 정보 없음"
)
dog_short = f"{dog_data['breed']}" if dog_data else "강아지 정보 없음"

share_text = (
    f"📊 AI 습관 트래커 ({date.today().isoformat()})\n"
    f"- 달성률: {achievement_pct}% ({checked_count}/5)\n"
    f"- 기분: {mood}/10\n"
    f"- (참고) 컨디션 추정: {approx_grade}\n"
    f"- 날씨: {weather_short}\n"
    f"- 오늘의 강아지: {dog_short}\n\n"
    f"[체크한 습관]\n"
    + "\n".join([f"- {emoji} {label}" for k, emoji, label in HABITS if habits_checked.get(k)])
)
st.code(share_text, language="text")

# ----------------------------
# API 안내
# ----------------------------
with st.expander("ℹ️ API 안내 / 설정 팁"):
    st.markdown(
        """
- **OpenAI API Key**: OpenAI 대시보드에서 발급한 키를 입력하세요.
- **OpenWeatherMap API Key**: OpenWeatherMap에서 키를 발급받아 입력하세요.
- 배포 시에는 Streamlit **Secrets**에 키를 저장하는 방식을 권장합니다.
- 네트워크/키 오류가 나면 날씨/리포트 생성이 실패할 수 있어요.
- Dog CEO API는 공개 API라 키 없이 동작합니다.
        """.strip()
    )
