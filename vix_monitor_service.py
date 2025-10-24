import os
import sys
import asyncio
import aiohttp
import io
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

# =========================================================
# 💡 [2] 그래프/데이터 관련 외부 라이브러리 (글로벌 임포트로 이동)
# 사용자님의 지적대로, 반복 임포트 비효율성 및 클린 코드 준수를 위해 상단으로 이동
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
# zoneinfo는 Python 3.9 이상에서 표준 라이브러리입니다.
from zoneinfo import ZoneInfo 
# =========================================================

# 로깅 설정 (INFO 레벨로 주요 동작만 기록)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================================================
# --- [1] 설정 및 환경 변수 로드 ---
# =========================================================
# 한국 시간 (KST)은 UTC+9입니다.
KST_OFFSET_HOURS = 9
# ⏰ 사용자가 원하는 발송 시간 설정 (시, 분)
TARGET_HOUR_KST = 10    # 한국 시간 '시'
TARGET_MINUTE_KST = 45 # 한국 시간 '분' (예: 8시 30분)
MONITOR_INTERVAL_SECONDS = 60 # 1분마다 시간 체크 (중복 발송 방지를 위해 유지)

# ⚠️ 환경 변수에서 로드 (Render 환경에 필수)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
TELEGRAM_TARGET_CHAT_ID = os.environ.get('TELEGRAM_TARGET_CHAT_ID', '-1000000000')

# 서버 RAM에서 상태 유지 (Render 재시작 시 초기화될 수 있음)
status = {"last_sent_date_kst": "1970-01-01", "last_check_time_kst": "N/A"}

if 'YOUR_BOT_TOKEN_HERE' in TELEGRAM_BOT_TOKEN or TELEGRAM_TARGET_CHAT_ID == '-1000000000':
    logging.warning("⚠️ 경고: TELEGRAM_BOT_TOKEN 또는 CHAT_ID가 기본값입니다. 환경 변수를 설정해주세요.")


# =========================================================
# --- [2] VIX Plotter 함수 (그래프 생성 로직) ---
# *주의: 이제 내부에서 임포트 하지 않습니다!*
# =========================================================
def plot_vix_sp500(width=6.4, height=4.8):
    """
    VIX/S&P 500 데이터를 다운로드하고 그래프를 생성하여 BytesIO로 반환합니다.
    """
    
    # vix_plotter.py의 설정 반영
    matplotlib.use('Agg')
    plt.style.use('dark_background')
    # 한글 폰트 설정 (서버 환경에 맞춰 Noto Sans CJK JP 사용 권장)
    try:
        # 'Malgun Gothic' 대신 서버 환경에 맞는 폰트 사용
        matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP' 
    except Exception:
        logging.warning("Noto Sans CJK JP 폰트 로드 실패. 기본 폰트 사용.")
    
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d") # 최근 6개월 데이터

    try:
        logging.info("그래프 데이터 생성 중... (yfinance 다운로드)")
        # ⚠️ yfinance는 이제 전역(Global)에서 임포트되었습니다.
        vix_df = yf.download("^VIX", start=start_date, end=None, progress=False)
        qqq_df = yf.download("^GSPC", start=start_date, end=None, progress=False)
        
        vix = vix_df["Close"].dropna()
        qqq = qqq_df["Close"].dropna()
        common_dates = vix.index.intersection(qqq.index)
        vix = vix.loc[common_dates]
        qqq = qqq.loc[common_dates]
        if vix.empty or qqq.empty: 
            logging.error("yfinance에서 데이터를 가져오지 못했습니다.")
            return None

        # 플로팅 로직 (이하 동일)
        fig, ax1 = plt.subplots(figsize=(width, height)) 
        ax2 = ax1.twinx()
        fig.patch.set_facecolor('#222222')
        ax1.set_facecolor('#2E2E2E')
        ax2.set_facecolor('#2E2E2E')
        
        last_vix_price = vix.iloc[-1].item()
        last_qqq_price = qqq.iloc[-1].item()
        title_text = f"VIX ({last_vix_price:.2f}) vs S&P 500 ({last_qqq_price:.2f})"
        vix_color = '#FF6B6B'
        qqq_color = '#6BCBFF'
        new_fontsize = 8 * 1.3
        
        # 마지막 10% 구간에 주석 위치 지정
        if len(common_dates) > 10:
             new_text_x_pos = common_dates[int(len(common_dates)*0.9)]
        else:
             new_text_x_pos = common_dates[-1]

        ax2.plot(common_dates, vix.values, color=vix_color, linewidth=1.5)
        ax1.plot(common_dates, qqq.values, color=qqq_color, linewidth=1.5)
        
        formatter = mdates.DateFormatter('%y-%m-%d') 
        ax1.xaxis.set_major_formatter(formatter)
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        fig.autofmt_xdate(rotation=45)

        # 주석
        ax2.text(new_text_x_pos, 15, "VIX 15 (탐욕/매도)", color='lightGreen', fontsize=new_fontsize, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')
        ax2.text(new_text_x_pos, 30, "VIX 30 (경고)", color='peru', fontsize=new_fontsize, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')
        ax2.text(new_text_x_pos, 40, "VIX 40 (공포/매수)", color='orange', fontsize=new_fontsize, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')
        ax2.axhline(y=15, color='lightGreen', linestyle='--', linewidth=1.2, alpha=0.8)
        ax2.axhline(y=30, color='peru', linestyle='--', linewidth=1.0, alpha=0.8)
        ax2.axhline(y=40, color='orange', linestyle='--', linewidth=1.2, alpha=0.8)

        fig.suptitle(title_text, color='white', fontsize=12, fontweight='bold', y=0.97) 
        fig.tight_layout(rect=[0, 0.01, 1, 0.98]) 
        
        # 메모리 저장 및 반환
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100) 
        buf.seek(0)
        plt.close(fig)
        return buf
        
    except Exception as e:
        logging.error(f"그래프 생성 중 오류 발생: {e}")
        return None
# -----------------------------------------------------------------


# =========================================================
# --- [3] Telegram 전송 함수 (HTTP API) ---
# =========================================================
async def send_photo_via_http(chat_id: str, photo_bytes: io.BytesIO, caption: str):
    """
    aiohttp를 사용하여 텔레그램 sendPhoto API로 직접 이미지를 전송합니다.
    """
    api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    data = aiohttp.FormData()
    data.add_field('chat_id', chat_id)
    data.add_field('caption', caption)
    data.add_field('photo', 
                    photo_bytes, 
                    filename='vix_plot.png', 
                    content_type='image/png')
    
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, data=data, timeout=15) as resp:
                    resp.raise_for_status() 
                    response_json = await resp.json()
                    if response_json.get('ok'):
                        logging.info(f"텔레그램 전송 성공. (채널: {chat_id})")
                        return True
                    else:
                        error_desc = response_json.get('description', 'Unknown Error')
                        raise Exception(f"Telegram API Error: {error_desc}")
                        
        except Exception as e:
            logging.error(f"텔레그램 발송 실패 (시도 {attempt + 1}/3): {e}. 잠시 후 재시도.")
            await asyncio.sleep(2 ** attempt) 
            
    logging.error("텔레그램 발송 최종 실패.")
    return False


async def run_and_send_plot():
    """
    그래프를 생성하고 전송을 실행하는 메인 함수입니다.
    """
    logging.info("VIX/S&P 500 그래프 생성 및 전송 시작...")
    
    # 1. 그래프 데이터 (메모리 내 바이트) 생성
    plot_data = plot_vix_sp500(width=6.4, height=4.8)
    
    if not plot_data:
        logging.error("그래프 데이터 생성 실패로 전송 중단.")
        return False # 전송 실패

    # 2. 이미지 전송 (HTTP API 사용)
    current_kst = datetime.utcnow() + timedelta(hours=KST_OFFSET_HOURS)
    caption = (
        f"VIX V.S. S&P 500 ({current_kst.strftime('%Y년 %m월 %d일 %H:%M KST')})"
    )
    
    success = await send_photo_via_http(TELEGRAM_TARGET_CHAT_ID, plot_data, caption)

    # 3. 바이트 객체 정리 (메모리에서 제거)
    plot_data.close() 
    logging.info("메모리 바이트 객체 정리 완료.")
    
    return success


# =========================================================
# --- [4] 스케줄링 및 루프 로직 ---
# =========================================================

async def main_monitor_loop():
    """
    Render 백그라운드에서 실행될 메인 스케줄링 루프입니다. (1분 간격 체크)
    """
    logging.info("--- VIX 그래프 모니터링 스케줄러 (백그라운드 - 1분 주기) 시작 ---")
    
    while True:
        try:
            now_utc = datetime.utcnow()
            now_kst = now_utc + timedelta(hours=KST_OFFSET_HOURS)
            today_kst_str = now_kst.strftime("%Y-%m-%d")
            
            # ⏰ 시, 분을 추출하여 사용자가 설정한 값과 비교
            current_hour = now_kst.hour
            current_minute = now_kst.minute
            
            # Health Check 엔드포인트에서 마지막 확인 시간을 보여주기 위해 업데이트
            status['last_check_time_kst'] = now_kst.strftime("%Y-%m-%d %H:%M:%S") 
            
            current_weekday = now_kst.weekday() # Mon=0, Tue=1, ..., Sat=5, Sun=6
            
            # 1. 유효 요일 확인 (화요일(1) ~ 토요일(5)) - 일요일(6), 월요일(0) 제외
            is_valid_day = 1 <= current_weekday <= 5
            
            # 2. 목표 시간 확인 (정확히 TARGET_HOUR:TARGET_MINUTE KST)
            is_target_time = (current_hour == TARGET_HOUR_KST and current_minute == TARGET_MINUTE_KST)
            target_time_str = f"{TARGET_HOUR_KST:02d}:{TARGET_MINUTE_KST:02d}"
            
            # 3. 오늘 발송 완료 여부 확인 (하루 1회 발송 보장)
            is_already_sent = (status['last_sent_date_kst'] == today_kst_str)

            current_time_str = f"{current_hour:02d}:{current_minute:02d}"
            
            if is_valid_day and is_target_time and not is_already_sent:
                # 조건 충족: 발송 시작
                logging.info(f"[ACTION] KST:{current_time_str} | DAY:{current_weekday} | 목표 시간({target_time_str}) 도달, 발송 시작")

                success = await run_and_send_plot()
                
                if success:
                    # 발송 성공 시에만 상태 업데이트 (중복 발송 방지)
                    status['last_sent_date_kst'] = today_kst_str
            
            elif is_target_time and is_already_sent:
                logging.debug(f"[SKIP] KST:{current_time_str} | 금일({today_kst_str}) 이미 발송 완료됨.")

            elif not is_valid_day:
                logging.debug(f"[SKIP] KST:{current_time_str} | 비영업일(일/월)이므로 건너뜁니다.")

            elif not is_target_time:
                # 목표 시간이 아닌 경우, INFO 대신 DEBUG 레벨로 출력하여 로그 폭주 방지
                logging.debug(f"[WAIT] KST:{current_time_str} | 다음 목표 시간({target_time_str}) 대기 중")
            
        except Exception as e:
            logging.error(f"[ERROR] 스케줄링 루프 중 치명적인 오류 발생: {e}. 60초 후 재시도.")
            
        # Fixed 1-minute sleep
        await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
            
# =========================================================
# --- [5] FastAPI 웹 서비스 및 핑 체크 설정 ---
# =========================================================
# FastAPI 및 uvicorn import는 파일 상단에 이미 있습니다.
app = FastAPI(
    title="VIX Plot Telegram Scheduler",
    description="VIX/S&P 500 Chart Sender running on Render Free Tier.",
    version="1.0.0"
)

# 서버 시작 시 백그라운드 작업 시작
@app.on_event("startup")
async def startup_event():
    logging.info("FastAPI Server Startup: Launching main_monitor_loop as a background task.")
    asyncio.create_task(main_monitor_loop())

# Health Check Endpoint (외부 모니터링 서비스(UptimeRobot 등)가 사용자의 서버 슬립을 방지하는 용도)
@app.get("/")
@app.head("/") 
async def health_check():
    return {
        "status": "running", 
        "message": f"VIX scheduler is active in the background (checking every {MONITOR_INTERVAL_SECONDS} seconds).",
        "last_plot_sent_date_kst": status.get('last_sent_date_kst'),
        "last_check_time_kst": status.get('last_check_time_kst'),
        "target_time_kst": f"{TARGET_HOUR_KST:02d}:{TARGET_MINUTE_KST:02d}" 
    }

# =========================================================
# --- [6] 실행 ---
# =========================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    logging.info(f"Starting uvicorn server on port {port}...")
    # uvicorn import는 파일 상단에 이미 있습니다.
    uvicorn.run(app, host="0.0.0.0", port=port)
