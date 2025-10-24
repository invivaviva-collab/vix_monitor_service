import os
import sys
import asyncio
import aiohttp
import io
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

# FastAPI 관련 임포트
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from starlette.responses import RedirectResponse

# 그래프/데이터 관련 외부 라이브러리
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import numpy as np

# Matplotlib 백엔드 설정 (헤드리스 서버 환경을 위해 필수)
matplotlib.use('Agg')

# =========================================================
# --- [1] 설정 및 환경 변수 로드 및 전역 상태 ---
# =========================================================
# 한국 시간 (KST) 타임존 설정
KST_TZ = ZoneInfo("Asia/Seoul")
MONITOR_INTERVAL_SECONDS = 60 # 1분마다 시간 체크

# ⏰ 전역 상태: 사용자가 설정할 수 있는 발송 시간 (KST)
TARGET_HOUR_KST = int(os.environ.get('TARGET_HOUR_KST', 7))
TARGET_MINUTE_KST = int(os.environ.get('TARGET_MINUTE_KST', 20))

# ⚠️ 환경 변수에서 로드 (Render 환경에 필수) - 사용자가 지정한 하드코딩 값 유지
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
TELEGRAM_TARGET_CHAT_ID = os.environ.get('TELEGRAM_TARGET_CHAT_ID', '-1000000000')
SERVER_PORT = int(os.environ.get("PORT", 8000))

# 로깅 설정 (INFO 레벨로 주요 동작만 기록)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 특수 문자 오류 제거
if 'YOUR_BOT_TOKEN_HERE' in TELEGRAM_BOT_TOKEN or TELEGRAM_TARGET_CHAT_ID == '-1000000000':
    logger.warning("⚠️ Warning: TELEGRAM_BOT_TOKEN or CHAT_ID is set to default. Please configure environment variables.")

# 💾 서버 RAM에서 상태 유지 (Render 재시작 시 초기화될 수 있음 - 디스크 미사용)
status = {
    "last_sent_date_kst": "1970-01-01", 
    "last_check_time_kst": "N/A",
    "next_scheduled_time_kst": "N/A",
    "last_self_ping_kst": "N/A"
}

# =========================================================
# --- [2] VIX Plotter 함수 (그래프 생성 로직) - 최적화: 데이터 반환 추가 ---
# =========================================================
def plot_vix_sp500(width=6.4, height=4.8) -> Optional[Tuple[io.BytesIO, float, float, str]]:
    """
    VIX와 S&P 500의 종가 추이를 비교하는 차트를 생성하고, 
    생성된 차트 버퍼와 함께 최신 데이터를 반환합니다.
    """
    logger.info("📈 Starting data download and chart generation...")

    max_retry = 4 
    tickers = ["^VIX", "^GSPC"]
    vix, qqq = None, None
    latest_vix, latest_gspc, latest_date_utc = 0.0, 0.0, "N/A" # 새로 추가된 반환 변수
    
    start_date = "2025-04-01" 
    
    for attempt in range(1, max_retry + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_retry}: Downloading VIX and S&P 500 data (start={start_date})...")
            
            # 데이터 다운로드 (period 대신 start 사용)
            data_all = yf.download(tickers, start=start_date, progress=False, timeout=20)
            
            # Close 데이터 추출
            vix_df = data_all['Close']['^VIX'].dropna()
            gspc_df = data_all['Close']['^GSPC'].dropna()
            
            # 공통 날짜 맞추기
            common_dates = vix_df.index.intersection(gspc_df.index)
            vix = vix_df.loc[common_dates]
            qqq = gspc_df.loc[common_dates]

            # 데이터 유효성 검사
            if vix.empty or qqq.empty:
                raise ValueError("Downloaded data is empty after aligning dates.")

            # ⭐️ 최신 데이터 추출 (캡션 생성을 위해) ⭐️
            latest_vix = vix.iloc[-1].item()
            latest_gspc = qqq.iloc[-1].item()
            # VIX와 GSPC의 마지막 인덱스 중 더 최근 날짜를 사용 (일반적으로 같음)
            latest_date_utc = max(vix.index[-1], qqq.index[-1]).strftime('%Y-%m-%d')

            logger.info(f"Attempt {attempt}: Data downloaded successfully (VIX={latest_vix:.2f}, S&P500={latest_gspc:.0f}).")
            break 
            
        except Exception as e:
            logger.warning(f"Data download failed (Attempt {attempt}): {e}")
            if attempt < max_retry:
                # ⭐️ 지수적 백오프(Exponential Backoff) 적용: 2^1=2s, 2^2=4s, 2^3=8s 대기
                sleep_time = 5 ** attempt
                logger.info(f"Applying Exponential Backoff. Waiting {sleep_time} seconds before next retry...")
                time.sleep(sleep_time)
            else:
                logger.error("Max retries exceeded. Failed to acquire data.")
                return None
    
    if vix is None or qqq is None:
        return None

    # 최종 확정된 차트 디자인 로직 적용
    try:
        # 폰트 설정 제거 (서버 환경 안정화를 위해)
        plt.style.use('dark_background')
        
        fig, ax1 = plt.subplots(figsize=(width, height)) 
        ax2 = ax1.twinx()
        
        # 배경색 설정
        fig.patch.set_facecolor('#222222')
        ax1.set_facecolor('#2E2E2E')
        ax2.set_facecolor('#2E2E2E')
        
        # 데이터 및 색상
        common_dates = vix.index 
        title_text = f"VIX ({latest_vix:.2f}) vs S&P 500 ({latest_gspc:.0f})"
        vix_color = '#FF6B6B' # VIX 색상 (빨간색 계열)
        qqq_color = '#6BCBFF' # S&P 500 색상 (파란색 계열)
        new_fontsize = 8 * 1.3
        
        # 플로팅
        ax2.plot(common_dates, vix.values, color=vix_color, linewidth=1.5)
        ax1.plot(common_dates, qqq.values, color=qqq_color, linewidth=1.5)
        
        # X축 날짜 포맷 및 간격 설정
        formatter = mdates.DateFormatter('%Y-%m-%d') 
        ax1.xaxis.set_major_formatter(formatter)
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4)) # 4개월 간격 유지
        fig.autofmt_xdate(rotation=45)

        # Y축 레이블 설정 (한글 -> 영어 변경)
        ax1.set_ylabel('S&P 500 Index', color=qqq_color, fontsize=12, fontweight='bold', labelpad=5)
        ax2.set_ylabel('VIX', color=vix_color, fontsize=12, fontweight='bold', labelpad=5)
        
        # VIX 레벨 주석 및 수평선 추가
        try:
            # 전체 데이터 기간의 90% 지점 날짜를 찾습니다.
            new_text_x_pos = common_dates[int(len(common_dates)*0.9)]
        except:
             # 데이터가 너무 적을 경우의 안전 장치
             new_text_x_pos = common_dates[-1] + timedelta(days=1)
        
        # VIX 주석 (한글 -> 영어 변경)
        ax2.text(new_text_x_pos, 15.5, "VIX 15 (Greed/Sell)", color='yellow', fontsize=new_fontsize, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')
        ax2.text(new_text_x_pos, 30.5, "VIX 30 (Warning)", color='peru', fontsize=new_fontsize, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')
        ax2.text(new_text_x_pos, 40.5, "VIX 40 (Fear/Buy)", color='lightGreen', fontsize=new_fontsize, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')
        
        # VIX 수평선
        ax2.axhline(y=15, color='yellow', linestyle='--', linewidth=1.2, alpha=0.8)
        ax2.axhline(y=30, color='peru', linestyle='--', linewidth=1.0, alpha=0.8)
        ax2.axhline(y=40, color='lightGreen', linestyle='--', linewidth=1.2, alpha=0.8)
        
        # 제목 및 여백 최소화
        fig.suptitle(title_text, color='white', fontsize=12, fontweight='bold', y=0.98) 
        fig.tight_layout(rect=[0.025, 0.025, 1, 1]) 
        
        # ⭐️ 메모리 버퍼에 PNG 이미지로 저장 (디스크 미사용 핵심) ⭐️
        plot_data = io.BytesIO()
        plt.savefig(plot_data, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1) 
        plot_data.seek(0)
        
        plt.close(fig) # **매우 중요: 메모리 누수 방지**
        logger.info("✅ Chart generation complete (saved to memory).")
        
        # ⭐️ 차트 버퍼와 함께 최신 데이터를 튜플로 반환 ⭐️
        return plot_data, latest_vix, latest_gspc, latest_date_utc

    except Exception as e:
        logger.error(f"❌ Exception during chart generation: {e}", exc_info=True)
        return None

# =========================================================
# --- [3] Telegram 전송 함수 (HTTP API) ---
# =========================================================
async def send_photo_via_http(chat_id: str, photo_bytes: io.BytesIO, caption: str) -> bool:
    """텔레그램 봇으로 차트 이미지를 발송합니다."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    data = aiohttp.FormData()
    data.add_field('chat_id', chat_id)
    data.add_field('caption', caption)
    data.add_field('parse_mode', 'Markdown')
    # ⭐️ io.BytesIO 객체를 직접 photo 필드에 전달 ⭐️
    data.add_field('photo', 
                    photo_bytes, 
                    filename='vix_gspc_chart.png', 
                    content_type='image/png')

    # 재시도 로직 추가 (네트워크 문제 대비)
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                logger.info(f"Telegram send request initiated (Attempt {attempt + 1}/3, Chat ID: {chat_id})...")
                async with session.post(url, data=data) as response:
                    response.raise_for_status() # HTTP 오류 발생 시 예외 발생
                    response_json = await response.json()
                    if response_json.get('ok'):
                        logger.info("✅ Telegram send successful!")
                        return True
                    else:
                        error_desc = response_json.get('description', 'Unknown Error')
                        raise Exception(f"Telegram API Error: {error_desc}")
                        
        except Exception as e:
            logger.warning(f"❌ Telegram send error (Attempt {attempt + 1}/3): {e}. Retrying shortly.")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt) # Exponential Backoff: 1s, 2s 대기
            
    logger.error("Telegram send final failure.")
    return False

async def run_and_send_plot() -> bool:
    """차트 생성 및 전송의 전체 프로세스를 실행합니다. (최적화 적용)"""
    global status
    
    if 'YOUR_BOT_TOKEN_HERE' in TELEGRAM_BOT_TOKEN or TELEGRAM_TARGET_CHAT_ID == '-1000000000':
        logger.error("Telegram token or Chat ID is set to default. Skipping send.")
        return False
        
    # ⭐️ plot_vix_sp500 호출 결과 변경 반영 ⭐️
    plot_result = plot_vix_sp500()
    
    if not plot_result:
        logger.error("Chart generation failed. Skipping send and recalculating next target time.")
        return False
    
    plot_buffer, latest_vix, latest_gspc, latest_date_utc = plot_result
    
    # -------------------------------------------------------------
    # 🚫 제거된 코드: 중복 데이터 다운로드 로직을 삭제했습니다.
    # -------------------------------------------------------------
    # 최신 데이터는 plot_vix_sp500 함수에서 이미 가져와 반환했으므로 
    # 다시 다운로드할 필요가 없습니다.
    # -------------------------------------------------------------

    caption = (
        # 한글 -> 영어 변경
        f"\n🗓️ {latest_date_utc} (US Market Close)\n"
        f"📉 VIX (Volatility): **{latest_vix:.2f}**\n"
        f"📈 S&P 500 (Index): **{latest_gspc:.0f}**\n\n"
        f"VIX and the S&P 500 typically move in opposite directions.\n"
    )

    success = await send_photo_via_http(TELEGRAM_TARGET_CHAT_ID, plot_buffer, caption)
    plot_buffer.close() # 메모리 버퍼 닫기 (메모리 해제)

    if success:
        current_kst = datetime.now(KST_TZ)
        status['last_sent_date_kst'] = current_kst.strftime("%Y-%m-%d")
        logger.info(f"Successfully sent. Last sent date updated: {status['last_sent_date_kst']}")
    
    return success

# =========================================================
# --- [4] 스케줄링 및 루프 로직 ---
# =========================================================
def calculate_next_target_time(now_kst: datetime) -> datetime:
    """현재 시간을 기준으로 다음 발송 목표 시간 (KST)을 계산합니다. (전역 변수 사용)"""
    global TARGET_HOUR_KST, TARGET_MINUTE_KST
    
    target_time_today = now_kst.replace(
        hour=TARGET_HOUR_KST, 
        minute=TARGET_MINUTE_KST, 
        second=0, 
        microsecond=0
    )
    
    if now_kst >= target_time_today:
        # 오늘 목표 시간을 지났다면, 내일로 설정
        next_target = target_time_today + timedelta(days=1)
    else:
        # 오늘 목표 시간이 아직 안 되었다면, 오늘로 설정
        next_target = target_time_today
        
    return next_target

async def main_monitor_loop():
    """1분마다 실행되며, 발송 시간을 확인하고 작업을 트리거합니다."""
    global status
    
    # 초기 다음 발송 시간 설정
    now_kst = datetime.now(KST_TZ)
    next_target_time_kst = calculate_next_target_time(now_kst)
    status['next_scheduled_time_kst'] = next_target_time_kst.strftime("%Y-%m-%d %H:%M:%S KST")
    
    logger.info(f"🔍 Monitoring started. Next scheduled time (KST): {status['next_scheduled_time_kst']}")
    
    while True:
        await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
        
        current_kst = datetime.now(KST_TZ)
        status['last_check_time_kst'] = current_kst.strftime("%Y-%m-%d %H:%M:%S KST")
        
        # 🔔 요청에 따라 1분마다 스케줄 확인 로그를 WARNING 레벨로 출력합니다.
        logger.warning(f"Monitor: Checking schedule (KST: {current_kst.strftime('%H:%M:%S')}).")
        
        # 발송 조건 확인 (하루에 한 번, 지정된 시간에 발송)
        target_date_kst = next_target_time_kst.strftime("%Y-%m-%d")

        # -----------------------------------------------------------
        # 🌟 [수정된 로직] 요일 체크 추가 (월요일=0, 일요일=6)
        # -----------------------------------------------------------
        # current_kst.weekday()는 월요일(0)부터 일요일(6)까지 반환합니다.
        is_monday_or_sunday = (current_kst.weekday() == 0) or (current_kst.weekday() == 6)

        if current_kst >= next_target_time_kst and \
           current_kst < next_target_time_kst + timedelta(minutes=1) and \
           target_date_kst != status['last_sent_date_kst']:

            if is_monday_or_sunday:
                # 월요일(0) 또는 일요일(6)일 경우 발송을 건너뛰고 다음 목표 시간만 업데이트
                logger.info(f"🚫 Skip send: Today is Monday or Sunday (KST). Only updating next scheduled time.")
            else:
                # 한글 -> 영어 변경
                logger.info(f"⏰ Send time reached (KST: {current_kst.strftime('%H:%M:%S')}). Executing job.")
                
                # 발송 로직 실행
                await run_and_send_plot()
            
            # 다음 목표 시간 업데이트 (발송 성공 여부와 관계없이)
            next_target_time_kst = calculate_next_target_time(current_kst)
            status['next_scheduled_time_kst'] = next_target_time_kst.strftime("%Y-%m-%d %H:%M:%S KST")
            # 한글 -> 영어 변경
            logger.info(f"➡️ Next scheduled time (KST): {status['next_scheduled_time_kst']}")
            
        elif current_kst.day != next_target_time_kst.day and \
             current_kst.hour > TARGET_HOUR_KST + 1:
            # 목표 날짜가 현재 날짜를 지나쳤는데 아직 업데이트가 안 된 경우 (예: 서버 재시작 직후)
            next_target_time_kst = calculate_next_target_time(current_kst)
            status['next_scheduled_time_kst'] = next_target_time_kst.strftime("%Y-%m-%d %H:%M:%S KST")

async def self_ping_loop():
    """
    [내부용 슬립 방지] 5분마다 내부적으로 자신의 Health Check 엔드포인트에 핑을 보내는 루프.
    """
    global status
    # Render 내부에서 자신의 IP/포트로 요청
    ping_url = f"http://127.0.0.1:{SERVER_PORT}/" 
    logger.info(f"🛡️ Starting internal self-ping loop. Requesting {ping_url} every 5 minutes.")
    
    async with aiohttp.ClientSession() as session:
        while True:
            await asyncio.sleep(5 * 60) # 5분 대기
            
            try:
                # HEAD 요청은 GET보다 가볍습니다.
                async with session.head(ping_url, timeout=10) as response:
                    if response.status == 200:
                        status['last_self_ping_kst'] = datetime.now(KST_TZ).strftime("%Y-%m-%d %H:%M:%S KST")
                        logger.debug(f"✅ Self-ping successful: {status['last_self_ping_kst']}")
                    else:
                        logger.warning(f"❌ Self-ping failed (Status: {response.status})")
                        
            except Exception as e:
                logger.error(f"❌ Exception during self-ping: {e}")


# =========================================================
# --- [5] FastAPI 웹 서비스 및 핑 체크 설정 ---
# =========================================================

app = FastAPI(
    title="VIX Plot Telegram Scheduler",
    description="VIX/S&P 500 Chart Sender running on Render Free Tier.",
    version="1.0.0"
)

# 서버 시작 시 백그라운드 작업 시작
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 스케줄러 루프와 셀프 핑 루프를 백그라운드에서 시작합니다."""
    # 메인 스케줄링 루프
    asyncio.create_task(main_monitor_loop()) 
    # 슬립 방지 보조용 셀프 핑 루프
    asyncio.create_task(self_ping_loop())    
    logger.info("🚀 Background scheduling and self-ping loops have started.") # 한글 -> 영어 변경

# ---------------------------------------------------------
# 새로운 엔드포인트: 스케줄링 시간 설정
# ---------------------------------------------------------
@app.post("/set-time")
async def set_schedule_time(
    hour: str = Form(...), 
    minute: str = Form(...) 
):
    """사용자가 입력한 KST 시간을 저장하고 다음 스케줄 시간을 업데이트합니다."""
    global TARGET_HOUR_KST, TARGET_MINUTE_KST
    global status

    try:
        hour_int = int(hour)
        minute_int = int(minute)
    except ValueError:
        raise HTTPException(status_code=400, detail="Hour and minute must be integers.") # 한글 -> 영어 변경
        
    # 유효성 검사
    if not (0 <= hour_int <= 23 and 0 <= minute_int <= 59):
        raise HTTPException(status_code=400, detail="Invalid hour (0-23) or minute (0-59).") # 한글 -> 영어 변경
        
    # 전역 변수 업데이트
    TARGET_HOUR_KST = hour_int
    TARGET_MINUTE_KST = minute_int
    
    # 변경 사항을 즉시 반영하여 다음 목표 시간 재계산
    now_kst = datetime.now(KST_TZ)
    next_target_time_kst = calculate_next_target_time(now_kst)
    status['next_scheduled_time_kst'] = next_target_time_kst.strftime("%Y-%m-%d %H:%M:%S KST")

    # 한글 -> 영어 변경
    logger.info(f"⏰ Schedule time changed to: {TARGET_HOUR_KST:02d}:{TARGET_MINUTE_KST:02d} KST. Next send time updated: {status['next_scheduled_time_kst']}") 
    
    # 상태 페이지로 리다이렉트 (303 See Other)
    return RedirectResponse(url="/", status_code=303)

# ---------------------------------------------------------
# Health Check Endpoint (Request 객체 추가 및 HEAD 처리 로직 수정)
# ---------------------------------------------------------
@app.get("/")
@app.head("/")
async def health_check(request: Request): # 👈 Request 객체를 인수로 받음
    """Render Free Tier의 Spin Down을 방지하기 위한 Health Check 엔드포인트."""
    global TARGET_HOUR_KST, TARGET_MINUTE_KST
    current_kst = datetime.now(KST_TZ)
    
    # HEAD 요청의 경우 간단한 응답만 반환하여 부하 최소화
    # request.method로 요청 방식을 확인합니다.
    if request.method == "HEAD":
        return {"status": "ok"}
        
    status_html = f"""
    <html>
        <head>
            <title>VIX Scheduler Status (KST)</title>
            <style>
                body {{ font-family: 'Arial', sans-serif; background-color: #f4f7f6; color: #333; text-align: center; padding: 50px; }}
                .container {{ background-color: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); display: inline-block; text-align: left; max-width: 600px; width: 90%; }}
                h1 {{ color: #2ecc71; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                h2 {{ color: #3498db; margin-top: 25px; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
                p {{ margin: 10px 0; line-height: 1.5; }}
                .highlight {{ font-weight: bold; color: #3498db; background-color: #ecf0f1; padding: 2px 5px; border-radius: 3px; }}
                .alert {{ color: #e74c3c; font-weight: bold; margin-top: 20px; padding: 10px; border: 1px dashed #e74c3c; border-radius: 5px; }}
                .form-group {{ display: flex; align-items: center; gap: 10px; margin-bottom: 15px; }}
                .form-group label {{ font-weight: bold; width: 120px; }}
                .form-group input {{ padding: 8px; border: 1px solid #ccc; border-radius: 5px; width: 60px; text-align: center; }}
                .form-group button {{ background-color: #3498db; color: white; padding: 8px 15px; border: none; border-radius: 5px; cursor: pointer; transition: background-color 0.3s; }}
                .form-group button:hover {{ background-color: #2980b9; }}
                .time-setting {{ margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>✅ VIX Scheduler Status (KST)</h1>

                <h2>Current Schedule Status</h2>
                <p>Current KST Time: <span class="highlight">{current_kst.strftime('%Y-%m-%d %H:%M:%S KST')}</span></p>
                <p>Current Set Send Time: <span class="highlight">{TARGET_HOUR_KST:02d}:{TARGET_MINUTE_KST:02d} KST</span></p>
                <p>Next Scheduled Send Time: <span class="highlight">{status.get('next_scheduled_time_kst')}</span></p>
                <p>Last Successful Send Date: <span class="highlight">{status.get('last_sent_date_kst')}</span></p>
                <p>🛡️ Last Self-Ping: <span class="highlight">{status.get('last_self_ping_kst')}</span></p>

                <div class="time-setting">
                    <h2>Set Send Time (KST)</h2>
                    <form action="/set-time" method="POST">
                        <div class="form-group">
                            <label for="hour">Hour (0-23):</label>
                            <input type="number" id="hour" name="hour" min="0" max="23" value="{TARGET_HOUR_KST}" required>
                            <label for="minute">Minute (0-59):</label>
                            <input type="number" id="minute" name="minute" min="0" max="59" value="{TARGET_MINUTE_KST}" required>
                        </div>
                        <div class="form-group" style="justify-content: flex-end;">
                            <button type="submit">Update Schedule Time</button>
                        </div>
                    </form>
                </div>

                <div class="alert">
                    🔔 **IMPORTANT**: To keep this service alive, you must configure an external monitoring service (e.g., UptimeRobot) to periodically request this URL (every 5 minutes).
                </div>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=status_html, status_code=200)

# =========================================================
# --- [6] 실행 (Render는 이 부분을 사용하지 않고 Procfile을 사용) ---
# =========================================================
if __name__ == '__main__':
    # 이 부분은 로컬 테스트를 위한 코드이며, Render 환경에서는 uvicorn vix_monitor_service:app 명령어를 사용합니다.
    import uvicorn
    logger.info(f"Starting uvicorn server on port {SERVER_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
