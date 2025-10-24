import os
import sys
import asyncio
import aiohttp
import io
import logging
import time
from datetime import datetime, timedelta
from typing import Optional
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
TARGET_HOUR_KST = int(os.environ.get('TARGET_HOUR_KST', 12))
TARGET_MINUTE_KST = int(os.environ.get('TARGET_MINUTE_KST', 10))

# ⚠️ 환경 변수에서 로드 (Render 환경에 필수) - 사용자가 지정한 하드코딩 값 유지
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
TELEGRAM_TARGET_CHAT_ID = os.environ.get('TELEGRAM_TARGET_CHAT_ID', '-1000000000')
SERVER_PORT = int(os.environ.get("PORT", 8000))

# 로깅 설정 (INFO 레벨로 주요 동작만 기록)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 특수 문자 오류 제거
if 'YOUR_BOT_TOKEN_HERE' in TELEGRAM_BOT_TOKEN or TELEGRAM_TARGET_CHAT_ID == '-1000000000':
    logger.warning("⚠️ 경고: TELEGRAM_BOT_TOKEN 또는 CHAT_ID가 기본값입니다. 환경 변수를 설정해주세요.")

# 💾 서버 RAM에서 상태 유지 (Render 재시작 시 초기화될 수 있음 - 디스크 미사용)
status = {
    "last_sent_date_kst": "1970-01-01", 
    "last_check_time_kst": "N/A",
    "next_scheduled_time_kst": "N/A",
    "last_self_ping_kst": "N/A"
}

# =========================================================
# --- [2] VIX Plotter 함수 (그래프 생성 로직) - Render 안정화 적용 ---
# =========================================================
def plot_vix_sp500(width=6.4, height=4.8) -> Optional[io.BytesIO]:
    """
    VIX와 S&P 500의 종가 추이를 비교하는 차트를 생성합니다.
    Rate Limit 회피를 위한 지수적 백오프 재시도 로직이 적용되어 있습니다.
    """
    logger.info("📈 데이터 다운로드 및 차트 생성 시작...")

    # Rate Limit 회피를 위한 지수적 백오프 재시도 로직
    max_retry = 4 # 최대 4번 시도 (1차 + 3번 재시도)
    tickers = ["^VIX", "^GSPC"]
    vix, qqq = None, None
    
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

            logger.info(f"Attempt {attempt}: Data downloaded successfully (VIX={vix.iloc[-1]:.2f}, S&P500={qqq.iloc[-1]:.0f}).")
            break # 성공적으로 다운로드 및 유효성 검사 완료
            
        except Exception as e:
            logger.warning(f"Data download failed (Attempt {attempt}): {e}")
            if attempt < max_retry:
                # ⭐️ 지수적 백오프(Exponential Backoff) 적용: 2^1=2s, 2^2=4s, 2^3=8s 대기
                sleep_time = 2 ** attempt
                logger.info(f"Applying Exponential Backoff. Waiting {sleep_time} seconds before next retry...")
                time.sleep(sleep_time)
            else:
                logger.error("Max retries exceeded. Failed to acquire data.")
                return None
    
    if vix is None or qqq is None:
        return None

    # 최종 확정된 차트 디자인 로직 적용 (그래프 로직은 변경 없음)
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
        common_dates = vix.index # 재정의
        last_vix_price = vix.iloc[-1].item()
        last_qqq_price = qqq.iloc[-1].item()
        title_text = f"VIX ({last_vix_price:.2f}) vs S&P 500 ({last_qqq_price:.2f})"
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

        # Y축 레이블 설정
        ax1.set_ylabel('S&P 500', color=qqq_color, fontsize=12, fontweight='bold', labelpad=15)
        ax2.set_ylabel('VIX', color=vix_color, fontsize=12, fontweight='bold', labelpad=15)
        
        # VIX 레벨 주석 및 수평선 추가
        try:
            # 전체 데이터 기간의 90% 지점 날짜를 찾습니다.
            new_text_x_pos = common_dates[int(len(common_dates)*0.9)]
        except:
             # 데이터가 너무 적을 경우의 안전 장치
             new_text_x_pos = common_dates[-1] + timedelta(days=1)
        
        # VIX 주석 (오른쪽 정렬)
        ax2.text(new_text_x_pos, 15.5, "VIX 15 (탐욕/매도)", color='yellow', fontsize=new_fontsize, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')
        ax2.text(new_text_x_pos, 30.5, "VIX 30 (경고)", color='peru', fontsize=new_fontsize, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')
        ax2.text(new_text_x_pos, 40.5, "VIX 40 (공포/매수)", color='lightGreen', fontsize=new_fontsize, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')
        
        # VIX 수평선
        ax2.axhline(y=15, color='yellow', linestyle='--', linewidth=1.2, alpha=0.8)
        ax2.axhline(y=30, color='peru', linestyle='--', linewidth=1.0, alpha=0.8)
        ax2.axhline(y=40, color='lightGreen', linestyle='--', linewidth=1.2, alpha=0.8)
        
        # 제목 및 여백 최소화
        fig.suptitle(title_text, color='white', fontsize=12, fontweight='bold', y=0.98) 
        fig.tight_layout(rect=[0.025, 0.05, 0.975, 1.0]) 
        
        # ⭐️ 메모리 버퍼에 PNG 이미지로 저장 (디스크 미사용 핵심) ⭐️
        plot_data = io.BytesIO()
        plt.savefig(plot_data, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1) 
        plot_data.seek(0)
        
        plt.close(fig) # **매우 중요: 메모리 누수 방지**
        logger.info("✅ 차트 생성 완료 (메모리 저장).")
        return plot_data

    except Exception as e:
        logger.error(f"❌ 차트 생성 중 예외 발생: {e}", exc_info=True)
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
                logger.info(f"텔레그램 발송 요청 시작 (시도 {attempt + 1}/3, Chat ID: {chat_id})...")
                async with session.post(url, data=data) as response:
                    response.raise_for_status() # HTTP 오류 발생 시 예외 발생
                    response_json = await response.json()
                    if response_json.get('ok'):
                        logger.info("✅ 텔레그램 발송 성공!")
                        return True
                    else:
                        error_desc = response_json.get('description', 'Unknown Error')
                        raise Exception(f"Telegram API Error: {error_desc}")
                        
        except Exception as e:
            logger.warning(f"❌ 텔레그램 전송 중 오류 발생 (시도 {attempt + 1}/3): {e}. 잠시 후 재시도.")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt) # Exponential Backoff: 1s, 2s 대기
            
    logger.error("텔레그램 발송 최종 실패.")
    return False

async def run_and_send_plot() -> bool:
    """차트 생성 및 전송의 전체 프로세스를 실행합니다."""
    global status
    
    if 'YOUR_BOT_TOKEN_HERE' in TELEGRAM_BOT_TOKEN or TELEGRAM_TARGET_CHAT_ID == '-1000000000':
        logger.error("텔레그램 토큰 또는 Chat ID가 기본값입니다. 발송을 건너뜁니다.")
        return False
        
    plot_buffer = plot_vix_sp500()
    if not plot_buffer:
        logger.error("차트 생성 실패로 인해 전송을 건너뛰고 다음 목표 시간을 다시 계산합니다.")
        return False
    
    # 캡션을 위해 최신 데이터 가져오기 (차트 생성 실패를 대비해 별도 로직 유지)
    latest_vix, latest_gspc, latest_date_utc = "N/A", "N/A", "최신 데이터 확보 실패"
    try:
        # 짧은 기간으로 데이터를 가져와서 캡션에 사용 (메모리 사용)
        data = yf.download(["^VIX", "^GSPC"], period="5d", progress=False, timeout=10)
        vix_data = data['Close']['^VIX'].dropna()
        gspc_data = data['Close']['^GSPC'].dropna()

        if not vix_data.empty and not gspc_data.empty:
            latest_vix = vix_data.iloc[-1].item()
            latest_gspc = gspc_data.iloc[-1].item()
            # VIX와 GSPC의 마지막 인덱스 중 더 최근 날짜를 사용 (일반적으로 같음)
            latest_date_utc = max(vix_data.index[-1], gspc_data.index[-1]).strftime('%Y-%m-%d')
    except Exception:
        logger.warning("캡션에 사용할 최신 VIX/S&P 500 데이터 확보 실패. 'N/A' 사용.")


    caption = (
        f"\n🗓️ {latest_date_utc} (미국 시장 마감 기준)\n"
        f"📉 VIX (변동성): **{latest_vix:.2f}**\n"
        f"📈 S&P 500 (지수): **{latest_gspc:.0f}**\n\n"
        f"VIX and the S&P 500 typically move in opposite directions.\n"
    )

    success = await send_photo_via_http(TELEGRAM_TARGET_CHAT_ID, plot_buffer, caption)
    plot_buffer.close() # 메모리 버퍼 닫기 (메모리 해제)

    if success:
        current_kst = datetime.now(KST_TZ)
        status['last_sent_date_kst'] = current_kst.strftime("%Y-%m-%d")
        logger.info(f"성공적으로 발송 완료. 마지막 발송 날짜 업데이트: {status['last_sent_date_kst']}")
    
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
    
    logger.info(f"🔍 모니터링 시작. 다음 발송 예정 시간 (KST): {status['next_scheduled_time_kst']}")
    
    while True:
        await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
        
        current_kst = datetime.now(KST_TZ)
        status['last_check_time_kst'] = current_kst.strftime("%Y-%m-%d %H:%M:%S KST")
        
        # 발송 조건 확인 (하루에 한 번, 지정된 시간에 발송)
        target_date_kst = next_target_time_kst.strftime("%Y-%m-%d")

        if current_kst >= next_target_time_kst and \
           current_kst < next_target_time_kst + timedelta(minutes=1) and \
           target_date_kst != status['last_sent_date_kst']:

            logger.info(f"⏰ 발송 시간 도달 (KST: {current_kst.strftime('%H:%M:%S')}). 작업 실행.")
            
            # 발송 로직 실행
            await run_and_send_plot()
            
            # 다음 목표 시간 업데이트
            next_target_time_kst = calculate_next_target_time(current_kst)
            status['next_scheduled_time_kst'] = next_target_time_kst.strftime("%Y-%m-%d %H:%M:%S KST")
            logger.info(f"➡️ 다음 발송 예정 시간 (KST): {status['next_scheduled_time_kst']}")
            
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
    logger.info(f"🛡️ 내부 셀프 핑 루프 시작. 5분마다 {ping_url}로 요청을 보냅니다.")
    
    async with aiohttp.ClientSession() as session:
        while True:
            await asyncio.sleep(5 * 60) # 5분 대기
            
            try:
                # HEAD 요청은 GET보다 가볍습니다.
                async with session.head(ping_url, timeout=10) as response:
                    if response.status == 200:
                        status['last_self_ping_kst'] = datetime.now(KST_TZ).strftime("%Y-%m-%d %H:%M:%S KST")
                        logger.debug(f"✅ 셀프 핑 성공: {status['last_self_ping_kst']}")
                    else:
                        logger.warning(f"❌ 셀프 핑 실패 (Status: {response.status})")
                        
            except Exception as e:
                logger.error(f"❌ 셀프 핑 중 예외 발생: {e}")


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
    logger.info("🚀 백그라운드 스케줄링 및 셀프 핑 루프가 시작되었습니다.")

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
        raise HTTPException(status_code=400, detail="시간과 분은 정수여야 합니다.")
        
    # 유효성 검사
    if not (0 <= hour_int <= 23 and 0 <= minute_int <= 59):
        raise HTTPException(status_code=400, detail="유효하지 않은 시간(0-23) 또는 분(0-59)입니다.")
        
    # 전역 변수 업데이트
    TARGET_HOUR_KST = hour_int
    TARGET_MINUTE_KST = minute_int
    
    # 변경 사항을 즉시 반영하여 다음 목표 시간 재계산
    now_kst = datetime.now(KST_TZ)
    next_target_time_kst = calculate_next_target_time(now_kst)
    status['next_scheduled_time_kst'] = next_target_time_kst.strftime("%Y-%m-%d %H:%M:%S KST")

    logger.info(f"⏰ 스케줄링 시간 변경됨: {TARGET_HOUR_KST:02d}:{TARGET_MINUTE_KST:02d} KST. 다음 발송 시간 업데이트됨: {status['next_scheduled_time_kst']}")
    
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
                <h1>✅ VIX 스케줄러 상태 (KST)</h1>

                <h2>현재 스케줄 상태</h2>
                <p>현재 KST 시간: <span class="highlight">{current_kst.strftime('%Y-%m-%d %H:%M:%S KST')}</span></p>
                <p>현재 설정 발송 시간: <span class="highlight">{TARGET_HOUR_KST:02d}:{TARGET_MINUTE_KST:02d} KST</span></p>
                <p>다음 발송 예정 시간: <span class="highlight">{status.get('next_scheduled_time_kst')}</span></p>
                <p>마지막 성공 발송 날짜: <span class="highlight">{status.get('last_sent_date_kst')}</span></p>
                <p>🛡️ 마지막 셀프 핑: <span class="highlight">{status.get('last_self_ping_kst')}</span></p>

                <div class="time-setting">
                    <h2>발송 시간 설정 (KST)</h2>
                    <form action="/set-time" method="POST">
                        <div class="form-group">
                            <label for="hour">시 (Hour, 0-23):</label>
                            <input type="number" id="hour" name="hour" min="0" max="23" value="{TARGET_HOUR_KST}" required>
                            <label for="minute">분 (Minute, 0-59):</label>
                            <input type="number" id="minute" name="minute" min="0" max="59" value="{TARGET_MINUTE_KST}" required>
                        </div>
                        <div class="form-group" style="justify-content: flex-end;">
                            <button type="submit">스케줄 시간 변경</button>
                        </div>
                    </form>
                </div>

                <div class="alert">
                    🔔 **중요**: 이 서비스를 유지하기 위해서는 외부 모니터링 서비스(예: UptimeRobot)를 설정하여 이 URL에 주기적으로(5분마다) 요청을 보내야 합니다.
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
