import os
import sys
import asyncio
import aiohttp
import io
import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

# FastAPI 관련 임포트 (RedirectResponse, Form, Request 추가)
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from starlette.responses import RedirectResponse

# 그래프/데이터 관련 외부 라이브러리
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib

# Matplotlib 백엔드 설정 (헤드리스 서버 환경을 위해 필수)
matplotlib.use('Agg')

# =========================================================
# --- [1] 설정 및 환경 변수 로드 및 전역 상태 ---
# =========================================================
# 한국 시간 (KST) 타임존 설정
KST_TZ = ZoneInfo("Asia/Seoul")
MONITOR_INTERVAL_SECONDS = 60 # 1분마다 시간 체크

# ⏰ 전역 상태: 사용자가 설정할 수 있는 발송 시간 (KST)
TARGET_HOUR_KST = int(os.environ.get('TARGET_HOUR_KST', 11))
TARGET_MINUTE_KST = int(os.environ.get('TARGET_MINUTE_KST', 0))

# ⚠️ 환경 변수에서 로드 (Render 환경에 필수)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
TELEGRAM_TARGET_CHAT_ID = os.environ.get('TELEGRAM_TARGET_CHAT_ID', '-1000000000')
SERVER_PORT = int(os.environ.get("PORT", 8000))

# 로깅 설정 (INFO 레벨로 주요 동작만 기록)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 특수 문자 오류 제거
if 'YOUR_BOT_TOKEN_HERE' in TELEGRAM_BOT_TOKEN or TELEGRAM_TARGET_CHAT_ID == '-1000000000':
    logger.warning("⚠️ 경고: TELEGRAM_BOT_TOKEN 또는 CHAT_ID가 기본값입니다. 환경 변수를 설정해주세요.")

# 서버 RAM에서 상태 유지 (Render 재시작 시 초기화될 수 있음)
status = {
    "last_sent_date_kst": "1970-01-01", 
    "last_check_time_kst": "N/A",
    "next_scheduled_time_kst": "N/A",
    "last_self_ping_kst": "N/A"
}

# =========================================================
# --- [2] VIX Plotter 함수 (그래프 생성 로직) ---
# =========================================================
def plot_vix_sp500(width=10, height=6) -> Optional[io.BytesIO]:
    """VIX와 S&P 500의 6개월 종가 추이를 비교하는 차트를 생성합니다."""
    logger.info("📈 데이터 다운로드 및 차트 생성 시작...")
    
    try:
        # 데이터 다운로드: 6개월치 (^VIX: VIX 지수, ^GSPC: S&P 500)
        tickers = ["^VIX", "^GSPC"]
        data = yf.download(tickers, period="6mo", progress=False)
        
        vix_data = data['Close']['^VIX'].dropna()
        gspc_data = data['Close']['^GSPC'].dropna()

        if vix_data.empty or gspc_data.empty:
            logger.error("데이터 수집 실패: VIX 또는 S&P 500 데이터가 비어있습니다.")
            return None

        # 듀얼 축 플롯 생성
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax1 = plt.subplots(figsize=(width, height))
        
        # 첫 번째 축: VIX (좌측)
        color_vix = '#0070FF' # 파란색
        ax1.set_xlabel('날짜', fontsize=10)
        ax1.set_ylabel('VIX (좌측)', color=color_vix, fontsize=12, fontweight='bold')
        ax1.plot(vix_data.index, vix_data.values, color=color_vix, linewidth=2, label='VIX (변동성)', alpha=0.8)
        ax1.tick_params(axis='y', labelcolor=color_vix)
        ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        # 두 번째 축: S&P 500 (우측)
        ax2 = ax1.twinx()  
        color_gspc = '#FF4500' # 주황색
        ax2.set_ylabel('S&P 500 (우측)', color=color_gspc, fontsize=12, fontweight='bold')
        ax2.plot(gspc_data.index, gspc_data.values, color=color_gspc, linewidth=2, label='S&P 500 (지수)', linestyle='-')
        ax2.tick_params(axis='y', labelcolor=color_gspc)
        ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

        # X축 날짜 포맷팅
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        
        # 제목 설정
        plt.title('VIX와 S&P 500 6개월 추이 비교', fontsize=14, fontweight='bold')
        fig.tight_layout() 
        
        # 메모리 버퍼에 PNG 이미지로 저장
        plot_data = io.BytesIO()
        plt.savefig(plot_data, format='png', bbox_inches='tight', dpi=100)
        plot_data.seek(0)
        
        plt.close(fig) # **매우 중요: 메모리 누수 방지**
        logger.info("✅ 차트 생성 완료.")
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
    
    data = {
        'chat_id': chat_id,
        'caption': caption,
        'parse_mode': 'Markdown'
    }
    
    files = {
        'photo': ('vix_gspc_chart.png', photo_bytes, 'image/png')
    }
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        try:
            logger.info(f"텔레그램 발송 요청 시작 (Chat ID: {chat_id})...")
            async with session.post(url, data=data, files=files) as response:
                if response.status == 200:
                    logger.info("✅ 텔레그램 발송 성공!")
                    return True
                else:
                    response_text = await response.text()
                    logger.error(f"❌ 텔레그램 발송 실패 (Status: {response.status}, Response: {response_text})")
                    return False
        except Exception as e:
            logger.error(f"❌ 텔레그램 발송 중 예외 발생: {e}", exc_info=True)
            return False

async def run_and_send_plot() -> bool:
    """차트 생성 및 전송의 전체 프로세스를 실행합니다."""
    global status
    global TARGET_HOUR_KST, TARGET_MINUTE_KST # 캡션에 현재 설정된 시간 반영
    
    if 'YOUR_BOT_TOKEN_HERE' in TELEGRAM_BOT_TOKEN or TELEGRAM_TARGET_CHAT_ID == '-1000000000':
        logger.error("텔레그램 토큰 또는 Chat ID가 기본값입니다. 발송을 건너뜁니다.")
        return False
        
    plot_buffer = plot_vix_sp500()
    if not plot_buffer:
        logger.error("차트 생성 실패로 인해 전송을 건너뜁니다.")
        return False
    
    # 임시 데이터 가져오기 (캡션을 위해)
    try:
        data = yf.download(["^VIX", "^GSPC"], period="5d", progress=False)
        vix_data = data['Close']['^VIX'].dropna()
        gspc_data = data['Close']['^GSPC'].dropna()

        latest_vix = vix_data.iloc[-1]
        latest_gspc = gspc_data.iloc[-1]
        latest_date_utc = vix_data.index[-1].strftime('%Y-%m-%d')
    except Exception:
        latest_vix = "N/A"
        latest_gspc = "N/A"
        latest_date_utc = "최신 데이터 확보 실패"

    caption = (
        f"**[일간 변동성 지수 모니터링]**\n"
        f"🗓️ 기준일: {latest_date_utc} (미국 시장 마감 기준)\n"
        f"📉 VIX (변동성): **{latest_vix:.2f}**\n"
        f"📈 S&P 500 (지수): **{latest_gspc:.0f}**\n\n"
        f"VIX는 S&P 500 지수와 일반적으로 역의 상관관계를 가집니다.\n"
        f"스케줄링 시간(KST): {TARGET_HOUR_KST:02d}:{TARGET_MINUTE_KST:02d}"
    )

    success = await send_photo_via_http(TELEGRAM_TARGET_CHAT_ID, plot_buffer, caption)
    plot_buffer.close()

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
