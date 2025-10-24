import os
import sys
import asyncio
import aiohttp
import io
from datetime import datetime, timedelta
from typing import Optional, Tuple

# FastAPI 및 uvicorn import (웹 서비스 구동을 위해 필요)
from fastapi import FastAPI
import uvicorn

# =========================================================
# --- [1] 설정 및 환경 변수 로드 ---
# =========================================================
# 한국 시간 (KST)은 UTC+9입니다.
KST_OFFSET_HOURS = 9 
TARGET_HOUR_KST = 8 # 한국 시간 오전 8시 발송 목표
MONITOR_INTERVAL_SECONDS = 60 # 1분마다 시간 체크

# ⚠️ 환경 변수에서 로드 (Render 환경에 필수)
# 환경 변수에 TELEGRAM_BOT_TOKEN과 TELEGRAM_TARGET_CHAT_ID를 설정해야 합니다.
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE') 
TELEGRAM_TARGET_CHAT_ID = os.environ.get('TELEGRAM_TARGET_CHAT_ID', '-1000000000') 

# 서버 RAM에서 상태 유지 (Render 재시작 시 초기화될 수 있음)
# last_sent_date_kst: 마지막으로 성공적으로 발송된 날짜를 추적합니다.
# last_check_time_kst: 핑 체크 엔드포인트에서 마지막 확인 시간을 보여줍니다.
status = {"last_sent_date_kst": "1970-01-01", "last_check_time_kst": "N/A"}

# 텔레그램 설정 검사
if 'YOUR_BOT_TOKEN_HERE' in TELEGRAM_BOT_TOKEN or TELEGRAM_TARGET_CHAT_ID == '-1000000000':
    print("⚠️ 경고: TELEGRAM_BOT_TOKEN 또는 CHAT_ID가 기본값입니다. 환경 변수를 설정해주세요.")


# =========================================================
# --- [2] VIX Plotter 함수 (임시 복제) ---
# =========================================================
# 이 함수는 이전 파일에서 그대로 가져온 그래프 생성 로직입니다.
def plot_vix_sp500(width=6.4, height=4.8):
    """
    VIX/S&P 500 데이터를 다운로드하고 그래프를 생성하여 BytesIO로 반환합니다.
    """
    import yfinance as yf
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib
    
    # vix_plotter.py의 설정 반영
    matplotlib.use('Agg')
    plt.style.use('dark_background')
    matplotlib.rcParams['font.family'] = 'Malgun Gothic' # 한국어 폰트
    
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d") # 최근 6개월 데이터

    try:
        # 그래프 데이터 생성 중 출력
        print("그래프 데이터 생성 중... (yfinance 다운로드)")
        vix_df = yf.download("^VIX", start=start_date, end=None, progress=False)
        qqq_df = yf.download("^GSPC", start=start_date, end=None, progress=False)
        
        vix = vix_df["Close"].dropna()
        qqq = qqq_df["Close"].dropna()
        common_dates = vix.index.intersection(qqq.index)
        vix = vix.loc[common_dates]
        qqq = qqq.loc[common_dates]
        if vix.empty or qqq.empty: 
            print("yfinance에서 데이터를 가져오지 못했습니다.")
            return None

        # 플로팅 로직 (생략 없는 전체 코드)
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
        print(f"그래프 생성 중 오류 발생: {e}")
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
                        print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] 텔레그램 전송 성공. (채널: {chat_id})")
                        return True
                    else:
                        error_desc = response_json.get('description', 'Unknown Error')
                        raise Exception(f"Telegram API Error: {error_desc}")
                        
        except Exception as e:
            print(f"텔레그램 발송 실패 (시도 {attempt + 1}/3): {e}. 잠시 후 재시도.")
            await asyncio.sleep(2 ** attempt) 
            
    print("텔레그램 발송 최종 실패.")
    return False


async def run_and_send_plot():
    """
    그래프를 생성하고 전송을 실행하는 메인 함수입니다.
    """
    print("VIX/S&P 500 그래프 생성 및 전송 시작...")
    
    # 1. 그래프 데이터 (메모리 내 바이트) 생성
    plot_data = plot_vix_sp500(width=6.4, height=4.8)
    
    if not plot_data:
        print("그래프 데이터 생성 실패로 전송 중단.")
        return False # 전송 실패

    # 2. 이미지 전송 (HTTP API 사용)
    current_kst = datetime.utcnow() + timedelta(hours=KST_OFFSET_HOURS)
    caption = (
        f"VIX V.S. S&P 500 (2025년 4월 이후)"
        # f"[15↓:탐욕(매도), 40↑:공포(매수)]"
    )
    
    success = await send_photo_via_http(TELEGRAM_TARGET_CHAT_ID, plot_data, caption)

    # 3. 바이트 객체 정리 (메모리에서 제거)
    plot_data.close() 
    print("메모리 바이트 객체 정리 완료.")
    
    return success


# =========================================================
# --- [4] 스케줄링 및 루프 로직 ---
# =========================================================

async def main_monitor_loop():
    """
    Render 백그라운드에서 실행될 메인 스케줄링 루프입니다. (1분 간격 체크)
    """
    print("--- VIX 그래프 모니터링 스케줄러 (백그라운드 - 1분 주기) 시작 ---")
    
    while True:
        try:
            now_utc = datetime.utcnow()
            now_kst = now_utc + timedelta(hours=KST_OFFSET_HOURS)
            today_kst_str = now_kst.strftime("%Y-%m-%d")
            current_time_kst_str = now_kst.strftime("%H:%M")
            
            # Health Check 엔드포인트에서 마지막 확인 시간을 보여주기 위해 업데이트
            status['last_check_time_kst'] = now_kst.strftime("%Y-%m-%d %H:%M:%S") 
            
            current_weekday = now_kst.weekday() # Mon=0, Tue=1, ..., Sat=5, Sun=6
            
            # 1. 유효 요일 확인 (화요일(1) ~ 토요일(5)) - 일요일(6), 월요일(0) 제외
            is_valid_day = 1 <= current_weekday <= 5
            
            # 2. 목표 시간 확인 (정확히 08:00 KST)
            target_time_str = f"{TARGET_HOUR_KST:02d}:00"
            is_target_time = (current_time_kst_str == target_time_str)
            
            # 3. 오늘 발송 완료 여부 확인
            is_already_sent = (status['last_sent_date_kst'] == today_kst_str)

            log_level = "INFO"
            log_message = ""
            
            if is_valid_day and is_target_time and not is_already_sent:
                # 조건 충족: 정각 8시 발송 시작
                log_level = "ACTION"
                log_message = "정각 8시 도달, 발송 시작"
                
                print(f"[{log_level}] KST:{current_time_kst_str} | DAY:{current_weekday} | {log_message}")

                success = await run_and_send_plot()
                
                if success:
                    # 발송 성공 시에만 상태 업데이트 (중복 발송 방지)
                    status['last_sent_date_kst'] = today_kst_str
                
            elif not is_valid_day:
                # SKIP 1: 비영업일 (일, 월)
                log_level = "INFO"
                log_message = "SKIP (비영업일: 일/월)"
                print(f"[{log_level}] KST:{current_time_kst_str} | DAY:{current_weekday} | {log_message}")

            elif is_already_sent:
                # SKIP 2: 이미 발송됨
                log_level = "INFO"
                log_message = f"SKIP (금일 {today_kst_str} 발송 완료됨)"
                print(f"[{log_level}] KST:{current_time_kst_str} | DAY:{current_weekday} | {log_message}")

            elif is_valid_day and not is_target_time and not is_already_sent:
                # SKIP 3: 유효 요일이지만, 시간이 8시 정각이 아님 (사용자 요청 WARNING)
                log_level = "WARNING"
                log_message = f"SKIP (시간 불일치) - {target_time_str} 정각 대기 중"
                print(f"[{log_level}] KST:{current_time_kst_str} | DAY:{current_weekday} | {log_message}")

            else:
                 # 기타 오류 방지용 로깅 (실제 작동 시 발생할 가능성 낮음)
                log_level = "INFO"
                log_message = "WAIT (Monitoring)"
                print(f"[{log_level}] KST:{current_time_kst_str} | DAY:{current_weekday} | {log_message}")

        except Exception as e:
            print(f"[ERROR] 스케줄링 루프 중 치명적인 오류 발생: {e}. 60초 후 재시도.")
            
        # Fixed 1-minute sleep (Render 슬립 방지용)
        await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
            
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
    print("FastAPI Server Startup: Launching main_monitor_loop as a background task.")
    asyncio.create_task(main_monitor_loop())

# Health Check Endpoint (외부 모니터링 서비스(UptimeRobot 등)가 사용자의 서버 슬립을 방지하는 용도)
@app.get("/")
@app.head("/") 
async def health_check():
    return {
        "status": "running", 
        "message": "VIX scheduler is active in the background (1-minute check).",
        "last_plot_sent_date_kst": status.get('last_sent_date_kst'),
        "last_check_time_kst": status.get('last_check_time_kst'),
        "check_interval_seconds": MONITOR_INTERVAL_SECONDS
    }

# =========================================================
# --- [6] 실행 ---
# =========================================================
if __name__ == '__main__':
    # Render는 환경 변수로 PORT를 제공합니다.
    port = int(os.environ.get("PORT", 8000))
    
    # 기본 로그 출력 방식으로 복귀 
    print(f"Starting uvicorn server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
