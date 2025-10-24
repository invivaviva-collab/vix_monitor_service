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

# FastAPI imports
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from starlette.responses import RedirectResponse

# Graph/Data external libraries
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import numpy as np

# Set Matplotlib backend (required for headless server environment)
matplotlib.use('Agg')

# =========================================================
# --- [1] Configuration, Environment Variables, and Global State ---
# =========================================================
# Set Korean Standard Time (KST) timezone
KST_TZ = ZoneInfo("Asia/Seoul")
MONITOR_INTERVAL_SECONDS = 60 # Check time every 1 minute

# ⏰ Global State: User-configurable send time (KST)
TARGET_HOUR_KST = int(os.environ.get('TARGET_HOUR_KST', 7))
TARGET_MINUTE_KST = int(os.environ.get('TARGET_MINUTE_KST', 20))

# ⚠️ Load from environment variables (essential for Render environment)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
TELEGRAM_TARGET_CHAT_ID = os.environ.get('TELEGRAM_TARGET_CHAT_ID', '-1000000000')
SERVER_PORT = int(os.environ.get("PORT", 8000))

# Logging setup (INFO level for main operations)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for default credentials
if 'YOUR_BOT_TOKEN_HERE' in TELEGRAM_BOT_TOKEN or TELEGRAM_TARGET_CHAT_ID == '-1000000000':
    logger.warning("⚠️ Warning: TELEGRAM_BOT_TOKEN or CHAT_ID is set to default. Please configure environment variables.")

# 💾 Server RAM state (may reset upon Render restart - no disk usage)
status = {
    "last_sent_date_kst": "1970-01-01", 
    "last_check_time_kst": "N/A",
    "next_scheduled_time_kst": "N/A",
    "last_self_ping_kst": "N/A"
}

# =========================================================
# --- [2] VIX Plotter Function (Chart Generation Logic) ---
# =========================================================
def plot_vix_sp500(width=6.4, height=4.8) -> Optional[Tuple[io.BytesIO, float, float, str]]:
    """
    Generates a comparative chart of VIX and S&P 500 closing prices,
    and returns the chart buffer along with the latest data.
    """
    logger.info("📈 Starting data download and chart generation...")

    max_retry = 4 
    tickers = ["^VIX", "^GSPC"]
    vix, qqq = None, None
    latest_vix, latest_gspc, latest_date_utc = 0.0, 0.0, "N/A" 
    
    # Using a fixed start date for consistent comparison
    start_date = "2025-04-01" 
    
    for attempt in range(1, max_retry + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_retry}: Downloading VIX and S&P 500 data (start={start_date})...")
            
            # Download data using yfinance
            data_all = yf.download(tickers, start=start_date, progress=False, timeout=20)
            
            # Extract Close data
            vix_df = data_all['Close']['^VIX'].dropna()
            gspc_df = data_all['Close']['^GSPC'].dropna()
            
            # Align common dates
            common_dates = vix_df.index.intersection(gspc_df.index)
            vix = vix_df.loc[common_dates]
            qqq = gspc_df.loc[common_dates]

            # Data validation
            if vix.empty or qqq.empty:
                raise ValueError("Downloaded data is empty after aligning dates.")

            # ⭐️ Extract latest data for caption ⭐️
            latest_vix = vix.iloc[-1].item()
            latest_gspc = qqq.iloc[-1].item()
            # Use the latest date from either index
            latest_date_utc = max(vix.index[-1], qqq.index[-1]).strftime('%Y-%m-%d')

            logger.info(f"Attempt {attempt}: Data downloaded successfully (VIX={latest_vix:.2f}, S&P500={latest_gspc:.0f}).")
            break 
            
        except Exception as e:
            logger.warning(f"Data download failed (Attempt {attempt}): {e}")
            if attempt < max_retry:
                # Apply Exponential Backoff
                sleep_time = 5 ** attempt
                logger.info(f"Applying Exponential Backoff. Waiting {sleep_time} seconds before next retry...")
                time.sleep(sleep_time)
            else:
                logger.error("Max retries exceeded. Failed to acquire data.")
                return None
    
    if vix is None or qqq is None:
        return None

    # Apply final chart design logic
    try:
        plt.style.use('dark_background')
        
        fig, ax1 = plt.subplots(figsize=(width, height)) 
        ax2 = ax1.twinx()
        
        # Set background color
        fig.patch.set_facecolor('#222222')
        ax1.set_facecolor('#2E2E2E')
        ax2.set_facecolor('#2E2E2E')
        
        # Data and colors
        common_dates = vix.index 
        title_text = f"VIX ({latest_vix:.2f}) vs S&P 500 ({latest_gspc:.0f})"
        vix_color = '#FF6B6B' # VIX color (Red tone)
        qqq_color = '#6BCBFF' # S&P 500 color (Blue tone)
        new_fontsize = 8 * 1.3
        
        # Plotting
        ax2.plot(common_dates, vix.values, color=vix_color, linewidth=1.5)
        # S&P 500 (GSPC)
        ax1.plot(common_dates, qqq.values, color=qqq_color, linewidth=1.5)
        
        # X-axis date format and interval setting
        formatter = mdates.DateFormatter('%Y-%m-%d') 
        ax1.xaxis.set_major_formatter(formatter)
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4)) 
        fig.autofmt_xdate(rotation=45)

        # Y-axis label setting
        ax1.set_ylabel('S&P 500 Index', color=qqq_color, fontsize=12, fontweight='bold', labelpad=5)
        ax2.set_ylabel('VIX', color=vix_color, fontsize=12, fontweight='bold', labelpad=5)
        
        # Add VIX level annotations and horizontal lines
        try:
            # Find the date position for annotation
            new_text_x_pos = common_dates[int(len(common_dates)*0.9)]
        except:
             # Safety net for very little data
            new_text_x_pos = common_dates[-1] + timedelta(days=1)
        
        # VIX annotations
        ax2.text(new_text_x_pos, 15.5, "VIX 15 (Greed/Sell)", color='yellow', fontsize=new_fontsize, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')
        ax2.text(new_text_x_pos, 30.5, "VIX 30 (Warning)", color='peru', fontsize=new_fontsize, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')
        ax2.text(new_text_x_pos, 40.5, "VIX 40 (Fear/Buy)", color='lightGreen', fontsize=new_fontsize, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')
        
        # VIX horizontal lines
        ax2.axhline(y=15, color='yellow', linestyle='--', linewidth=1.2, alpha=0.8)
        ax2.axhline(y=30, color='peru', linestyle='--', linewidth=1.0, alpha=0.8)
        ax2.axhline(y=40, color='lightGreen', linestyle='--', linewidth=1.2, alpha=0.8)
        
        # Title and minimal margin
        fig.suptitle(title_text, color='white', fontsize=12, fontweight='bold', y=0.98) 
        fig.tight_layout(rect=[0.025, 0.025, 1, 1]) 
        
        # ⭐️ Save to memory buffer as PNG image (Crucial: no disk usage) ⭐️
        plot_data = io.BytesIO()
        plt.savefig(plot_data, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1) 
        plot_data.seek(0)
        
        plt.close(fig) # **VERY IMPORTANT: Prevent memory leak**
        logger.info("✅ Chart generation complete (saved to memory).")
        
        # ⭐️ Return chart buffer and latest data as a tuple ⭐️
        return plot_data, latest_vix, latest_gspc, latest_date_utc

    except Exception as e:
        logger.error(f"❌ Exception during chart generation: {e}", exc_info=True)
        return None

# =========================================================
# --- [3] Telegram Sending Function (HTTP API) ---
# =========================================================
async def send_photo_via_http(chat_id: str, photo_bytes: io.BytesIO, caption: str) -> bool:
    """Sends the chart image to the Telegram bot."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    data = aiohttp.FormData()
    data.add_field('chat_id', chat_id)
    data.add_field('caption', caption)
    data.add_field('parse_mode', 'Markdown')
    # ⭐️ Pass the io.BytesIO object directly to the photo field ⭐️
    data.add_field('photo', 
                    photo_bytes, 
                    filename='vix_gspc_chart.png', 
                    content_type='image/png')

    # Add retry logic (for network issues)
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                logger.info(f"Telegram send request initiated (Attempt {attempt + 1}/3, Chat ID: {chat_id})...")
                async with session.post(url, data=data) as response:
                    response.raise_for_status() # Raise exception for HTTP errors
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
                await asyncio.sleep(2 ** attempt) # Exponential Backoff: 1s, 2s wait
            
    logger.error("Telegram send final failure.")
    return False

async def run_and_send_plot() -> bool:
    """Executes the full process of chart generation and sending."""
    global status
    
    if 'YOUR_BOT_TOKEN_HERE' in TELEGRAM_BOT_TOKEN or TELEGRAM_TARGET_CHAT_ID == '-1000000000':
        logger.error("Telegram token or Chat ID is set to default. Skipping send.")
        return False
        
    # ⭐️ Reflect the change in the return value of plot_vix_sp500 ⭐️
    plot_result = plot_vix_sp500()
    
    if not plot_result:
        logger.error("Chart generation failed. Skipping send and recalculating next target time.")
        return False
    
    plot_buffer, latest_vix, latest_gspc, latest_date_utc = plot_result
    
    # Latest data is already fetched inside plot_vix_sp500

    caption = (
        f"\n🗓️ {latest_date_utc} (US Market Close)\n"
        f"📉 VIX (Volatility): **{latest_vix:.2f}**\n"
        f"📈 S&P 500 (Index): **{latest_gspc:.0f}**\n\n"
        f"VIX and the S&P 500 typically move in opposite directions.\n"
    )

    success = await send_photo_via_http(TELEGRAM_TARGET_CHAT_ID, plot_buffer, caption)
    plot_buffer.close() # Close memory buffer (release memory)

    if success:
        current_kst = datetime.now(KST_TZ)
        status['last_sent_date_kst'] = current_kst.strftime("%Y-%m-%d")
        logger.info(f"Successfully sent. Last sent date updated: {status['last_sent_date_kst']}")
    
    return success

# =========================================================
# --- [4] Scheduling and Loop Logic ---
# =========================================================
def calculate_next_target_time(now_kst: datetime) -> datetime:
    """Calculates the next target send time (KST) based on the current time (uses global variables)."""
    global TARGET_HOUR_KST, TARGET_MINUTE_KST
    
    target_time_today = now_kst.replace(
        hour=TARGET_HOUR_KST, 
        minute=TARGET_MINUTE_KST, 
        second=0, 
        microsecond=0
    )
    
    if now_kst >= target_time_today:
        # If today's target time has passed, set it for tomorrow
        next_target = target_time_today + timedelta(days=1)
    else:
        # If today's target time has not yet arrived, set it for today
        next_target = target_time_today
        
    return next_target

async def main_monitor_loop():
    """Runs every minute, checks the send time, and triggers the job.
    Includes a top-level try/except for maximum stability."""
    global status
    
    # Initial setup of next send time
    now_kst = datetime.now(KST_TZ)
    next_target_time_kst = calculate_next_target_time(now_kst)
    status['next_scheduled_time_kst'] = next_target_time_kst.strftime("%Y-%m-%d %H:%M:%S KST")
    
    logger.info(f"🔍 Monitoring started. Next scheduled time (KST): {status['next_scheduled_time_kst']}")
    
    while True:
        # Sleep first to wait for the next interval
        await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
        
        # ⭐️ Top-level try/except block for maximum stability ⭐️
        try:
            current_kst = datetime.now(KST_TZ)
            status['last_check_time_kst'] = current_kst.strftime("%Y-%m-%d %H:%M:%S KST")
            
            # Output schedule check log every minute at WARNING level.
            logger.warning(f"Monitor: Checking schedule (KST: {current_kst.strftime('%H:%M:%S')}).")
            
            # Check send condition (once per day, at the specified time)
            target_date_kst = next_target_time_kst.strftime("%Y-%m-%d")

            # Logic: Added day of week check (Monday=0, Sunday=6)
            is_monday_or_sunday = (current_kst.weekday() == 0) or (current_kst.weekday() == 6)

            if current_kst >= next_target_time_kst and \
               current_kst < next_target_time_kst + timedelta(minutes=1) and \
               target_date_kst != status['last_sent_date_kst']:

                if is_monday_or_sunday:
                    # If it's Monday (0) or Sunday (6), skip sending
                    logger.info(f"🚫 Skip send: Today is Monday or Sunday (KST). Only updating next scheduled time.")
                else:
                    logger.info(f"⏰ Send time reached (KST: {current_kst.strftime('%H:%M:%S')}). Executing job.")
                    
                    # Execute send logic
                    await run_and_send_plot()
                
                # Update the next target time (regardless of send success)
                next_target_time_kst = calculate_next_target_time(current_kst)
                status['next_scheduled_time_kst'] = next_target_time_kst.strftime("%Y-%m-%d %H:%M:%S KST")
                logger.info(f"➡️ Next scheduled time (KST): {status['next_scheduled_time_kst']}")
                
            elif current_kst.day != next_target_time_kst.day and \
                 current_kst.hour > TARGET_HOUR_KST + 1:
                # Catch-up logic for missed target time (e.g., right after server restart)
                next_target_time_kst = calculate_next_target_time(current_kst)
                status['next_scheduled_time_kst'] = next_target_time_kst.strftime("%Y-%m-%d %H:%M:%S KST")

        except Exception as e:
            # If any unhandled exception occurs in the main loop logic, log it and continue to the next iteration
            logger.error(f"⚠️ Major exception in main monitor loop. Continuing after 60s: {e}", exc_info=True)


async def self_ping_loop():
    """
    [Internal Sleep Prevention] Loop that internally pings its own Health Check endpoint every 5 minutes.
    """
    global status
    # Request to its own IP/Port inside Render
    ping_url = f"http://127.0.0.1:{SERVER_PORT}/" 
    logger.info(f"🛡️ Starting internal self-ping loop. Requesting {ping_url} every 5 minutes.")
    
    async with aiohttp.ClientSession() as session:
        while True:
            await asyncio.sleep(5 * 60) # Wait 5 minutes
            
            try:
                # HEAD request is lighter than GET.
                async with session.head(ping_url, timeout=10) as response:
                    if response.status == 200:
                        status['last_self_ping_kst'] = datetime.now(KST_TZ).strftime("%Y-%m-%d %H:%M:%S KST")
                        logger.debug(f"✅ Self-ping successful: {status['last_self_ping_kst']}")
                    else:
                        logger.warning(f"❌ Self-ping failed (Status: {response.status})")
                        
            except Exception as e:
                logger.error(f"❌ Exception during self-ping: {e}")


# =========================================================
# --- [5] FastAPI Web Service and Ping Check Setup ---
# =========================================================

app = FastAPI(
    title="VIX Plot Telegram Scheduler",
    description="VIX/S&P 500 Chart Sender running on Render Free Tier.",
    version="1.0.0"
)

# Start background task upon server startup
@app.on_event("startup")
async def startup_event():
    """Starts the scheduler loop and self-ping loop in the background upon server startup."""
    # Main scheduling loop
    asyncio.create_task(main_monitor_loop()) 
    # Auxiliary self-ping loop for sleep prevention
    asyncio.create_task(self_ping_loop())    
    logger.info("🚀 Background scheduling and self-ping loops have started.")

# ---------------------------------------------------------
# New Endpoint: Set Scheduling Time (Completed in this turn)
# ---------------------------------------------------------
@app.post("/set-time")
async def set_schedule_time(
    hour: str = Form(...), 
    minute: str = Form(...) 
):
    """Saves the KST time entered by the user and updates the next scheduled time."""
    global TARGET_HOUR_KST, TARGET_MINUTE_KST
    global status

    try:
        hour_int = int(hour)
        minute_int = int(minute)
    except ValueError:
        raise HTTPException(status_code=400, detail="Hour and minute must be integers.")
        
    # Validation check
    if not (0 <= hour_int <= 23 and 0 <= minute_int <= 59):
        raise HTTPException(status_code=400, detail="Hour must be 0-23 and minute 0-59.")

    # ⭐️ Update global variables
    TARGET_HOUR_KST = hour_int
    TARGET_MINUTE_KST = minute_int
    
    # ⭐️ Recalculate next send time immediately ⭐️
    now_kst = datetime.now(KST_TZ)
    next_target_time_kst = calculate_next_target_time(now_kst)
    status['next_scheduled_time_kst'] = next_target_time_kst.strftime("%Y-%m-%d %H:%M:%S KST")

    logger.info(f"⏰ New send time set to KST {TARGET_HOUR_KST:02d}:{TARGET_MINUTE_KST:02d}. Next run: {status['next_scheduled_time_kst']}")
    
    # Redirect back to the status page
    return RedirectResponse(url="/", status_code=303)


# ---------------------------------------------------------
# Root Endpoint (Status Dashboard)
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home_status(request: Request):
    """Simple status dashboard with an option to change the schedule time."""
    global status
    
    # Check if necessary environment variables are set
    config_warning = ""
    if 'YOUR_BOT_TOKEN_HERE' in TELEGRAM_BOT_TOKEN:
        config_warning += "<li>⚠️ **TELEGRAM_BOT_TOKEN** is using the default placeholder. Sending is disabled.</li>"
    if TELEGRAM_TARGET_CHAT_ID == '-1000000000':
        config_warning += "<li>⚠️ **TELEGRAM_TARGET_CHAT_ID** is using the default placeholder. Sending is disabled.</li>"
    
    # Calculate current KST
    current_kst = datetime.now(KST_TZ).strftime("%Y-%m-%d %H:%M:%S KST")
    
    # Get current scheduled time for the form
    current_hour = TARGET_HOUR_KST
    current_minute = TARGET_MINUTE_KST

    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>VIX 모니터링 서비스 상태</title>
        <style>
            body {{ font-family: 'Arial', sans-serif; background-color: #f4f7f9; color: #333; margin: 0; padding: 20px; }}
            .container {{ max-width: 800px; margin: 0 auto; background: #ffffff; padding: 25px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            h1 {{ color: #0056b3; border-bottom: 3px solid #0056b3; padding-bottom: 10px; margin-top: 0; }}
            h2 {{ color: #007bff; margin-top: 20px; }}
            .status-box {{ background-color: #e9f7ff; border: 1px solid #b3e0ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
            .status-box p {{ margin: 5px 0; font-size: 1.1em; }}
            .status-box strong {{ color: #333; display: inline-block; width: 150px; }}
            .warning {{ background-color: #ffe0e0; border-left: 5px solid #ff4d4d; padding: 10px; margin-top: 15px; border-radius: 5px; }}
            form {{ background-color: #f9f9f9; padding: 20px; border-radius: 8px; border: 1px solid #ddd; }}
            label {{ display: block; margin-bottom: 5px; font-weight: bold; color: #555; }}
            input[type="number"] {{ width: 80px; padding: 8px; margin-right: 10px; border: 1px solid #ccc; border-radius: 4px; }}
            button {{ background-color: #28a745; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; margin-top: 10px; }}
            button:hover {{ background-color: #218838; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>VIX/S&P 500 차트 스케줄러 상태</h1>
            
            <h2>현재 설정 및 상태</h2>
            <div class="status-box">
                <p><strong>현재 KST 시간:</strong> {current_kst}</p>
                <p><strong>다음 전송 시각 (KST):</strong> {status['next_scheduled_time_kst']}</p>
                <p><strong>마지막 전송일:</strong> {status['last_sent_date_kst']}</p>
                <p><strong>마지막 확인 시각:</strong> {status['last_check_time_kst']}</p>
                <p><strong>마지막 자체 핑 시각:</strong> {status['last_self_ping_kst']}</p>
                <p><strong>현재 전송 시간 (KST):</strong> {TARGET_HOUR_KST:02d}:{TARGET_MINUTE_KST:02d}</p>
            </div>

            {f'<div class="warning"><h3>설정 경고</h3><ul>{config_warning}</ul></div>' if config_warning else ''}

            <h2>전송 시각 변경 (KST)</h2>
            <form method="POST" action="/set-time">
                <label for="hour">시 (0-23):</label>
                <input type="number" id="hour" name="hour" min="0" max="23" value="{current_hour}" required>
                
                <label for="minute">분 (0-59):</label>
                <input type="number" id="minute" name="minute" min="0" max="59" value="{current_minute}" required>
                
                <button type="submit">시간 설정 및 재시작</button>
            </form>
            
            <p style="margin-top: 25px; font-size: 0.9em; color: #777;">* 전송 시각 변경 후 다음 전송 시각이 자동으로 재계산됩니다. 재시작 시점에 따라 다음 전송은 오늘 또는 내일로 설정됩니다.</p>
            <p style="font-size: 0.9em; color: #777;">* 서비스는 **월요일**과 **일요일**에는 전송하지 않습니다 (미국 시장 휴장 및 데이터 불충분). 다음 평일로 자동 연기됩니다.</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# The following standard block is necessary to make the service runnable on cloud platforms like Render.
if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 for all interfaces to be reachable by the platform's proxy
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
