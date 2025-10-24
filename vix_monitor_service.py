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

# FastAPI imports
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from starlette.responses import RedirectResponse

# Graph/Data related external libraries
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import numpy as np
import pandas as pd # <-- pandas import 추가

# Matplotlib backend setting (essential for headless server environments)
matplotlib.use('Agg')

# =========================================================
# --- [1] Configuration, Environment Variables, and Global State ---
# =========================================================
# Korean Standard Time (KST) Timezone setup
KST_TZ = ZoneInfo("Asia/Seoul")
# New York Timezone (EST/EDT) for market close validation
NY_TZ = ZoneInfo("America/New_York")
MONITOR_INTERVAL_SECONDS = 60 # Check time every 1 minute

# ⏰ Global State: User-configurable send time (KST)
TARGET_HOUR_KST = int(os.environ.get('TARGET_HOUR_KST', 13))
TARGET_MINUTE_KST = int(os.environ.get('TARGET_MINUTE_KST', 45))

# ⚠️ Load from Environment Variables (Essential for Render) - Retain user-specified hardcoded defaults
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
TELEGRAM_TARGET_CHAT_ID = os.environ.get('TELEGRAM_TARGET_CHAT_ID', '-1000000000')
SERVER_PORT = int(os.environ.get("PORT", 8000))

# Logging setup (INFO level for key operations)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for default tokens
if 'YOUR_BOT_TOKEN_HERE' in TELEGRAM_BOT_TOKEN or TELEGRAM_TARGET_CHAT_ID == '-1000000000':
    logger.warning("⚠️ Warning: TELEGRAM_BOT_TOKEN or CHAT_ID is set to default. Please configure environment variables.")

# 💾 State maintained in server RAM (Will reset on Render restart - No disk usage)
status = {
    "last_sent_date_kst": "1970-01-01", 
    "last_check_time_kst": "N/A",
    "next_scheduled_time_kst": "N/A",
    "last_self_ping_kst": "N/A"
}

# =========================================================
# --- [2] Data Download Function (Consolidated to avoid duplicate calls) ---
# =========================================================
def download_market_data() -> Optional[pd.DataFrame]:
    """
    Downloads full historical data for VIX and S&P 500 using exponential backoff retry.
    (데이터를 한 번만 다운로드하도록 통합했습니다)
    """
    logger.info("📈 Starting market data download...")

    max_retry = 4 # Maximum 4 attempts
    tickers = ["^VIX", "^GSPC"]
    start_date = "2025-04-01" # Data start date
    data_all = None
    
    for attempt in range(1, max_retry + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_retry}: Downloading VIX and S&P 500 data (start={start_date})...")
            
            # Download data
            data_all = yf.download(tickers, start=start_date, progress=False, timeout=20)
            
            # Basic validation
            if data_all.empty or data_all['Close'].empty:
                raise ValueError("Downloaded data is empty.")

            logger.info(f"Attempt {attempt}: Data downloaded successfully.")
            return data_all # Successful download
            
        except Exception as e:
            logger.warning(f"Data download failed (Attempt {attempt}): {e}")
            if attempt < max_retry:
                # ⭐️ Apply Exponential Backoff: Wait 2^1=2s, 2^2=4s, 2^3=8s
                sleep_time = 2 ** attempt
                logger.info(f"Applying Exponential Backoff. Waiting {sleep_time} seconds before next retry...")
                time.sleep(sleep_time)
            else:
                logger.error("Max retries exceeded. Failed to acquire data.")
                return None
    
    return None

# =========================================================
# --- [3] VIX Plotter Function (Chart Generation Logic) ---
# =========================================================
def plot_vix_sp500(data_all: pd.DataFrame, width=6.4, height=4.8) -> Optional[io.BytesIO]:
    """
    Generates a chart comparing the closing price trends of VIX and S&P 500
    from the provided DataFrame.
    """
    logger.info("🎨 Starting chart generation from downloaded data...")

    try:
        # Extract and align Close data from the provided DataFrame
        vix_df = data_all['Close']['^VIX'].dropna()
        gspc_df = data_all['Close']['^GSPC'].dropna()
        
        common_dates = vix_df.index.intersection(gspc_df.index)
        vix = vix_df.loc[common_dates]
        qqq = gspc_df.loc[common_dates] # Renamed from gspc to qqq in old code, keeping qqq for plot variable names

        if vix.empty or qqq.empty:
            raise ValueError("Aligned data is empty.")
            
        # Apply finalized chart design logic
        plt.style.use('dark_background')
        
        fig, ax1 = plt.subplots(figsize=(width, height)) 
        ax2 = ax1.twinx()
        
        # Set background color
        fig.patch.set_facecolor('#222222')
        ax1.set_facecolor('#2E2E2E')
        ax2.set_facecolor('#2E2E2E')
        
        # Data and colors
        last_vix_price = vix.iloc[-1].item()
        last_qqq_price = qqq.iloc[-1].item()
        title_text = f"VIX ({last_vix_price:.2f}) vs S&P 500 ({last_qqq_price:.2f})"
        vix_color = '#FF6B6B' # VIX color (Reddish)
        qqq_color = '#6BCBFF' # S&P 500 color (Blueish)
        new_fontsize = 8 * 1.3
        
        # Plotting
        ax2.plot(common_dates, vix.values, color=vix_color, linewidth=1.5)
        ax1.plot(common_dates, qqq.values, color=qqq_color, linewidth=1.5)
        
        # X-axis date format and interval
        formatter = mdates.DateFormatter('%Y-%m-%d') 
        ax1.xaxis.set_major_formatter(formatter)
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4)) # Keep 4-month interval
        fig.autofmt_xdate(rotation=45)

        # Y-axis label setting
        ax1.set_ylabel('S&P 500 Index', color=qqq_color, fontsize=12, fontweight='bold', labelpad=5)
        ax2.set_ylabel('VIX', color=vix_color, fontsize=12, fontweight='bold', labelpad=5)
        
        # Add VIX level annotations and horizontal lines
        try:
            # Find the date position for annotations (90% through the data period)
            new_text_x_pos = common_dates[int(len(common_dates)*0.9)]
        except:
            # Safety net for very small data sets
            new_text_x_pos = common_dates[-1] + timedelta(days=1)
        
        # VIX annotations
        ax2.text(new_text_x_pos, 15.5, "VIX 15 (Greed/Sell)", color='yellow', fontsize=new_fontsize, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')
        ax2.text(new_text_x_pos, 30.5, "VIX 30 (Warning)", color='peru', fontsize=new_fontsize, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')
        ax2.text(new_text_x_pos, 40.5, "VIX 40 (Fear/Buy)", color='lightGreen', fontsize=new_fontsize, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')
        
        # VIX horizontal lines
        ax2.axhline(y=15, color='yellow', linestyle='--', linewidth=1.2, alpha=0.8)
        ax2.axhline(y=30, color='peru', linestyle='--', linewidth=1.0, alpha=0.8)
        ax2.axhline(y=40, color='lightGreen', linestyle='--', linewidth=1.2, alpha=0.8)
        
        # Title and minimal margins
        fig.suptitle(title_text, color='white', fontsize=12, fontweight='bold', y=0.98) 
        fig.tight_layout(rect=[0.025, 0.025, 1, 1]) 
        
        # ⭐️ Save PNG image to memory buffer (crucial: no disk usage) ⭐️
        plot_data = io.BytesIO()
        plt.savefig(plot_data, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1) 
        plot_data.seek(0)
        
        plt.close(fig) # **VERY IMPORTANT: Prevent memory leaks**
        logger.info("✅ Chart generation complete (saved to memory).")
        return plot_data

    except Exception as e:
        logger.error(f"❌ Exception during chart generation: {e}", exc_info=True)
        return None

# =========================================================
# --- [4] Telegram Sending Function (HTTP API) ---
# =========================================================
async def send_photo_via_http(chat_id: str, photo_bytes: io.BytesIO, caption: str) -> bool:
    """Sends the chart image to the Telegram bot."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    data = aiohttp.FormData()
    data.add_field('chat_id', chat_id)
    data.add_field('caption', caption)
    data.add_field('parse_mode', 'Markdown')
    # ⭐️ Pass the io.BytesIO object directly to the 'photo' field ⭐️
    data.add_field('photo', 
                   photo_bytes, 
                   filename='vix_gspc_chart.png', 
                   content_type='image/png')

    # Add retry logic (for network resilience)
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
    """
    Executes the complete process: data download (once), date validation, chart generation,
    and transmission. Guarantees memory buffer cleanup (plot_buffer.close()).
    """
    global status
    
    if 'YOUR_BOT_TOKEN_HERE' in TELEGRAM_BOT_TOKEN or TELEGRAM_TARGET_CHAT_ID == '-1000000000':
        logger.error("Telegram token or Chat ID is set to default. Skipping send.")
        return False
        
    # 1. Download Data (Heavy lifting only done ONCE)
    data_all = download_market_data()
    if data_all is None:
        logger.error("Data download failed. Skipping job.")
        return False
        
    # --- [2. Validation Logic: Check if data is fresh for today] ---
    try:
        # Extract and align data for validation purposes
        vix_data = data_all['Close']['^VIX'].dropna()
        gspc_data = data_all['Close']['^GSPC'].dropna()
        
        if vix_data.empty or gspc_data.empty:
            raise ValueError("Downloaded data is incomplete or empty after cleanup.")
            
        latest_vix = vix_data.iloc[-1].item()
        latest_gspc = gspc_data.iloc[-1].item()
        
        # The date of the latest closing price available (last index of the combined data)
        latest_data_date_str = max(vix_data.index[-1], gspc_data.index[-1]).strftime('%Y-%m-%d')
    except Exception as e:
        logger.error(f"❌ Failed to process latest VIX/S&P 500 data for validation: {e}")
        return False
        
    # Get the current date in New York (EST/EDT)
    ny_now = datetime.now(NY_TZ)
    ny_current_date_str = ny_now.strftime('%Y-%m-%d')
    
    logger.info(f"Date Check: Latest Data Date (US Close) = {latest_data_date_str}, Current NY Date = {ny_current_date_str}")

    # Check: If the latest data date is NOT the current New York date, skip.
    if latest_data_date_str != ny_current_date_str:
        logger.warning(
            f"⚠️ Data is not fresh for NY today. Skipping send. "
            f"Latest available market close date: {latest_data_date_str} (Must match {ny_current_date_str})."
        )
        return True # Skip send, but job considered complete for this cycle
    
    logger.info(f"✅ Data freshness confirmed. Proceeding with chart generation and Telegram send.")
    # --- [End of Validation Logic] ---
    
    # 3. Chart Generation & Send (with guaranteed memory cleanup)
    plot_buffer = None
    success = False
    try:
        plot_buffer = plot_vix_sp500(data_all) # Pass the downloaded data
        if not plot_buffer:
            logger.error("Chart generation failed. Skipping send.")
            return False # Failed job
            
        # 4. Telegram Send
        caption = (
            f"\n🗓️ {latest_data_date_str} (US Market Close)\n"
            f"📉 VIX (Volatility): **{latest_vix:.2f}**\n"
            f"📈 S&P 500 (Index): **{latest_gspc:.0f}**\n\n"
            f"VIX and the S&P 500 typically move in opposite directions.\n"
        )

        success = await send_photo_via_http(TELEGRAM_TARGET_CHAT_ID, plot_buffer, caption)

        if success:
            current_kst = datetime.now(KST_TZ)
            status['last_sent_date_kst'] = current_kst.strftime("%Y-%m-%d")
            logger.info(f"Successfully sent. Last sent date updated: {status['last_sent_date_kst']}")
        
        return success
        
    finally:
        # 5. Memory Cleanup (GUARANTEED cleanup, addressing user concern 2)
        if plot_buffer:
            plot_buffer.close() 
            logger.info("🗑️ Chart memory buffer closed successfully.")


# =========================================================
# --- [5] Scheduling and Loop Logic ---
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
        # If today's target time has not yet passed, set it for today
        next_target = target_time_today
        
    return next_target

async def main_monitor_loop():
    """Runs every minute, checking the send time and triggering the job."""
    global status
    
    # Initial next send time setup
    now_kst = datetime.now(KST_TZ)
    next_target_time_kst = calculate_next_target_time(now_kst)
    status['next_scheduled_time_kst'] = next_target_time_kst.strftime("%Y-%m-%d %H:%M:%S KST")
    
    logger.info(f"🔍 Monitoring started. Next scheduled time (KST): {status['next_scheduled_time_kst']}")
    
    while True:
        await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
        
        current_kst = datetime.now(KST_TZ)
        status['last_check_time_kst'] = current_kst.strftime("%Y-%m-%d %H:%M:%S KST")
        
        # Logging the schedule check every minute at WARNING level as requested
        logger.warning(f"Monitor: Checking schedule (KST: {current_kst.strftime('%H:%M:%S')}).")
        
        # Check send condition (once per day, at the specified time)
        target_date_kst = next_target_time_kst.strftime("%Y-%m-%d")

        if current_kst >= next_target_time_kst and \
           current_kst < next_target_time_kst + timedelta(minutes=1) and \
           target_date_kst != status['last_sent_date_kst']:

            logger.info(f"⏰ Send time reached (KST: {current_kst.strftime('%H:%M:%S')}). Executing job.")
            
            # Execute send logic (This is where the new date check happens inside)
            await run_and_send_plot()
            
            # Update next target time regardless of whether data was fresh or not
            # If data wasn't fresh, the job will be run again tomorrow.
            if status['last_sent_date_kst'] == target_date_kst:
                # Only if actual sending happened (last_sent_date_kst was updated), move to next day
                next_target_time_kst = calculate_next_target_time(current_kst)
                status['next_scheduled_time_kst'] = next_target_time_kst.strftime("%Y-%m-%d %H:%M:%S KST")
                logger.info(f"➡️ Next scheduled time (KST): {status['next_scheduled_time_kst']}")
            else:
                 # If we ran but didn't send (e.g., data was not fresh), we wait for the time boundary to pass naturally.
                 # The next check will try again tomorrow at the same time.
                 pass

        elif current_kst.day != next_target_time_kst.day and \
             current_kst.hour > TARGET_HOUR_KST + 1:
            # If the target date has passed but hasn't been updated (e.g., right after server restart)
            next_target_time_kst = calculate_next_target_time(current_kst)
            status['next_scheduled_time_kst'] = next_target_time_kst.strftime("%Y-%m-%d %H:%M:%S KST")

async def self_ping_loop():
    """
    [Internal Sleep Prevention] A loop that internally pings its own Health Check endpoint every 5 minutes.
    """
    global status
    # Request to its own IP/Port within Render environment
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
# --- [6] FastAPI Web Service and Ping Check Setup ---
# =========================================================

app = FastAPI(
    title="VIX Plot Telegram Scheduler",
    description="VIX/S&P 500 Chart Sender running on Render Free Tier.",
    version="1.0.0"
)

# Start background tasks on server startup
@app.on_event("startup")
async def startup_event():
    """Starts the scheduler loop and self-ping loop in the background upon server start."""
    # Main scheduling loop
    asyncio.create_task(main_monitor_loop()) 
    # Self-ping loop for sleep prevention
    asyncio.create_task(self_ping_loop())   
    logger.info("🚀 Background scheduling and self-ping loops have started.")

# ---------------------------------------------------------
# New Endpoint: Set Scheduling Time
# ---------------------------------------------------------
@app.post("/set-time")
async def set_schedule_time(
    hour: str = Form(...), 
    minute: str = Form(...) 
):
    """Saves the user-input KST time and updates the next scheduled send time."""
    global TARGET_HOUR_KST, TARGET_MINUTE_KST
    global status

    try:
        hour_int = int(hour)
        minute_int = int(minute)
    except ValueError:
        raise HTTPException(status_code=400, detail="Hour and minute must be integers.")
        
    # Validation check
    if not (0 <= hour_int <= 23 and 0 <= minute_int <= 59):
        raise HTTPException(status_code=400, detail="Invalid hour (0-23) or minute (0-59).")
        
    # Update global variables
    TARGET_HOUR_KST = hour_int
    TARGET_MINUTE_KST = minute_int
    
    # Recalculate next target time immediately to reflect changes
    now_kst = datetime.now(KST_TZ)
    next_target_time_kst = calculate_next_target_time(now_kst)
    status['next_scheduled_time_kst'] = next_target_time_kst.strftime("%Y-%m-%d %H:%M:%S KST")

    logger.info(f"⏰ Schedule time changed to: {TARGET_HOUR_KST:02d}:{TARGET_MINUTE_KST:02d} KST. Next send time updated: {status['next_scheduled_time_kst']}") 
    
    # Redirect to the status page (303 See Other)
    return RedirectResponse(url="/", status_code=303)

# ---------------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------------
@app.get("/")
@app.head("/")
async def health_check(request: Request): # 👈 Accepts Request object as argument
    """Health Check endpoint to prevent Render Free Tier Spin Down."""
    global TARGET_HOUR_KST, TARGET_MINUTE_KST
    current_kst = datetime.now(KST_TZ)
    
    # For HEAD requests, return a simple response to minimize load
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
# --- [7] Execution (Render uses Procfile, this is for local testing) ---
# =========================================================
if __name__ == '__main__':
    # This part is for local testing. In Render, use the command: uvicorn vix_monitor_service:app
    import uvicorn
    logger.info(f"Starting uvicorn server on port {SERVER_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
