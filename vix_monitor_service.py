import os
import sys
import asyncio
import aiohttp
import io
import logging
import time
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
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
import pandas as pd

class FearGreedFetcher:
    """
    CNN + Upbit 공포/탐욕 지수 및 P/C 비율 통합 클래스
    개별 값 단위로 오류 발생 시 0으로 처리
    """
    CNN_BASE_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
    UPBIT_FG_API = "https://datalab-api.upbit.com/api/v1/indicator/overview"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    ERROR_VALUE = 0  # 숫자 오류 시 0 반환

    def __init__(self):
        self.공탐레이팅: float = self.ERROR_VALUE
        self.공탐: float = self.ERROR_VALUE
        self.풋엔콜레이팅: float = self.ERROR_VALUE
        self.풋엔콜값: float = self.ERROR_VALUE
        self.코인레이팅: float = self.ERROR_VALUE
        self.코인: float = self.ERROR_VALUE

    def fetch_all(self) -> tuple[float, float, float, float, float, float]:
        """CNN + Upbit 데이터 모두 조회, 개별 오류 시 0 반환"""
        self._fetch_cnn_data()
        self._fetch_upbit_data()
        return (self.공탐레이팅, self.공탐, self.풋엔콜레이팅, self.풋엔콜값, self.코인레이팅, self.코인)

    def _fetch_cnn_data(self):
        today = datetime.now().date()
        dates_to_try = [today.strftime("%Y-%m-%d"), (today - timedelta(days=1)).strftime("%Y-%m-%d")]

        data = None
        for date_str in dates_to_try:
            try:
                r = requests.get(self.CNN_BASE_URL + date_str, headers=self.HEADERS, timeout=10)
                r.raise_for_status()
                data = r.json()
                break
            except:
                continue

        # CNN 데이터가 아예 없으면 모두 0
        if not data:
            self.공탐레이팅 = self.공탐 = self.풋엔콜레이팅 = self.풋엔콜값 = 0
            return

        # Fear & Greed
        fg_data = data.get("fear_and_greed", {})
        self.공탐레이팅 = fg_data.get("rating", 0) or 0
        self.공탐 = fg_data.get("score", 0) or 0

        # Put/Call
        put_call_data = data.get("put_call_options", {})
        self.풋엔콜레이팅 = put_call_data.get("rating", 0) or 0
        pc_list = put_call_data.get("data", [])
        self.풋엔콜값 = pc_list[-1].get("y", 0) if pc_list else 0

    def _fetch_upbit_data(self):
        try:
            r = requests.get(self.UPBIT_FG_API, headers=self.HEADERS, timeout=10)
            r.raise_for_status()
            data = r.json()
        except:
            self.코인레이팅 = self.코인 = 0
            return

        coin_fg = None
        for indicator in data.get("data", {}).get("indicators", []):
            if indicator.get("info", {}).get("category") == "fear":
                coin_fg = indicator
                break

        if not coin_fg:
            self.코인레이팅 = self.코인 = 0
            return

        self.코인레이팅 = coin_fg.get("chart", {}).get("gauge", {}).get("name", 0) or 0
        self.코인 = coin_fg.get("price", {}).get("tradePrice", 0) or 0
fetcher = FearGreedFetcher()


class TetherMonitor:
    def __init__(self):
        self.GOOD_THRESHOLD_KIMF = 1.5
        self.NORMAL_THRESHOLD_KIMF = 3.0
        self.dxy_hist = pd.Series(dtype=float)
        self.usdkrw = None
        self.usdtkrw = None

    def fetch_data(self):
        errors = []

        # USD/KRW 환율 (다음 금융)
        try:
            url = "https://finance.daum.net/api/exchanges/FRX.KRWUSD"
            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.daum.net/exchanges"}
            resp = requests.get(url, headers=headers, timeout=5).json()
            self.usdkrw = float(resp.get('basePrice', 0))
            if self.usdkrw == 0:
                errors.append("USD/KRW 파싱 오류")
        except:
            self.usdkrw = None
            errors.append("USD/KRW 파싱 오류")

        # USDT/KRW (업비트)
        try:
            url_upbit_USDT = "https://api.upbit.com/v1/ticker?markets=KRW-USDT"
            resp = requests.get(url_upbit_USDT, timeout=5).json()
            self.usdtkrw = float(resp[0]['trade_price'])
            if self.usdtkrw == 0:
                errors.append("테더 파싱 오류")
        except:
            self.usdtkrw = None
            errors.append("테더 파싱 오류")

        # DXY (달러 인덱스) – 재시도 3회
        attempt = 0
        max_attempts = 3
        self.dxy_hist = pd.Series(dtype=float)

        while attempt < max_attempts:
            try:
                ticker = yf.Ticker("DX-Y.NYB")
                self.dxy_hist = ticker.history(period="7d", interval="1h")['Close']
                if not self.dxy_hist.empty:
                    break  # 성공하면 루프 종료
                else:
                    errors.append("DXY 파싱 오류 (빈 데이터)")
            except Exception as e:
                errors.append(f"DXY 파싱 오류: {e}")

            attempt += 1
            if attempt < max_attempts:
                time.sleep(3)  # 3초 인터벌 후 재시도

        if self.dxy_hist.empty:
            errors.append("DXY 데이터 가져오기 실패 (3회)")

        return errors

    def evaluate(self):
        errors = self.fetch_data()
        if errors:
            return "데이터 오류"

        kimchi_premium = ((self.usdtkrw / self.usdkrw) - 1) * 100

        usdkrw_series = pd.Series([self.usdkrw] * len(self.dxy_hist), index=self.dxy_hist.index)
        dxy_norm = (self.dxy_hist - self.dxy_hist.min()) / (self.dxy_hist.max() - self.dxy_hist.min())
        usdkrw_norm = (usdkrw_series - usdkrw_series.min()) / (usdkrw_series.max() - usdkrw_series.min())

        relative_diff = usdkrw_norm.iloc[-1] - dxy_norm.iloc[-1]

        if relative_diff < 0:
            return "원화 강세"
        else:
            return "원화 약세"
Tmonitor = TetherMonitor()



def get_usdt_and_exchange_rate(refresh_count=0):
    테더원 = 1
    달러원 = 0
    달러테더괴리율 = 0

    # 환율 갱신 (refresh_count 기준)
    if refresh_count % 20 == 0 or 달러원 == 0:
        # 달러-원 환율 (Daum 금융)
        try:
            url = "https://finance.daum.net/api/exchanges/FRX.KRWUSD"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
                "Referer": "https://finance.daum.net/exchanges"
            }
            resp = requests.get(url, headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            base_price = data.get("basePrice")
            if base_price is not None:
                달러원 = float(base_price)
        except:
            달러원 = 0

        # 업비트 USDT 가격
        try:
            url_upbit_USDT = "https://api.upbit.com/v1/ticker?markets=KRW-USDT"
            resp = requests.get(url_upbit_USDT, timeout=5).json()
            테더원 = float(resp[0]['trade_price'])
            time.sleep(1)
        except:
            테더원 = 1

    # 달러-테더 괴리율 계산
    try:
        달러테더괴리율 = round((테더원 / 달러원 - 1) * 100, 2)
    except ZeroDivisionError:
        달러테더괴리율 = 0

    return 테더원, 달러원, 달러테더괴리율



class GoldKimpAnalyzer:
    API_URL = "https://goldkimp.com/wp-json/ck/v1/kpri"
    OUNCE_TO_GRAM = 31.1034768

    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url

    def _fetch_data(self):
        try:
            resp = requests.get(self.api_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("rows"):
                return None
            return data
        except Exception:
            return None

    def _calculate_metrics(self, data):
        try:
            df = pd.DataFrame(data.get("rows", []))
            df['time'] = pd.to_datetime(df['time'], format='%y/%m/%d %H:%M')
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)

            df['xauusd_oz'] = pd.to_numeric(df['xauusd_oz'], errors='coerce')
            df['usdkrw'] = pd.to_numeric(df['usdkrw'], errors='coerce')
            df['krxkrw_g'] = pd.to_numeric(df['krxkrw_g'], errors='coerce')
            df.dropna(subset=['xauusd_oz', 'usdkrw', 'krxkrw_g'], inplace=True)
            if df.empty:
                return None

            df['xau_krw_g'] = (df['xauusd_oz'] * df['usdkrw']) / self.OUNCE_TO_GRAM
            df['premium_rate'] = ((df['krxkrw_g'] - df['xau_krw_g']) / df['xau_krw_g']) * 100

            latest = df.iloc[-1]
            return (
                float(latest['krxkrw_g']),          
                float(latest['xau_krw_g']),         
                round(float(latest['premium_rate']), 4)  
            )
        except Exception:
            return None

    # 🔹 메인 루프용 안전한 호출 메서드
    def get_core_metrics(self):
        data = self._fetch_data()
        metrics = self._calculate_metrics(data) if data else None
        if metrics is None:
            return 0.0, 0.0, 0.0  # 오류 발생 시 0으로 반환
        return metrics
Goldresult = GoldKimpAnalyzer().get_core_metrics()



# Set Matplotlib backend (required for headless server environment)
matplotlib.use('Agg')

# =========================================================
# --- [1] Configuration, Environment Variables, and Global State ---
# =========================================================
# Set Korean Standard Time (KST) timezone
KST_TZ = ZoneInfo("Asia/Seoul")
MONITOR_INTERVAL_SECONDS = 60 # Check time every 1 minute

# ⏰ Global State: User-configurable send time (KST)
TARGET_HOUR_KST = int(os.environ.get('TARGET_HOUR_KST', 6))
TARGET_MINUTE_KST = int(os.environ.get('TARGET_MINUTE_KST', 0))

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
def _sync_fetch_and_plot_data(width=6.4, height=4.8) -> Optional[Tuple[io.BytesIO, float, float, str]]:
    """
    INTERNAL: Synchronously fetches data using yfinance and generates the chart
    using Matplotlib. 이 함수는 별도의 스레드에서 실행되도록 설계되었습니다.
    
    데이터 다운로드 실패 시 예외(ValueError 등)를 발생시킵니다.
    """
    tickers = ["^VIX", "^GSPC"]
    vix, qqq = None, None
    
    # Using a fixed start date for consistent comparison
    start_date = "2025-03-01" 
    
    logger.info("Executing synchronous data download and chart generation...")
            
    # --- Data Download (Synchronous I/O) ---
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
    
    logger.info(f"Data downloaded successfully (VIX={latest_vix:.2f}, S&P500={latest_gspc:.0f}).")


    # --- Chart Generation (CPU-bound) ---
    try:
        plt.style.use('dark_background')
        
        fig, ax1 = plt.subplots(figsize=(width, height)) 
        ax2 = ax1.twinx()
        
        # Set background color
        fig.patch.set_facecolor('#222222')
        ax1.set_facecolor('#2E2E2E')
        ax2.set_facecolor('#2E2E2E')
        
        # Data and colors
        title_text = f"S&P 500 ({latest_gspc:.2f}) vs VIX ({latest_vix:.2f})"
        vix_color = '#FF6B6B' # VIX color (Red tone)
        qqq_color = '#6BCBFF' # S&P 500 color (Blue tone)
        new_fontsize = 8 * 1.3
        
        # Plotting
        ax2.plot(common_dates, vix.values, color=vix_color, linewidth=1.5)
        # S&P 500 (GSPC)
        ax1.plot(common_dates, qqq.values, color=qqq_color, linewidth=1.5)
        
        # X-axis date format and interval setting
        formatter = mdates.DateFormatter('%y-%m-%d') 
        ax1.xaxis.set_major_formatter(formatter)
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3)) 
        fig.autofmt_xdate(rotation=45)

        # Y-axis label setting
        ax1.set_ylabel('S&P 500 Index', color=qqq_color, fontsize=12, fontweight='bold', labelpad=5)
        ax2.set_ylabel('VIX (Volatility)', color=vix_color, fontsize=12, fontweight='bold', labelpad=5)
        
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
        ax2.text(new_text_x_pos, 40.5, "VIX 40 (Fear/Buy)", color='yellow', fontsize=new_fontsize, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')
        
        # VIX horizontal lines
        ax2.axhline(y=15, color='yellow', linestyle='--', linewidth=1.2, alpha=0.8)
        ax2.axhline(y=30, color='peru', linestyle='--', linewidth=1.0, alpha=0.8)
        ax2.axhline(y=40, color='yellow', linestyle='--', linewidth=1.2, alpha=0.8)
        
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
        # If plotting fails, return None
        return None


async def plot_vix_sp500(width=6.4, height=4.8) -> Optional[Tuple[io.BytesIO, float, float, str]]:
    """
    [ASYNC WRAPPER] Generates a comparative chart of VIX and S&P 500 closing prices,
    and returns the chart buffer along with the latest data.
    
    This function handles the retry logic asynchronously and enforces a strict timeout 
    for the synchronous execution thread.
    """
    logger.info("📈 Starting async data download and chart generation...")

    max_retry = 4 
    # Max time allowed for the plot function (well below the typical 60s gateway timeout)
    PLOT_TIMEOUT_SECONDS = 50 
    
    for attempt in range(1, max_retry + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_retry}: Executing data fetch and plot in background thread with a {PLOT_TIMEOUT_SECONDS}s timeout...")
            
            # ⭐️ Enforce a strict timeout on the background thread execution ⭐️
            plot_result = await asyncio.wait_for(
                asyncio.to_thread(_sync_fetch_and_plot_data, width, height),
                timeout=PLOT_TIMEOUT_SECONDS
            )
            
            if plot_result:
                return plot_result
            else:
                # _sync_fetch_and_plot_data returned None (plotting failed)
                raise Exception("Synchronous plot generation failed.")
            
        except asyncio.TimeoutError:
            # Handle the specific case where the background thread took too long
            logger.error(f"❌ Data download/plot exceeded the {PLOT_TIMEOUT_SECONDS}s timeout (Attempt {attempt}).")
            if attempt == max_retry:
                logger.error("Max retries exceeded due to timeout. Failed to acquire data.")
                return None
            # Continue to exponential backoff and retry
            
        except Exception as e:
            # Handle I/O (e.g., yfinance download failure) or plotting exceptions from the background thread
            logger.warning(f"Data download/plot failed (Attempt {attempt}): {e}")
            if attempt < max_retry:
                # Apply Exponential Backoff using non-blocking sleep
                sleep_time = 5 ** attempt
                logger.info(f"Applying Exponential Backoff. Waiting {sleep_time} seconds before next retry...")
                await asyncio.sleep(sleep_time) # NON-BLOCKING SLEEP
            else:
                logger.error("Max retries exceeded. Failed to acquire data.")
                return None
    
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
        
    # ⭐️ MUST AWAIT the call since plot_vix_sp500 is now an async function ⭐️
    plot_result = await plot_vix_sp500()
    
    if not plot_result:
        logger.error("Chart generation failed. Skipping send and recalculating next target time.")
        return False
    
    plot_buffer, latest_vix, latest_gspc, latest_date_utc = plot_result
    
    # Latest data is already fetched inside plot_vix_sp500
    공탐레이팅, 공탐, 풋엔콜레이팅, 풋엔콜값, 코인레이팅, 코인 = fetcher.fetch_all()
    테더원, 달러원, 달러테더괴리율 = get_usdt_and_exchange_rate(refresh_count=0)
    한국시세, 국제시세, 괴리율 = Goldresult

    caption = (
            f"\n🗓️ {latest_date_utc} (US Market Close)\n"
            f"📈 VIX (Volatility): **{latest_vix:.2f}**\n"   
            f"📉 S&P 500 (Index): **{latest_gspc:.2f}**\n"
            f"🙏 S&P 500 (Fear/Greed): {공탐레이팅}\n"                     
            
            # f"공탐: {공탐}\n"
            # f"💹 풋/콜: {풋엔콜레이팅}\n"
            # f"풋/콜 값: {풋엔콜값}\n"
            f"🪙 업비트 (공포/탐욕): {코인레이팅}\n\n"
            # f"코인: {코인}\n"
            
            f"💵 달러-원: {달러원:,.0f} 원\n"
            f"💸 테더-원: {테더원:,.0f} 원\n"            
            f"🏦 USDT/USD 괴리율: {달러테더괴리율:.2f} %\n\n"
            # f"🏦 달러 인덱스 대비 원화 평가: {달러대비원화}\n\n"
            
            f"🇰🇷 한국 금 시세: {한국시세:,.0f} 원\n"
            f"🇬🇧 국제 금 시세: {국제시세:,.0f} 원\n"
            f"⚖️ KRX 금 시장 김프: {괴리율:.2f} %"
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
                # Use HEAD request as it is the lightest check
                async with session.head(ping_url, timeout=10) as response:
                    # A 200 OK status indicates the server is alive and responded to HEAD
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
# New Endpoint: Set Scheduling Time
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
# Root Endpoint (Status Dashboard) - Now allows GET and HEAD
# ---------------------------------------------------------
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def home_status(request: Request):
    """Simple status dashboard with an option to change the schedule time.
    Allows both GET (for browser) and HEAD (for health check/ping)."""
    global status
    
    # If the request is a HEAD request, just return a 200 OK without content.
    if request.method == "HEAD":
        return HTMLResponse(status_code=200)

    # For GET requests, return the full status page.
    
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
                <p><strong>마지막 자체 핑:</strong> {status['last_self_ping_kst']}</p>
                <p><strong>설정된 전송 시간 (KST):</strong> {current_hour:02d}:{current_minute:02d}</p>
            </div>

            {f'<div class="warning"><h3>설정 경고</h3><ul>{config_warning}</ul></div>' if config_warning else ''}
            
            <h2>전송 시간 변경 (KST)</h2>
            <form method="POST" action="/set-time">
                <label for="hour">시 (0-23):</label>
                <input type="number" id="hour" name="hour" value="{current_hour}" min="0" max="23" required>
                
                <label for="minute">분 (0-59):</label>
                <input type="number" id="minute" name="minute" value="{current_minute}" min="0" max="59" required>
                
                <button type="submit">전송 시간 업데이트</button>
            </form>
            
            <p style="margin-top: 20px; font-size: 0.9em; color: #666;">
                *이 서비스는 매일 한 번, 설정된 KST 시간에 맞춰 텔레그램으로 VIX 및 S&P 500 차트를 전송합니다.
            </p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    # If running locally (not via uvicorn/gunicorn, which Render typically uses)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
