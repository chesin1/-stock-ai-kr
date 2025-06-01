import yfinance as yf
import pandas as pd
import ta
import time
import os
from datetime import datetime, timedelta
from pandas_datareader import data as web
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
import difflib
import xml.etree.ElementTree as ET
import requests
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.font_manager as fm
import platform




# ------------------------
# 설정
# ------------------------
# Pandas 출력 설정 변경 (모든 컬럼 표시)
pd.set_option('display.max_columns', None)
start_date = "2019-01-01"
end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
API_KEY = 'VADVWXGUAJ1D7O7H9PKX'  # API 키를 발급받아 입력하세요.

# ✅ 한국 주식 티커 설정
kr_tickers = {
    '삼성전자': '005930.KS',
    'SK하이닉스': '000660.KS',
    'LG에너지솔루션': '373220.KS',
    '삼성바이오로직스': '207940.KS',
    '현대차': '005380.KS',
    '카카오': '035720.KS',
    '기아': '000270.KS',
    'LG화학': '051910.KS',
    'NAVER': '035420.KS',
    'POSCO홀딩스': '005490.KS'
}
ticker_to_name = {v: k for k, v in kr_tickers.items()}
start_date = "2019-01-01"
end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

OUTPUT_DIR = "data"
MERGED_FILE = os.path.join(OUTPUT_DIR, "kr_stock_macro_merged.csv")
PREDICTED_FILE = os.path.join(OUTPUT_DIR, "kr_predicted_with_scores.csv")
SIMULATION_FILE_SIMPLE_FORMATTED = os.path.join(OUTPUT_DIR, "kr_simulation_result_simple.csv")

FEATURE_COLUMNS = [
    # 🔹 OHLC 가격 정보
    'Open', 'High', 'Low', 'Close',
    
    # 🔹 기술적 지표
    '변화율(%)', '5일이평', '20일이평', '60일이평', 'RSI',
    'MACD', 'MACD_Signal', 'MACD_Histogram',
    'Bollinger_Bandwidth', 'Momentum_10', 'Cumulative_Return',
    
    # 🔹 거시 지표 (ECOS 기반)
    '한국은행 기준금리', '콜금리(익일물)', '예금은행 수신금리', '예금은행 대출금리',
    '가계신용', 'M2(광의통화, 평잔)', 
    '실업률', '소비자물가지수', '농산물 및 석유류제외 소비자물가지수'
]
# ✅ 통계표 및 아이템 코드 구조 (ECOS)
stat_codes = {
    '722Y001': {'name': '한국은행 기준금리', 'item_codes': [{'code': '0101000', 'cycle': 'Q'}]},
    '817Y002': {'name': '콜금리(익일물)', 'item_codes': [{'code': '010101000', 'cycle': 'D'}]},
    '121Y002': {'name': '예금은행 수신금리', 'item_codes': [{'code': 'BEABAA2', 'cycle': 'M'}]},
    '121Y006': {'name': '예금은행 대출금리', 'item_codes': [{'code': 'BECBLA01', 'cycle': 'M'}]},
    '151Y001': {'name': '가계신용', 'item_codes': [{'code': '1000000', 'cycle': 'Q'}]},
    '101Y003': {'name': 'M2(광의통화, 평잔)', 'item_codes': [{'code': 'BBHS00', 'cycle': 'M'}]},
    '901Y027': {'name': '실업률', 'item_codes': [{'code': 'I61BC', 'cycle': 'M'}]},
    '901Y009': {'name': '소비자물가지수', 'item_codes': [{'code': '0', 'cycle': 'M'}]},
    '901Y010': {'name': '농산물 및 석유류제외 소비자물가지수', 'item_codes': [{'code': '00', 'cycle': 'M'}]}
}

# ------------------------
# 1단계: 주가 + 거시지표 수집
# ------------------------
# ✅ ECOS 데이터 유효성 검사
def get_item_list(stat_code):
    url = f"https://ecos.bok.or.kr/api/StatisticItemList/{API_KEY}/xml/kr/1/1000/{stat_code}/"
    res = requests.get(url)
    root = ET.fromstring(res.content)

    item_list = []
    for row in root.findall(".//row"):
        item_code = row.findtext("ITEM_CODE")
        item_name = row.findtext("ITEM_NAME")
        cycle = row.findtext("CYCLE")
        if item_code and item_name and cycle:
            item_list.append(item_code)

    return item_list

def dmqa_check(stat_code, item_code):
    # ✅ 실제 유효한 ITEM_CODE 목록 가져오기
    valid_codes = get_item_list(stat_code)
    if item_code not in valid_codes:
        # ✅ 유사한 코드 찾기
        suggestions = difflib.get_close_matches(item_code, valid_codes, n=3)
        print(f"\n⚠️ 잘못된 ITEM_CODE: '{item_code}'")
        if suggestions:
            print(f"🔄 수정 제안: {', '.join(suggestions)}")
        else:
            print("🚫 유사한 코드가 없습니다.")
        return False
    return True

# ✅ ECOS 데이터 수집 (중복 제거)
def get_ecos_data(stat_code, item_code, freq, name, start_date="201901", end_date="202505"):
    try:
        if not dmqa_check(stat_code, item_code):
            return pd.DataFrame()

        # ✅ 주기별 날짜 포맷 지정
        if freq == 'A':
            start_date, end_date = "2019", "2025"
        elif freq == 'M':
            start_date, end_date = "201901", "202504"
        elif freq == 'D':
            start_date, end_date = "20190101", "20250430"
        elif freq == 'Q':
            start_date, end_date = "2019Q1", "2024Q5"

        url = f"https://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/xml/kr/1/10000/{stat_code}/{freq}/{start_date}/{end_date}/{item_code}"
        res = requests.get(url)
        root = ET.fromstring(res.content)

        dates, values = [], []
        for row in root.findall(".//row"):
            time = row.findtext("TIME")
            value = row.findtext("DATA_VALUE")
            if time and value:
                if freq == 'M':
                    date = pd.to_datetime(time, format='%Y%m')
                elif freq == 'Q':
                    date = pd.to_datetime(time.replace("Q1", "-03-31").replace("Q2", "-06-30").replace("Q3", "-09-30").replace("Q4", "-12-31"))
                elif freq == 'A':
                    date = pd.to_datetime(time, format='%Y')
                elif freq == 'D':
                    date = pd.to_datetime(time, format='%Y%m%d')

                dates.append(date)
                values.append(float(value))

        if dates:
            df = pd.DataFrame(values, index=pd.to_datetime(dates), columns=[name])
            
            return df
        else:
            print(f"⚠️ {name} 데이터가 없습니다.")
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ [ECOS 오류] {name} ({stat_code}-{item_code}) → {e}")
        return pd.DataFrame()

    dates, values = [], []
    for row in root.findall(".//row"):
        time = row.findtext("TIME")
        value = row.findtext("DATA_VALUE")
        if time and value:
            try:
                # ✅ 날짜 형식 처리
                if freq == 'M':
                    date = pd.to_datetime(time, format='%Y%m')
                elif freq == 'Q':
                    date = pd.to_datetime(time.replace("Q1", "-03-31").replace("Q2", "-06-30").replace("Q3", "-09-30").replace("Q4", "-12-31"))
                elif freq == 'A':
                    date = pd.to_datetime(time, format='%Y')
                elif freq == 'D':
                    date = pd.to_datetime(time, format='%Y%m%d')

                # ✅ 중복 방지
                if date not in dates:
                    dates.append(date)
                    values.append(float(value))

            except Exception as e:
                print(f"⚠️ 날짜 변환 오류: {time}, {e}")
# ✅ 추가할 기술적 지표 계산 함수
def add_technical_indicators(df):
    # ✅ Log Return
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # ✅ MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # ✅ Bollinger Bands
    df['Bollinger_Mid'] = df['Close'].rolling(window=20).mean()
    df['Bollinger_Upper'] = df['Bollinger_Mid'] + 2 * df['Close'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['Bollinger_Mid'] - 2 * df['Close'].rolling(window=20).std()
    df['Bollinger_Bandwidth'] = (df['Bollinger_Upper'] - df['Bollinger_Lower']) / df['Bollinger_Mid'] * 100

    # ✅ Donchian Channel
    df['Donchian_High'] = df['Close'].rolling(window=20).max()
    df['Donchian_Low'] = df['Close'].rolling(window=20).min()

    # ✅ Momentum
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)

    # ✅ Cumulative Return
    df['Cumulative_Return'] = (df['Close'] / df['Close'].iloc[0] - 1) * 100

    return df

# ✅ 주식 데이터 수집 (월별 데이터)
def get_stock_data(name, ticker):
    stock = yf.Ticker(ticker)
    end_date_dynamic = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    df = stock.history(start="2019-01-01", end=end_date_dynamic, interval='1d')  # 일별 데이터

    if df.empty:
        print(f"❌ {name} ({ticker}) 데이터 없음")
        return pd.DataFrame()

    # ✅ 종목명 컬럼 추가
    df['종목명'] = name
    df['Ticker'] = ticker 

    # ✅ 기술적 지표 계산
    # 이동평균 계산 초반에는 윈도우 크기만큼의 NaN이 발생합니다. (계산 특성)
    df["변화율(%)"] = df["Close"].pct_change() * 100
    df["5일이평"] = df["Close"].rolling(window=5).mean()
    df["20일이평"] = df["Close"].rolling(window=20).mean()
    df["60일이평"] = df["Close"].rolling(window=60).mean()
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"]).rsi()

    # ✅ 시간대 통일 (UTC -> Asia/Seoul) 및 월별 첫째 날로 인덱스 설정
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Seoul')
    elif df.index.tz.zone != 'Asia/Seoul':
        df.index = df.index.tz_convert('Asia/Seoul')
    # 주식 데이터의 인덱스를 월의 첫째 날로 통일
    df = df.ffill().bfill()
    
    df = add_technical_indicators(df)

    df.reset_index(inplace=True)
    return df





def update_stock_and_macro_data():
    print("[1단계] 한국 주가 + 거시지표 수집 및 병합 시작")

    # ✅ ECOS 지표 수집
    ecos_df_list = []
    for code, details in stat_codes.items():
        for item in details["item_codes"]:
            df = get_ecos_data(code, item["code"], item["cycle"], details["name"])
            print(f"📦 {details['name']} ({code}-{item['code']}) →", "✅ 성공" if not df.empty else "❌ 실패")
            if df is not None and not df.empty:
                ecos_df_list.append(df)
    # ✅ 하나도 못 가져온 경우 에러 방지
    if not ecos_df_list:
        print("❌ 모든 ECOS 지표 수집 실패. API 키 또는 네트워크 상태를 확인하세요.")
        return None
    
    for i in range(len(ecos_df_list)):
        df = ecos_df_list[i]
        df = df[~df.index.duplicated()]                  # 중복 인덱스 제거
        df = df.sort_index()                             # 인덱스 정렬
        
        ecos_df_list[i] = df

    # 🔹 병합
    macro_df = pd.concat(ecos_df_list, axis=1)
    macro_df = macro_df[~macro_df.index.duplicated()]  # 병합 후 중복 제거
    macro_df = macro_df.sort_index()       

    # ✅ 한국 주가 수집 + 기술적 지표
    stock_data_list = [get_stock_data(name, ticker) for name, ticker in kr_tickers.items()]
    stock_data_list = [df for df in stock_data_list if not df.empty]

    if not stock_data_list:
        print("❌ 주가 수집 실패")
        return None

    all_stock_data = pd.concat(stock_data_list, ignore_index=True)

    # ✅ 날짜 병합: 주식은 'Date', 거시지표는 index → reset 후 병합
    macro_df = macro_df.reset_index()

    # ✅ 여기에서 타임존 제거 추가
    all_stock_data["Date"] = pd.to_datetime(all_stock_data["Date"]).dt.tz_localize(None)
    macro_df["index"] = pd.to_datetime(macro_df["index"]).dt.tz_localize(None)
    merged_df = pd.merge(all_stock_data, macro_df, left_on="Date", right_on="index", how="left")
    merged_df.drop(columns=["index"], inplace=True)

    # ✅ 결측치 보간 + 저장
    merged_df.ffill(inplace=True)
    merged_df.bfill(inplace=True)
    merged_df["UpdatedAt"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_df.to_csv(MERGED_FILE, index=False, encoding="utf-8-sig")

    print(f"[1단계] 저장 완료 → {MERGED_FILE}")
    return merged_df

# ------------------------
# 2단계: AI 모델 예측
# ------------------------
def predict_ai_scores(df):
    print("[2단계] AI 예측 시작")

    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.loc[:, ~df.columns.duplicated()]

    # 수익률 및 타겟 생성
    df["Target_1D"] = df.groupby("Ticker")["Close"].shift(-1)
    df["Target_20D"] = df.groupby("Ticker")["Close"].shift(-20)
    df["Return_1D"] = (df["Target_1D"] - df["Close"]) / df["Close"]
    df["Return_20D"] = (df["Target_20D"] - df["Close"]) / df["Close"]
    df["Return"] = df.groupby("Ticker")["Close"].pct_change()
    df = df.ffill().bfill()

    # 훈련 데이터 분리
    train_df = df[df["Date"] <= pd.to_datetime("2024-12-31")].copy()
    X_train = train_df[FEATURE_COLUMNS].fillna(0)
    y_train_1d = train_df["Return_1D"]
    y_train_20d = train_df["Return_20D"]

    # Gradient Boosting 모델 학습
    gb_1d = GradientBoostingRegressor()
    gb_1d.fit(X_train, y_train_1d)

    gb_20d = GradientBoostingRegressor()
    gb_20d.fit(X_train, y_train_20d)

    # ✅ LSTM: 시퀀스 생성
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.callbacks import EarlyStopping

    SEQUENCE_LENGTH = 10
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)
    scaled_df = pd.DataFrame(X_scaled)

    X_lstm_train, y_lstm_train = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_df)):
        X_lstm_train.append(scaled_df.iloc[i - SEQUENCE_LENGTH:i].values)
        y_lstm_train.append(y_train_1d.iloc[i])

    X_lstm_train = np.array(X_lstm_train)
    y_lstm_train = np.array(y_lstm_train)

    # ✅ LSTM 모델 정의 (seed 고정 없음)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    lstm_model = Sequential()
    lstm_model.add(Input(shape=(SEQUENCE_LENGTH, X_scaled.shape[1])))
    lstm_model.add(LSTM(128, return_sequences=False))
    lstm_model.add(BatchNormalization())
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(64, activation='relu'))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(32, activation='relu'))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(
        X_lstm_train,
        y_lstm_train,
        epochs=30,
        batch_size=16,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # ✅ 예측 시작
    test_dates = df[df["Date"] >= pd.to_datetime("2025-05-01")]["Date"].drop_duplicates().sort_values()
    all_preds = []

    for current_date in test_dates:
        test_df = df[df["Date"] == current_date].copy()
        if test_df.empty:
            continue

        test_df[FEATURE_COLUMNS] = test_df[FEATURE_COLUMNS].fillna(method='ffill').fillna(method='bfill').fillna(0)
        if test_df[FEATURE_COLUMNS].isnull().values.any():
            print(f"⚠️ {current_date.date()} → 여전히 NaN 있음, 예측 스킵")
            continue

        # ✅ GB 예측
        test_df["Predicted_Return_GB_1D"] = gb_1d.predict(test_df[FEATURE_COLUMNS]) * 4
        test_df["Predicted_Return_GB_20D"] = gb_20d.predict(test_df[FEATURE_COLUMNS])

        # ✅ LSTM 예측 (종목별 과거 10일 기준)
        lstm_preds = []
        valid_rows = []

        for _, row in test_df.iterrows():
            ticker = row["Ticker"]
            date = row["Date"]
            past_window = df[(df["Ticker"] == ticker) & (df["Date"] < date)].sort_values("Date").tail(SEQUENCE_LENGTH)

            if len(past_window) < SEQUENCE_LENGTH:
                continue

            past_feats = past_window[FEATURE_COLUMNS].fillna(0)
            scaled_feats = scaler.transform(past_feats)
            input_seq = np.expand_dims(scaled_feats, axis=0)
            pred = lstm_model.predict(input_seq, verbose=0)[0][0]
            lstm_preds.append(pred * 30)
            valid_rows.append(row)

        if not valid_rows:
            continue

        test_df = pd.DataFrame(valid_rows)
        test_df["Predicted_Return_LSTM"] = lstm_preds

        all_preds.append(test_df)
        print(f"✅ {current_date.date()} 예측 완료 - {len(test_df)}종목")

    if not all_preds:
        print("❌ 예측 결과 없음. 시뮬레이션 불가")
        return pd.DataFrame()

    result_df = pd.concat(all_preds, ignore_index=True)

    # ✅ 예측 종가 계산
    result_df["예측종가_GB_1D"] = result_df["Close"] * (1 + result_df["Predicted_Return_GB_1D"])
    result_df["예측종가_GB_20D"] = result_df["Close"] * (1 + result_df["Predicted_Return_GB_20D"])
    result_df["예측종가_LSTM"] = result_df["Close"] * (1 + result_df["Predicted_Return_LSTM"])

    result_df.to_csv(PREDICTED_FILE, index=False)
    print(f"[2단계] 전체 예측 결과 저장 완료 → {PREDICTED_FILE}")
    return result_df



# ------------------------
SIMULATION_FILE_SIMPLE_FORMATTED = "data/kr_simulation_result_simple.csv"

def simulate_combined_trading_simple_formatted(df):
    print("[3단계] 통합 모의투자 시작 (누적 보유 + 조건부 부분매도)")

    initial_capital = 10000000
    portfolios = {
        "GB_1D": {"capital": initial_capital, "holding": {}},
        "GB_20D": {"capital": initial_capital, "holding": {}},
        "Dense-LSTM": {"capital": initial_capital, "holding": {}},
    }
    history = []
    TRADE_AMOUNT = 2000000

    df_sorted = df.sort_values(by=["Date", "Ticker"]).copy()
    df_sorted = df_sorted.fillna(method='ffill').fillna(method='bfill')
    df_sorted["Date"] = pd.to_datetime(df_sorted["Date"]).dt.tz_localize(None)
    df_sorted = df_sorted[
    (df_sorted["Date"] >= pd.to_datetime("2025-05-01")) &
    (df_sorted["Date"] <= pd.to_datetime(datetime.now().strftime("%Y-%m-%d")))
]

    if df_sorted.empty:
        print("  - 시뮬레이션할 데이터가 없습니다.")
        return pd.DataFrame()

    print(f"  - {df_sorted['Date'].min().date()} 부터 시뮬레이션 실행 중...")

    for date, date_df in df_sorted.groupby("Date"):
        for model, score_col in zip(
            portfolios.keys(),
            ["Predicted_Return_GB_1D", "Predicted_Return_GB_20D", "Predicted_Return_LSTM"]
        ):
            portfolio = portfolios[model]
            current_holdings = list(portfolio["holding"].keys())

            # 1. 매도 판단
            for ticker in current_holdings:
                holding_info = portfolio["holding"][ticker]
                holding_stock_data = date_df[date_df["Ticker"] == ticker]

                if not holding_stock_data.empty:
                    current_price = holding_stock_data.iloc[0]["Close"]
                    holding_score = holding_stock_data.iloc[0][score_col]
                    holding_value = holding_info["shares"] * current_price

                    if holding_score <= -0.02:
                        shares_to_sell_value = min(TRADE_AMOUNT, holding_value)
                        shares_to_sell = int(shares_to_sell_value // current_price)

                        if shares_to_sell == 0 and holding_value > 0:
                            shares_to_sell = 1
                        shares_to_sell = min(shares_to_sell, holding_info["shares"])

                        if shares_to_sell > 0:
                            sell_price = current_price
                            sell_amount = shares_to_sell * sell_price * 0.999
                            portfolio["capital"] += sell_amount
                            portfolio["holding"][ticker]["shares"] -= shares_to_sell

                            buy_price = holding_info["buy_price"]
                            profit = (sell_price * 0.999 - buy_price) * shares_to_sell
                            profit_str = f"{profit:+.2f}달러"

                            total_asset_after_sell = portfolio["capital"] + sum(
                                h["shares"] * current_price for h in portfolio["holding"].values()
                            )

                            history.append({
                                "날짜": date,
                                "모델": model,
                                "종목명": ticker_to_name.get(ticker, ticker),
                                "예측 수익률": holding_score * 10000,
                                "현재가": sell_price,
                                "매수(매도)": f"SELL ({shares_to_sell}주)",
                                "잔여 현금": portfolio["capital"],
                                "총 자산": total_asset_after_sell
                            })

                            if portfolio["holding"][ticker]["shares"] <= 0:
                                del portfolio["holding"][ticker]

            # 2. 매수 판단 (예측 수익률 > 1%인 상위 4개 종목)
            top_candidates = date_df[date_df[score_col] > 0.01].sort_values(by=score_col, ascending=False).head(2)

            for _, row in top_candidates.iterrows():
                ticker = row["Ticker"]
                score = row[score_col]

                if portfolio["capital"] >= TRADE_AMOUNT:
                    buy_price = row["Close"]
                    buy_value_to_spend = min(TRADE_AMOUNT, portfolio["capital"] * 0.99)
                    shares = int(buy_value_to_spend // (buy_price * 1.001))

                    if shares > 0:
                        cost = shares * buy_price * 1.001
                        portfolio["capital"] -= cost

                        if ticker in portfolio["holding"]:
                            portfolio["holding"][ticker]["shares"] += shares
                        else:
                            portfolio["holding"][ticker] = {
                                "shares": shares,
                                "buy_price": buy_price
                            }

                        total_asset_after_buy = portfolio["capital"] + sum(
                            h["shares"] * buy_price for h in portfolio["holding"].values()
                        )

                        history.append({
                            "날짜": date,
                            "모델": model,
                            "종목명": ticker_to_name.get(ticker, ticker),
                            "티커": ticker,
                            "예측 수익률": score * 10000,
                            "현재가": buy_price,
                            "매수(매도)": f"BUY ({shares}주)",
                            "잔여 현금": portfolio["capital"],
                            "총 자산": total_asset_after_buy
                        })

    result_df = pd.DataFrame(history)
    if not result_df.empty:
        # ✅ 예측 종가 계산 (예측 수익률이 x10000 단위)
        result_df["예측종가"] = result_df["현재가"] * (1 + result_df["예측 수익률"] / 10000)
    
        result_df = result_df[["날짜", "모델", "종목명", "티커", "예측 수익률", "현재가", "예측종가", "매수(매도)", "잔여 현금", "총 자산"]]
        os.makedirs("data", exist_ok=True)
        result_df.to_csv(SIMULATION_FILE_SIMPLE_FORMATTED, index=False)
        print(f"[3단계] 시뮬레이션 결과 저장 완료 → {SIMULATION_FILE_SIMPLE_FORMATTED}")
    else:
        print("[3단계] 시뮬레이션 결과 없음")

    final_assets = {}
    for model, port in portfolios.items():
        holding_summary = {}
        total_holding_value = 0
        for ticker, info in port["holding"].items():
            latest_price_row = df_sorted[df_sorted["Ticker"] == ticker].sort_values(by="Date", ascending=False).head(1)

            if not latest_price_row.empty:
                current_price = latest_price_row.iloc[0]["Close"]
                shares = info["shares"]
                holding_value = shares * current_price
                holding_summary[ticker] = {
                    "보유 수량": shares,
                    "현재가": round(current_price, 2),
                    "평가 금액": round(holding_value, 2)
                }
                total_holding_value += holding_value

        total_asset = total_holding_value + port["capital"]
        final_assets[model] = {
            "현금 잔액": round(port["capital"], 2),
            "총 자산": round(total_asset, 2),
            "보유 종목 수": len(port["holding"]),
            "보유 종목": holding_summary
        }

    return result_df, final_assets

# 4단계: 시각화 (간단한 시뮬레이션 결과로는 시각화가 제한될 수 있습니다)
# ------------------------
def visualize_trades_simple(df, sim_df_simple):
    print("[4단계] 시각화 시작")
    os.makedirs("charts", exist_ok=True)

    # ✅ 한글 깨짐 방지용 폰트 설정
    if platform.system() == 'Windows':
        font_path = "C:/Windows/Fonts/malgun.ttf"
    else:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    sim_df_simple["날짜"] = pd.to_datetime(sim_df_simple["날짜"]).dt.tz_localize(None)

    # ✅ 삼성전자만 필터링
    target_stock = "삼성전자"
    ticker = df[df["종목명"] == target_stock]["Ticker"].unique()[0]
    stock_df = df[df["Ticker"] == ticker].sort_values(by="Date")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(stock_df["Date"], stock_df["Close"], label="종가", linewidth=2, alpha=0.7)

    for model in sim_df_simple["모델"].unique():
        trades = sim_df_simple[(sim_df_simple["티커"] == ticker) & (sim_df_simple["모델"] == model)].copy()

        if trades.empty:
            continue

        trades = pd.merge(
            trades,
            stock_df[["Date", "Close"]].rename(columns={"Close": "Actual_Close"}),
            left_on="날짜",
            right_on="Date",
            how="left"
        )

        # ✅ 매수/매도 시각화
        buys = trades[trades["매수(매도)"].str.contains("BUY", na=False)]
        sells = trades[trades["매수(매도)"].str.contains("SELL", na=False)]

        ax.scatter(buys["날짜"], buys["Actual_Close"], label=f"{model} 매수", marker="^", color="green", zorder=5)
        ax.scatter(sells["날짜"], sells["Actual_Close"], label=f"{model} 매도", marker="v", color="red", zorder=5)

        # ✅ MAE 계산
        if "Predicted_Return_LSTM" in trades.columns and "Actual_Close" in trades.columns:
            trades["예측_종가"] = trades["Actual_Close"] * (1 + trades["Predicted_Return_LSTM"])
            mae = mean_absolute_error(trades["Actual_Close"], trades["예측_종가"])
            ax.plot(trades["날짜"], trades["예측_종가"], label="Dense-LSTM 예측 종가", linestyle="--", alpha=0.7)
        else:
            mae = np.nan

        # ✅ 타이틀에 MAE 추가
        ax.set_title(f"삼성전자 - Dense-LSTM 시뮬레이션 (MAE: {mae:.2f})")

    ax.set_xlabel("날짜")
    ax.set_ylabel("주가 (원)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    # ✅ 저장
    save_path = f"charts/SAMSUNG_trades_simple_Dense-LSTM.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[완료] → {save_path}")


# ------------------------
# 실행
# ------------------------
if __name__ == "__main__":
    merged_df = update_stock_and_macro_data()
    if merged_df is not None:
        merged_df["Date"] = pd.to_datetime(merged_df["Date"])
        latest_date = merged_df["Date"].max().date()
        print(f"📅 수집된 데이터의 가장 결정 날짜: {latest_date}")

        predicted_df = predict_ai_scores(merged_df.copy())

        if not predicted_df.empty:
            simulation_results_simple, final_assets = simulate_combined_trading_simple_formatted(predicted_df.copy())

            if not simulation_results_simple.empty:
                # 간단한 시뮬리언 결과로는 보개화는 ì \xec96b음
                # 거래 시점만 표시하는 시간간화 함수 사용
                visualize_trades_simple(merged_df.copy(), simulation_results_simple.copy())

            print("\n📊 [예측 결과 미리보기 - 마지막 20행]")
            print(predicted_df.tail(20))

            print("\n📈 [시뮬리언 결과 미리보기 - 첫번째 2행]")
            print(simulation_results_simple.to_string(index=False))

            print("\n💼 [최종 포트폴리오 자사현황]")
            if not simulation_results_simple.empty:
                for model, info in final_assets.items():
                    print(f"\n📌 모델: {model}")
                    print(f"  - 총 자산: {info['총 자산']}")
                    print(f"  - 현금 잔액: {info['현금 잔액']}")
                    print(f"  - 보유 종목 수: {info['보유 종목 수']}")

                    if info["보유 종목"]:
                        print("  - 보유 종목:")
                        for ticker, details in info["보유 종목"].items():
                            stock_name = ticker_to_name.get(ticker, ticker)
                            print(f"     ▸ {stock_name}: 수량={details['보유 수량']}주, 현재가=${details['현재가']}, 평가금액=${details['평가 금액']}")
                    else:
                        print("  - 보유 종목 없음")
        else:
            print("시뮬리언 결과가 없습니다.")
