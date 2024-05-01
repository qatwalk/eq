import yfinance as yf

def main(ticker):
    data = yf.Ticker("ticker")
    print(data.dividends)


if __name__ == "__main__":
    main("^SPX")