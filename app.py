import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.optimize import minimize

# Page configuration
st.set_page_config(
    page_title="Investment Portfolio Optimizer",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("📊 Investment Portfolio Optimizer")
st.markdown("""
This app helps you create, analyze, and optimize investment portfolios based on modern portfolio theory.
You can add multiple assets, analyze risk and return, and find the optimal allocation based on your risk tolerance.
""")

# Sidebar for inputs
st.sidebar.header("Portfolio Settings")

# Date range selection
st.sidebar.subheader("Time Period")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365 * 3))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Validate date range
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
elif end_date > datetime.now().date():
    st.sidebar.warning("End date is in the future. Using today's date instead.")
    end_date = datetime.now().date()

# Risk-free rate input
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 5.0, 2.0, 0.1) / 100

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Portfolio Builder", "Risk-Return Analysis", "Portfolio Optimization", "Historical Performance"])


# Function to validate tickers
def validate_tickers(tickers):
    """Check if tickers are valid by fetching a single day of data."""
    valid_tickers = []
    invalid_tickers = []

    with st.spinner("Validating ticker symbols..."):
        for ticker in tickers:
            ticker = ticker.strip()
            if not ticker:  # Skip empty tickers
                continue

            try:
                data = yf.download(ticker, period="1d", progress=False)
                if not data.empty:
                    valid_tickers.append(ticker)
                else:
                    invalid_tickers.append(ticker)
            except Exception:
                invalid_tickers.append(ticker)

    if invalid_tickers:
        st.warning(f"The following tickers could not be validated: {', '.join(invalid_tickers)}")

    return valid_tickers


# Cache function for data loading to improve performance
@st.cache_data(ttl=86400)
def load_data(tickers, start, end):
    """A simpler, more robust function to load stock data."""
    if not tickers:
        return None

    # Create an empty DataFrame to store results
    result_df = pd.DataFrame()

    # Download data for each ticker individually to avoid multi-level column issues
    with st.spinner(f"Loading data for {len(tickers)} tickers..."):
        for ticker in tickers:
            try:
                # Download single ticker data
                ticker_data = yf.download(ticker, start=start, end=end, progress=False)

                if ticker_data.empty:
                    st.warning(f"No data available for {ticker}")
                    continue

                # Check if 'Adj Close' exists
                if 'Adj Close' in ticker_data.columns:
                    # Extract adjusted close and rename the column to the ticker symbol
                    adj_close = ticker_data['Adj Close']
                    adj_close = pd.DataFrame(adj_close)
                    adj_close.columns = [ticker]

                    # If this is the first valid ticker, initialize the result DataFrame
                    if result_df.empty:
                        result_df = adj_close
                    else:
                        # Join with existing data
                        result_df = result_df.join(adj_close, how='outer')
                else:
                    st.warning(f"'Adj Close' column not found for {ticker}")
            except Exception as e:
                st.warning(f"Error loading data for {ticker}: {str(e)}")

    # Check if we have any data
    if result_df.empty:
        st.error("Could not load data for any of the provided tickers.")
        return None

    return result_df


# Function to calculate portfolio performance
def calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    returns = np.sum(mean_returns * weights) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (returns - risk_free_rate) / volatility
    return returns, volatility, sharpe_ratio


# Function to optimize portfolio for maximum Sharpe ratio
def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))

    initial_guess = num_assets * [1. / num_assets]

    def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
        returns, volatility, sharpe = calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        return -sharpe

    result = minimize(neg_sharpe, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)

    return result['x']


# Tab 1: Portfolio Builder
with tab1:
    st.header("Build Your Portfolio")

    # Default tickers for common assets
    default_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'BRK-B', 'JNJ', 'JPM', 'V', 'PG', 'UNH',
                       'HD', 'MA', 'DIS', 'NFLX', 'NVDA', 'PYPL', 'ADBE', 'CRM', 'CSCO', 'VZ']

    col1, col2 = st.columns(2)

    with col1:
        # Manual ticker input with examples
        ticker_input = st.text_input("Enter ticker symbols (comma-separated)", "AAPL, MSFT, GOOGL, AMZN")
        tickers = [ticker.strip() for ticker in ticker_input.split(',') if ticker.strip()]

        # Add option to include common ETFs
        include_etfs = st.checkbox("Include common ETFs")
        if include_etfs:
            etfs = ["SPY", "QQQ", "VTI", "AGG", "GLD"]
            tickers.extend(etfs)
            st.info(f"Added ETFs: {', '.join(etfs)}")

    with col2:
        # Popular stocks selection
        st.subheader("Or select from popular stocks:")
        selected_defaults = st.multiselect("Select tickers", default_tickers)
        if selected_defaults:
            tickers.extend([t for t in selected_defaults if t not in tickers])

    # Remove duplicates and empty values
    tickers = list(dict.fromkeys(tickers))

    # Load data
    if st.button("Load Stock Data"):
        if not tickers:
            st.error("Please enter at least one ticker symbol.")
        else:
            # First validate the tickers
            valid_tickers = validate_tickers(tickers)

            if not valid_tickers:
                st.error("No valid ticker symbols found. Please check your input.")
            else:
                with st.spinner("Loading stock data..."):
                    data = load_data(valid_tickers, start_date, end_date)

                    if data is not None and not data.empty:
                        # Display stock price chart
                        st.subheader("Historical Stock Prices")
                        fig = px.line(data, x=data.index, y=data.columns,
                                      title="Historical Stock Prices",
                                      labels={"value": "Price ($)", "variable": "Stock"})
                        st.plotly_chart(fig, use_container_width=True)

                        # Calculate daily returns
                        returns = data.pct_change().dropna()

                        # Check if we have enough data points for correlation
                        if len(returns) > 5:
                            # Display correlation heatmap
                            st.subheader("Correlation Matrix")
                            corr = returns.corr()
                            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                            fig.update_layout(title="Correlation Between Assets")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(
                                "Not enough data points to calculate correlation. Please select a longer date range.")

                        # Store data in session state for other tabs
                        st.session_state['data'] = data
                        st.session_state['returns'] = returns
                        st.session_state['tickers'] = valid_tickers
                        st.success("Data loaded successfully! You can now proceed to the analysis tabs.")
                    else:
                        st.error("Failed to load stock data. Please try different tickers or date range.")

# Tab 2: Risk-Return Analysis
with tab2:
    st.header("Risk-Return Analysis")

    if 'returns' in st.session_state and st.session_state['returns'] is not None:
        returns = st.session_state['returns']

        # Calculate annualized returns and volatility
        mean_returns = returns.mean()
        ann_returns = mean_returns * 252 * 100  # Annualized returns in percentage
        ann_volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility in percentage

        # Display metrics
        metrics_df = pd.DataFrame({
            'Annualized Return (%)': ann_returns,
            'Annualized Volatility (%)': ann_volatility,
            'Sharpe Ratio': (ann_returns / 100 - risk_free_rate) / (ann_volatility / 100)
        })

        st.dataframe(metrics_df)

        # Risk-Return scatter plot
        st.subheader("Risk-Return Profile")
        fig = px.scatter(
            x=ann_volatility,
            y=ann_returns,
            text=returns.columns,
            labels={"x": "Annualized Volatility (%)", "y": "Annualized Return (%)"},
            title="Risk vs Return for Individual Assets"
        )
        fig.update_traces(marker=dict(size=12), textposition='top center')
        st.plotly_chart(fig, use_container_width=True)

        # Current weights input
        st.subheader("Current Portfolio Allocation")
        col1, col2 = st.columns([3, 1])

        with col1:
            weights_input = {}
            for ticker in returns.columns:
                weights_input[ticker] = st.slider(f"{ticker} Weight (%)", 0, 100, int(100 / len(returns.columns)))

            # Normalize weights to sum to 100%
            total_weight = sum(weights_input.values())
            if total_weight != 100:
                st.warning(f"Total weight is {total_weight}%. Normalizing to 100%.")
                weights_input = {k: v / total_weight * 100 for k, v in weights_input.items()}

            weights = np.array([weights_input[ticker] / 100 for ticker in returns.columns])

        with col2:
            # Display weights as pie chart
            fig = px.pie(values=weights, names=returns.columns, title="Portfolio Allocation")
            st.plotly_chart(fig)

        # Calculate portfolio performance
        portfolio_return = np.sum(mean_returns * weights) * 252 * 100
        cov_matrix = returns.cov()
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252) * 100
        portfolio_sharpe = (portfolio_return / 100 - risk_free_rate) / (portfolio_volatility / 100)

        # Display portfolio metrics
        st.subheader("Portfolio Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Annual Return", f"{portfolio_return:.2f}%")
        col2.metric("Expected Annual Volatility", f"{portfolio_volatility:.2f}%")
        col3.metric("Sharpe Ratio", f"{portfolio_sharpe:.2f}")

        # Store in session state for other tabs
        st.session_state['weights_input'] = weights_input
        st.session_state['weights'] = weights
        st.session_state['cov_matrix'] = cov_matrix

    else:
        st.info("Please load stock data in the Portfolio Builder tab first.")

# Tab 3: Portfolio Optimization
with tab3:
    st.header("Portfolio Optimization")

    if 'returns' in st.session_state and st.session_state['returns'] is not None:
        returns = st.session_state['returns']
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        st.subheader("Efficient Frontier")

        # Check if we have enough assets to optimize
        if len(returns.columns) < 2:
            st.warning("Portfolio optimization requires at least 2 assets. Please add more tickers.")
        else:
            try:
                # Generate random portfolios for efficient frontier
                num_portfolios = 3000
                results = np.zeros((3, num_portfolios))
                weights_record = []

                with st.spinner("Generating efficient frontier..."):
                    for i in range(num_portfolios):
                        weights = np.random.random(len(returns.columns))
                        weights /= np.sum(weights)
                        weights_record.append(weights)

                        portfolio_return, portfolio_volatility, portfolio_sharpe = calculate_portfolio_performance(
                            weights, mean_returns, cov_matrix, risk_free_rate)

                        results[0, i] = portfolio_volatility * 100  # Convert to percentage
                        results[1, i] = portfolio_return * 100  # Convert to percentage
                        results[2, i] = portfolio_sharpe

                # Find optimum portfolio
                optimal_weights = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)
                opt_return, opt_volatility, opt_sharpe = calculate_portfolio_performance(
                    optimal_weights, mean_returns, cov_matrix, risk_free_rate)

                # Find min volatility portfolio
                min_vol_weights = minimize(
                    lambda weights: np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252),
                    len(returns.columns) * [1. / len(returns.columns)],
                    method='SLSQP',
                    bounds=tuple((0, 1) for i in range(len(returns.columns))),
                    constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
                )['x']

                min_vol_return, min_vol_volatility, min_vol_sharpe = calculate_portfolio_performance(
                    min_vol_weights, mean_returns, cov_matrix, risk_free_rate)

                # Plot efficient frontier
                fig = go.Figure()

                # Add random portfolios scatter
                fig.add_trace(go.Scatter(
                    x=results[0, :],
                    y=results[1, :],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=results[2, :],
                        colorscale='Viridis',
                        colorbar=dict(title="Sharpe Ratio"),
                        showscale=True
                    ),
                    name='Portfolios'
                ))

                # Add max Sharpe portfolio
                fig.add_trace(go.Scatter(
                    x=[opt_volatility * 100],
                    y=[opt_return * 100],
                    mode='markers',
                    marker=dict(
                        size=15,
                        symbol='star',
                        color='red'
                    ),
                    name='Max Sharpe Ratio'
                ))

                # Add min volatility portfolio
                fig.add_trace(go.Scatter(
                    x=[min_vol_volatility * 100],
                    y=[min_vol_return * 100],
                    mode='markers',
                    marker=dict(
                        size=15,
                        symbol='circle',
                        color='green'
                    ),
                    name='Min Volatility'
                ))

                # Add individual assets
                for i, ticker in enumerate(returns.columns):
                    fig.add_trace(go.Scatter(
                        x=[returns.std()[i] * np.sqrt(252) * 100],
                        y=[mean_returns[i] * 252 * 100],
                        mode='markers+text',
                        marker=dict(size=10),
                        text=ticker,
                        textposition="top center",
                        name=ticker
                    ))

                fig.update_layout(
                    title='Portfolio Optimization - Efficient Frontier',
                    xaxis_title='Annualized Volatility (%)',
                    yaxis_title='Annualized Return (%)',
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

                # Store optimization results in session state
                st.session_state['optimal_weights'] = optimal_weights
                st.session_state['opt_return'] = opt_return
                st.session_state['opt_volatility'] = opt_volatility
                st.session_state['opt_sharpe'] = opt_sharpe
                st.session_state['min_vol_weights'] = min_vol_weights
                st.session_state['min_vol_return'] = min_vol_return
                st.session_state['min_vol_volatility'] = min_vol_volatility
                st.session_state['min_vol_sharpe'] = min_vol_sharpe

                # Display optimal portfolios
                st.subheader("Optimal Portfolio Allocations")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Maximum Sharpe Ratio Portfolio")
                    optimal_df = pd.DataFrame({
                        'Asset': returns.columns,
                        'Allocation (%)': [round(w * 100, 2) for w in optimal_weights]
                    }).sort_values('Allocation (%)', ascending=False)

                    st.dataframe(optimal_df)

                    # Display optimal portfolio metrics
                    st.metric("Expected Annual Return", f"{opt_return * 100:.2f}%")
                    st.metric("Expected Annual Volatility", f"{opt_volatility * 100:.2f}%")
                    st.metric("Sharpe Ratio", f"{opt_sharpe:.2f}")

                    # Pie chart of optimal allocation
                    fig = px.pie(optimal_df, values='Allocation (%)', names='Asset', title="Optimal Allocation")
                    st.plotly_chart(fig)

                with col2:
                    st.markdown("### Minimum Volatility Portfolio")
                    min_vol_df = pd.DataFrame({
                        'Asset': returns.columns,
                        'Allocation (%)': [round(w * 100, 2) for w in min_vol_weights]
                    }).sort_values('Allocation (%)', ascending=False)

                    st.dataframe(min_vol_df)

                    # Display min vol portfolio metrics
                    st.metric("Expected Annual Return", f"{min_vol_return * 100:.2f}%")
                    st.metric("Expected Annual Volatility", f"{min_vol_volatility * 100:.2f}%")
                    st.metric("Sharpe Ratio", f"{min_vol_sharpe:.2f}")

                    # Pie chart of min vol allocation
                    fig = px.pie(min_vol_df, values='Allocation (%)', names='Asset',
                                 title="Minimum Volatility Allocation")
                    st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")
                st.info(
                    "This could be due to insufficient data or numerical instability. Try different assets or a longer time period.")

    else:
        st.info("Please load stock data in the Portfolio Builder tab first.")

# Tab 4: Historical Performance
with tab4:
    st.header("Historical Performance Analysis")

    if 'data' in st.session_state and 'returns' in st.session_state and st.session_state['data'] is not None:
        data = st.session_state['data']
        returns = st.session_state['returns']

        # Input optimal weights
        st.subheader("Select Portfolio Allocation")
        allocation_option = st.radio(
            "Choose portfolio allocation:",
            ["Current Weights", "Max Sharpe Ratio Weights", "Min Volatility Weights", "Custom Weights"]
        )

        if allocation_option == "Current Weights":
            # Use weights from Tab 2 if available
            if 'weights' in st.session_state:
                weights = st.session_state['weights']
            else:
                weights = np.array([1 / len(returns.columns) for _ in returns.columns])  # Equal weights
                st.info("Using equal weights since current weights are not defined.")

        elif allocation_option == "Max Sharpe Ratio Weights":
            if 'optimal_weights' in st.session_state:
                weights = st.session_state['optimal_weights']
            else:
                # Calculate optimal weights
                try:
                    weights = optimize_portfolio(returns.mean(), returns.cov(), risk_free_rate)
                except Exception as e:
                    st.error(f"Error calculating optimal weights: {str(e)}")
                    weights = np.array(
                        [1 / len(returns.columns) for _ in returns.columns])  # Fall back to equal weights
                    st.info("Using equal weights due to optimization error.")

        elif allocation_option == "Min Volatility Weights":
            if 'min_vol_weights' in st.session_state:
                weights = st.session_state['min_vol_weights']
            else:
                # Calculate min vol weights
                try:
                    weights = minimize(
                        lambda weights: np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights))) * np.sqrt(252),
                        len(returns.columns) * [1. / len(returns.columns)],
                        method='SLSQP',
                        bounds=tuple((0, 1) for i in range(len(returns.columns))),
                        constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
                    )['x']
                except Exception as e:
                    st.error(f"Error calculating minimum volatility weights: {str(e)}")
                    weights = np.array(
                        [1 / len(returns.columns) for _ in returns.columns])  # Fall back to equal weights
                    st.info("Using equal weights due to optimization error.")

        else:  # Custom Weights
            st.write("Enter custom weights for each asset:")
            custom_weights = {}
            for ticker in returns.columns:
                custom_weights[ticker] = st.slider(f"{ticker} Weight (%)", 0, 100, int(100 / len(returns.columns)),
                                                   key=f"custom_{ticker}")

            # Normalize weights to sum to 100%
            total_weight = sum(custom_weights.values())
            if total_weight != 100:
                st.warning(f"Total weight is {total_weight}%. Normalizing to 100%.")
                custom_weights = {k: v / total_weight * 100 for k, v in custom_weights.items()}

            weights = np.array([custom_weights[ticker] / 100 for ticker in returns.columns])

        # Display selected weights
        weights_df = pd.DataFrame({
            'Asset': returns.columns,
            'Allocation (%)': [round(w * 100, 2) for w in weights]
        }).sort_values('Allocation (%)', ascending=False)

        col1, col2 = st.columns([3, 2])

        with col1:
            st.dataframe(weights_df)

        with col2:
            fig = px.pie(weights_df, values='Allocation (%)', names='Asset', title="Portfolio Allocation")
            st.plotly_chart(fig)

        # Calculate historical portfolio performance
        portfolio_returns = returns.dot(weights)

        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1

        # Get common benchmark returns (S&P 500)
        benchmark_ticker = st.selectbox("Select benchmark:", ["SPY", "QQQ", "VTI", "^GSPC", "^DJI", "^IXIC"])

        try:
            with st.spinner(f"Loading benchmark data for {benchmark_ticker}..."):
                benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)[
                    'Adj Close']

                if benchmark_data.empty:
                    st.error(
                        f"No data available for benchmark {benchmark_ticker}. Please select a different benchmark.")
                else:
                    benchmark_returns = benchmark_data.pct_change().dropna()

                    # Match benchmark returns index with portfolio returns
                    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
                    if len(common_dates) < 5:
                        st.error("Insufficient overlapping data between portfolio and benchmark.")
                    else:
                        portfolio_returns_aligned = portfolio_returns.loc[common_dates]
                        benchmark_returns_aligned = benchmark_returns.loc[common_dates]

                        # Calculate cumulative returns on aligned data
                        cumulative_portfolio = (1 + portfolio_returns_aligned).cumprod() - 1
                        cumulative_benchmark = (1 + benchmark_returns_aligned).cumprod() - 1

                        # Create combined dataframe for plotting
                        comparison_df = pd.DataFrame({
                            'Portfolio': cumulative_portfolio,
                            benchmark_ticker: cumulative_benchmark
                        })

                        # Plot cumulative returns comparison
                        st.subheader("Cumulative Return Comparison")
                        fig = px.line(
                            comparison_df,
                            title=f"Portfolio vs {benchmark_ticker} Performance",
                            labels={'value': 'Cumulative Returns', 'variable': ''}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Calculate performance metrics
                        portfolio_total_return = cumulative_portfolio.iloc[-1] * 100
                        benchmark_total_return = cumulative_benchmark.iloc[-1] * 100

                        portfolio_annual_return = portfolio_returns_aligned.mean() * 252 * 100
                        benchmark_annual_return = benchmark_returns_aligned.mean() * 252 * 100

                        portfolio_volatility = portfolio_returns_aligned.std() * np.sqrt(252) * 100
                        benchmark_volatility = benchmark_returns_aligned.std() * np.sqrt(252) * 100

                        portfolio_sharpe = (portfolio_annual_return / 100 - risk_free_rate) / (
                                    portfolio_volatility / 100)
                        benchmark_sharpe = (benchmark_annual_return / 100 - risk_free_rate) / (
                                    benchmark_volatility / 100)

                        # Calculate drawdowns
                        portfolio_cumulative = (1 + portfolio_returns_aligned).cumprod()
                        portfolio_running_max = portfolio_cumulative.cummax()
                        portfolio_drawdown = (portfolio_cumulative / portfolio_running_max - 1) * 100

                        benchmark_cumulative = (1 + benchmark_returns_aligned).cumprod()
                        benchmark_running_max = benchmark_cumulative.cummax()
                        benchmark_drawdown = (benchmark_cumulative / benchmark_running_max - 1) * 100

                        # Display metrics comparison
                        st.subheader("Performance Metrics Comparison")
                        metrics_comparison = pd.DataFrame({
                            'Metric': ['Total Return (%)', 'Annual Return (%)', 'Annual Volatility (%)', 'Sharpe Ratio',
                                       'Max Drawdown (%)'],
                            'Portfolio': [
                                f"{portfolio_total_return:.2f}%",
                                f"{portfolio_annual_return:.2f}%",
                                f"{portfolio_volatility:.2f}%",
                                f"{portfolio_sharpe:.2f}",
                                f"{portfolio_drawdown.min():.2f}%"
                            ],
                            f'{benchmark_ticker}': [
                                f"{benchmark_total_return:.2f}%",
                                f"{benchmark_annual_return:.2f}%",
                                f"{benchmark_volatility:.2f}%",
                                f"{benchmark_sharpe:.2f}",
                                f"{benchmark_drawdown.min():.2f}%"
                            ]
                        })

                        st.table(metrics_comparison)

                        # Plot drawdowns
                        drawdown_df = pd.DataFrame({
                            'Portfolio': portfolio_drawdown,
                            benchmark_ticker: benchmark_drawdown
                        })

                        st.subheader("Drawdown Analysis")
                        fig = px.line(
                            drawdown_df,
                            title="Historical Drawdowns",
                            labels={'value': 'Drawdown (%)', 'variable': ''}
                        )
                        fig.update_yaxes(autorange="reversed")  # Reverse y-axis for better visualization
                        st.plotly_chart(fig, use_container_width=True)

                        # Check if we have enough data for monthly analysis
                        if len(portfolio_returns_aligned) >= 30:  # At least a month of data
                            # Monthly returns heatmap
                            st.subheader("Monthly Returns Heatmap")

                            try:
                                # Resample to monthly returns
                                monthly_returns = portfolio_returns_aligned.resample('M').apply(
                                    lambda x: (1 + x).prod() - 1) * 100
                                monthly_returns_df = pd.DataFrame(monthly_returns)
                                monthly_returns_df.index = monthly_returns_df.index.strftime('%b-%Y')
                                monthly_returns_df.columns = ['Monthly Return (%)']

                                # Reshape for heatmap
                                monthly_pivot = monthly_returns_df.copy()
                                monthly_pivot['Year'] = pd.to_datetime(monthly_pivot.index, format='%b-%Y').year
                                monthly_pivot['Month'] = pd.to_datetime(monthly_pivot.index, format='%b-%Y').strftime(
                                    '%b')

                                # Sort months in chronological order
                                month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
                                               'Nov', 'Dec']
                                monthly_pivot['Month_num'] = monthly_pivot['Month'].apply(
                                    lambda x: month_order.index(x) if x in month_order else -1)
                                monthly_pivot = monthly_pivot.sort_values(['Year', 'Month_num'])

                                # Create pivot table
                                pivot_table = monthly_pivot.pivot_table(
                                    values='Monthly Return (%)',
                                    index='Year',
                                    columns='Month',
                                    aggfunc='sum'
                                )

                                # Reorder columns if present
                                available_months = [m for m in month_order if m in pivot_table.columns]
                                if available_months:
                                    pivot_table = pivot_table.reindex(columns=available_months)

                                # Create heatmap
                                fig = px.imshow(
                                    pivot_table,
                                    text_auto='.2f',
                                    color_continuous_scale='RdYlGn',
                                    labels=dict(x="Month", y="Year", color="Return (%)"),
                                    aspect="auto"
                                )
                                fig.update_layout(title="Monthly Portfolio Returns (%)")
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not generate monthly returns heatmap: {str(e)}")
                                st.info(
                                    "This might be due to insufficient data or date range spanning less than a month.")
                        else:
                            st.info("Not enough data for monthly returns analysis. Please select a longer date range.")


            # Instructions and additional information at the bottom
            st.markdown("---")
            st.markdown("""
                        ### How to Use This App
                        1. In the **Portfolio Builder** tab, enter stock tickers and load data
                        2. View the **Risk-Return Analysis** to understand individual asset performance
                        3. Use the **Portfolio Optimization** to find optimal asset allocations
                        4. Check **Historical Performance** to see how your portfolio would have performed

                        ### References
                        - [Modern Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
                        - [Efficient Frontier](https://en.wikipedia.org/wiki/Efficient_frontier)
                        - [Sharpe Ratio](https://en.wikipedia.org/wiki/Sharpe_ratio)
                        """)

            # Footer with project information
            st.markdown("---")
            st.markdown("""
                        **Course Name:** AF3005 – Programming for Finance  
                        **Instructor:** Dr. Usama Arshad  
                        **Created by:** Your Name  
                        **GitHub Repository:** [Link to your GitHub repository]  
                        """)

            # Display a helpful error message if the app was just started
            if 'data' not in st.session_state and 'returns' not in st.session_state:
                st.info(
                    "👆 Start by entering ticker symbols in the Portfolio Builder tab and clicking 'Load Stock Data'.")
        finally:
            print("Thank You for Using this App")