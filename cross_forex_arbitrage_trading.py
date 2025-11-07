"""
Forex Arbitrage System - Synthetic Currency Analysis
Author: Yevgeniy Koshtenko
Description: High-frequency arbitrage system that creates thousands of synthetic prices
             through cross-rates and detects market inefficiencies.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

# Configuration
TERMINAL_PATH = "C:/Program Files/MetaTrader 5/terminal64.exe"
INITIAL_CAPITAL = 10000
TAKE_PROFIT_PIPS = 8
STOP_LOSS_PIPS = 4
MIN_ARBITRAGE_THRESHOLD = 0.00008  # 0.8 pips


class ForexArbitrageSystem:
    """
    Main class for forex arbitrage analysis and trading simulation.
    Detects price discrepancies between direct pairs and synthetic cross-rates.
    """
    
    def __init__(self, terminal_path=TERMINAL_PATH, initial_capital=INITIAL_CAPITAL):
        self.terminal_path = terminal_path
        self.initial_capital = initial_capital
        self.symbols = [
            "AUDUSD", "AUDJPY", "CADJPY", "AUDCHF", "AUDNZD",
            "USDCAD", "USDCHF", "USDJPY", "NZDUSD", "GBPUSD",
            "EURUSD", "CADCHF", "CHFJPY", "NZDCAD", "NZDCHF",
            "NZDJPY", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD",
            "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD"
        ]
        
        # Pairs for synthetic price calculation
        self.synthetic_pairs = [
            ('AUDUSD', 'USDCHF'), ('AUDUSD', 'NZDUSD'), ('AUDUSD', 'USDJPY'),
            ('USDCHF', 'USDCAD'), ('USDCHF', 'NZDCHF'), ('USDCHF', 'CHFJPY'),
            ('USDJPY', 'USDCAD'), ('USDJPY', 'NZDJPY'), ('USDJPY', 'GBPJPY'),
            ('NZDUSD', 'NZDCAD'), ('NZDUSD', 'NZDCHF'), ('NZDUSD', 'NZDJPY'),
            ('GBPUSD', 'GBPCAD'), ('GBPUSD', 'GBPCHF'), ('GBPUSD', 'GBPJPY'),
            ('EURUSD', 'EURCAD'), ('EURUSD', 'EURCHF'), ('EURUSD', 'EURJPY'),
            ('CADCHF', 'CADJPY'), ('CADCHF', 'GBPCAD'), ('CADCHF', 'EURCAD'),
            ('CHFJPY', 'GBPCHF'), ('CHFJPY', 'EURCHF'), ('CHFJPY', 'NZDCHF'),
            ('NZDCAD', 'NZDJPY'), ('NZDCAD', 'GBPNZD'), ('NZDCAD', 'EURNZD'),
            ('NZDCHF', 'NZDJPY'), ('NZDCHF', 'GBPNZD'), ('NZDCHF', 'EURNZD'),
            ('NZDJPY', 'GBPNZD'), ('NZDJPY', 'EURNZD')
        ]
        
    def initialize_mt5(self):
        """Initialize connection to MetaTrader 5."""
        if not mt5.initialize(path=self.terminal_path):
            print(f"❌ Failed to connect to MT5 at {self.terminal_path}")
            print(f"Error: {mt5.last_error()}")
            return False
        print(f"✅ Connected to MT5. Version: {mt5.version()}")
        return True
    
    def shutdown_mt5(self):
        """Close MT5 connection."""
        mt5.shutdown()
        print("MT5 connection closed.")
    
    def get_historical_data(self, start_date, end_date):
        """
        Load historical M1 data for all currency pairs.
        
        Args:
            start_date: datetime with timezone
            end_date: datetime with timezone
            
        Returns:
            dict: {symbol: DataFrame} with OHLC data
        """
        print(f"\n📊 Loading historical data from {start_date.date()} to {end_date.date()}...")
        
        if not self.initialize_mt5():
            return None
        
        historical_data = {}
        timeframe = mt5.TIMEFRAME_M1
        
        for symbol in self.symbols:
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'tick_volume']]
                
                # Add bid/ask (simplified: using close with small spread)
                df['bid'] = df['close']
                df['ask'] = df['close'] + 0.00001  # 0.1 pip spread
                
                # Remove duplicate indices
                df = df[~df.index.duplicated(keep='first')]
                
                historical_data[symbol] = df
                print(f"  ✓ {symbol}: {len(df)} bars loaded")
            else:
                print(f"  ✗ {symbol}: No data available")
        
        self.shutdown_mt5()
        print(f"\n✅ Loaded {len(historical_data)} currency pairs")
        return historical_data
    
    def calculate_synthetic_prices(self, data):
        """
        Calculate synthetic prices through cross-rates.
        Example: AUDCHF_synthetic = AUDUSD / USDCHF
        
        Args:
            data: dict of DataFrames with price data
            
        Returns:
            DataFrame with synthetic prices
        """
        synthetic_prices = {}
        
        for pair1, pair2 in self.synthetic_pairs:
            if pair1 in data and pair2 in data:
                # Method 1: bid/ask (conservative)
                synthetic_prices[f'{pair1}_{pair2}_conservative'] = (
                    data[pair1]['bid'] / data[pair2]['ask']
                )
                
                # Method 2: bid/bid (aggressive)
                synthetic_prices[f'{pair1}_{pair2}_aggressive'] = (
                    data[pair1]['bid'] / data[pair2]['bid']
                )
        
        return pd.DataFrame(synthetic_prices)
    
    def analyze_arbitrage_opportunities(self, data, synthetic_prices):
        """
        Detect arbitrage opportunities by comparing real vs synthetic prices.
        
        Args:
            data: dict of real price DataFrames
            synthetic_prices: DataFrame with synthetic prices
            
        Returns:
            DataFrame with boolean arbitrage signals
        """
        spreads = {}
        
        for symbol in data.keys():
            for synth_column in synthetic_prices.columns:
                if symbol in synth_column:
                    # Calculate spread: real_price - synthetic_price
                    spread = data[symbol]['bid'] - synthetic_prices[synth_column]
                    spreads[synth_column] = spread
        
        spreads_df = pd.DataFrame(spreads)
        
        # Arbitrage exists when spread > threshold
        arbitrage_signals = spreads_df > MIN_ARBITRAGE_THRESHOLD
        
        return arbitrage_signals, spreads_df
    
    def simulate_trade(self, data, direction, entry_price, take_profit_price, stop_loss_price):
        """
        Simulate a single trade with TP/SL.
        
        Args:
            data: DataFrame with price data
            direction: "BUY" or "SELL"
            entry_price: entry price level
            take_profit_price: TP price level
            stop_loss_price: SL price level
            
        Returns:
            dict: {'profit': float, 'duration': int, 'exit_reason': str}
        """
        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row['bid'] if direction == "BUY" else row['ask']
            
            if direction == "BUY":
                # Check TP
                if current_price >= take_profit_price:
                    profit = (take_profit_price - entry_price) * 100000  # Per lot
                    return {'profit': profit, 'duration': i, 'exit_reason': 'TP'}
                
                # Check SL
                if current_price <= stop_loss_price:
                    profit = (stop_loss_price - entry_price) * 100000
                    return {'profit': profit, 'duration': i, 'exit_reason': 'SL'}
            
            else:  # SELL
                # Check TP
                if current_price <= take_profit_price:
                    profit = (entry_price - take_profit_price) * 100000
                    return {'profit': profit, 'duration': i, 'exit_reason': 'TP'}
                
                # Check SL
                if current_price >= stop_loss_price:
                    profit = (entry_price - stop_loss_price) * 100000
                    return {'profit': profit, 'duration': i, 'exit_reason': 'SL'}
        
        # Close at market if TP/SL not hit
        last_price = data['bid'].iloc[-1] if direction == "BUY" else data['ask'].iloc[-1]
        
        if direction == "BUY":
            profit = (last_price - entry_price) * 100000
        else:
            profit = (entry_price - last_price) * 100000
        
        return {'profit': profit, 'duration': len(data), 'exit_reason': 'Market Close'}
    
    def backtest(self, historical_data, start_date, end_date):
        """
        Run full backtest on historical data.
        
        Args:
            historical_data: dict of DataFrames
            start_date: datetime
            end_date: datetime
            
        Returns:
            tuple: (equity_curve, trades_log)
        """
        print(f"\n🔄 Starting backtest from {start_date.date()} to {end_date.date()}...")
        
        equity_curve = [self.initial_capital]
        trades_log = []
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        for current_date in dates:
            # Filter data for current day
            daily_data = {}
            for symbol, df in historical_data.items():
                day_df = df[df.index.date == current_date.date()]
                if not day_df.empty:
                    daily_data[symbol] = day_df
            
            if not daily_data:
                continue
            
            # Calculate synthetic prices
            synthetic_prices = self.calculate_synthetic_prices(daily_data)
            
            if synthetic_prices.empty:
                continue
            
            # Detect arbitrage
            arbitrage_signals, spreads = self.analyze_arbitrage_opportunities(
                daily_data, synthetic_prices
            )
            
            # Execute trades
            for synth_column in arbitrage_signals.columns:
                if arbitrage_signals[synth_column].any():
                    # Extract base symbol
                    base_symbol = synth_column.split('_')[0]
                    
                    if base_symbol not in daily_data or daily_data[base_symbol].empty:
                        continue
                    
                    # Determine direction based on spread sign
                    avg_spread = spreads[synth_column].mean()
                    direction = "BUY" if avg_spread > 0 else "SELL"
                    
                    # Entry parameters
                    entry_price = (daily_data[base_symbol]['bid'].iloc[0] 
                                 if direction == "BUY" 
                                 else daily_data[base_symbol]['ask'].iloc[0])
                    
                    tp_distance = TAKE_PROFIT_PIPS * 0.00001
                    sl_distance = STOP_LOSS_PIPS * 0.00001
                    
                    if direction == "BUY":
                        take_profit_price = entry_price + tp_distance
                        stop_loss_price = entry_price - sl_distance
                    else:
                        take_profit_price = entry_price - tp_distance
                        stop_loss_price = entry_price + sl_distance
                    
                    # Simulate trade
                    trade_result = self.simulate_trade(
                        daily_data[base_symbol],
                        direction,
                        entry_price,
                        take_profit_price,
                        stop_loss_price
                    )
                    
                    # Log trade
                    trades_log.append({
                        'date': current_date,
                        'symbol': base_symbol,
                        'direction': direction,
                        'entry_price': entry_price,
                        'profit': trade_result['profit'],
                        'duration_bars': trade_result['duration'],
                        'exit_reason': trade_result['exit_reason']
                    })
                    
                    # Update equity
                    equity_curve.append(equity_curve[-1] + trade_result['profit'])
            
            if current_date.day % 7 == 0:
                print(f"  📅 Processed: {current_date.date()} | Trades: {len(trades_log)} | Equity: ${equity_curve[-1]:.2f}")
        
        print(f"\n✅ Backtest completed!")
        return equity_curve, trades_log
    
    def generate_report(self, equity_curve, trades_log):
        """
        Generate performance statistics and visualizations.
        """
        if not trades_log:
            print("⚠️ No trades executed during backtest period.")
            return
        
        trades_df = pd.DataFrame(trades_log)
        
        # Calculate statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        losing_trades = len(trades_df[trades_df['profit'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_profit = trades_df['profit'].sum()
        avg_profit = trades_df['profit'].mean()
        avg_win = trades_df[trades_df['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['profit'] <= 0]['profit'].mean() if losing_trades > 0 else 0
        
        profit_factor = (abs(trades_df[trades_df['profit'] > 0]['profit'].sum()) / 
                        abs(trades_df[trades_df['profit'] <= 0]['profit'].sum()) 
                        if losing_trades > 0 else float('inf'))
        
        max_equity = max(equity_curve)
        max_drawdown = max_equity - min(equity_curve[equity_curve.index(max_equity):]) if len(equity_curve) > 1 else 0
        max_drawdown_pct = (max_drawdown / max_equity) * 100 if max_equity > 0 else 0
        
        # Print report
        print("\n" + "="*70)
        print("📊 BACKTEST PERFORMANCE REPORT")
        print("="*70)
        print(f"\n💰 PROFIT & LOSS")
        print(f"  Initial Capital:        ${self.initial_capital:,.2f}")
        print(f"  Final Equity:           ${equity_curve[-1]:,.2f}")
        print(f"  Total Profit:           ${total_profit:,.2f}")
        print(f"  Return on Investment:   {((equity_curve[-1] - self.initial_capital) / self.initial_capital * 100):.2f}%")
        
        print(f"\n📈 TRADE STATISTICS")
        print(f"  Total Trades:           {total_trades}")
        print(f"  Winning Trades:         {winning_trades}")
        print(f"  Losing Trades:          {losing_trades}")
        print(f"  Win Rate:               {win_rate:.2f}%")
        
        print(f"\n💵 AVERAGE PERFORMANCE")
        print(f"  Average Profit/Trade:   ${avg_profit:.2f}")
        print(f"  Average Win:            ${avg_win:.2f}")
        print(f"  Average Loss:           ${avg_loss:.2f}")
        print(f"  Profit Factor:          {profit_factor:.2f}")
        
        print(f"\n📉 RISK METRICS")
        print(f"  Max Drawdown:           ${max_drawdown:.2f}")
        print(f"  Max Drawdown %:         {max_drawdown_pct:.2f}%")
        
        print("\n" + "="*70)
        
        # Plot equity curve
        self.plot_equity_curve(equity_curve)
        
        # Plot trade distribution
        self.plot_trade_distribution(trades_df)
    
    def plot_equity_curve(self, equity_curve):
        """Plot and save equity curve."""
        plt.figure(figsize=(15, 8))
        plt.plot(equity_curve, linewidth=2, color='#2E86AB')
        plt.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        plt.fill_between(range(len(equity_curve)), self.initial_capital, equity_curve, 
                         where=[e >= self.initial_capital for e in equity_curve], 
                         alpha=0.3, color='green', label='Profit Zone')
        plt.fill_between(range(len(equity_curve)), self.initial_capital, equity_curve, 
                         where=[e < self.initial_capital for e in equity_curve], 
                         alpha=0.3, color='red', label='Loss Zone')
        
        plt.title('Forex Arbitrage System - Equity Curve', fontsize=16, fontweight='bold')
        plt.xlabel('Trade Number', fontsize=12)
        plt.ylabel('Account Balance ($)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('equity_curve.png', dpi=150)
        plt.close()
        print("\n📊 Equity curve saved as 'equity_curve.png'")
    
    def plot_trade_distribution(self, trades_df):
        """Plot profit distribution histogram."""
        plt.figure(figsize=(12, 6))
        plt.hist(trades_df['profit'], bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        plt.title('Trade Profit Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Profit ($)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('profit_distribution.png', dpi=150)
        plt.close()
        print("📊 Profit distribution saved as 'profit_distribution.png'")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("🚀 FOREX ARBITRAGE SYSTEM - BACKTEST")
    print("="*70)
    
    # Initialize system
    system = ForexArbitrageSystem()
    
    # Set backtest period
    start_date = datetime(2024, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2024, 8, 31, tzinfo=pytz.UTC)
    
    # Load historical data
    historical_data = system.get_historical_data(start_date, end_date)
    
    if historical_data is None or len(historical_data) == 0:
        print("❌ Failed to load historical data. Exiting.")
        return
    
    # Run backtest
    equity_curve, trades_log = system.backtest(historical_data, start_date, end_date)
    
    # Generate report
    system.generate_report(equity_curve, trades_log)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
