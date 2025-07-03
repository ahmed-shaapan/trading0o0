# Trading Signal Validation Rules

This document describes the enhanced validation rules implemented to prevent invalid trading signals.

## Validation Rules

### 1. **Sell Date Must Be After Buy Date (Minimum 1 Month)**
- A sell signal must be placed at least 30 days after the corresponding buy signal
- This prevents day trading and ensures meaningful position holding periods
- **Error Message**: "Sell signal must be at least 1 month after buy signal. Earliest allowed date: [DATE]"

### 2. **No Buy Signals Between Existing Buy-Sell Pairs**
- Cannot place a buy signal within an existing complete trade period
- If you have Buy(Jan 1) → Sell(Mar 1), you cannot add another Buy(Feb 1)
- **Error Message**: "Cannot place buy signal within existing trade period ([BUY_DATE] to [SELL_DATE])"

### 3. **No Sell Signals Between Existing Buy-Sell Pairs**
- Cannot place a sell signal within an existing complete trade period
- If you have Buy(Jan 1) → Sell(Mar 1), you cannot add another Sell(Feb 1)
- **Error Message**: "Cannot place sell signal within existing trade period ([BUY_DATE] to [SELL_DATE])"

### 4. **Proper Buy-Sell Sequence**
- First signal must always be a buy
- Signals must alternate: Buy → Sell → Buy → Sell...
- Cannot have two consecutive buy or sell signals
- **Error Messages**: 
  - "First signal must be a 'buy', not a 'sell'"
  - "Cannot add a 'buy' after another 'buy'. Must be a 'sell' first."
  - "Cannot add a 'sell' without a preceding 'buy'."

### 5. **Historical Sequence Integrity**
- When placing signals in the past, they must maintain proper sequence
- Cannot break the alternating buy-sell pattern at any point in time
- **Error Message**: "Signal at this date would break the buy-sell sequence. Expected [buy/sell] signal."

## Examples

### ✅ Valid Signal Sequences:
```
1. Buy(2024-01-01) → Sell(2024-02-15) → Buy(2024-03-01) → Sell(2024-04-15)
2. Buy(2024-01-01) → Sell(2024-03-01) [single complete trade]
3. Buy(2024-01-01) [open position, no sell yet]
```

### ❌ Invalid Signal Attempts:
```
1. Sell(2024-01-01) [first signal cannot be sell]
2. Buy(2024-01-01) → Sell(2024-01-15) [sell too soon, needs 30+ days]
3. Buy(2024-01-01) → Sell(2024-03-01) → Buy(2024-02-01) [buy between existing trade]
4. Buy(2024-01-01) → Buy(2024-02-01) [consecutive buys not allowed]
5. Buy(2024-01-01) → Sell(2024-03-01) → Sell(2024-04-01) [consecutive sells not allowed]
```

## Implementation

The validation is handled by the `validate_signal_placement()` function in `app.py`, which is called before any signal is added to the system. This ensures data integrity and enforces proper trading logic.

## Benefits

1. **Prevents Invalid Trades**: Ensures all signals represent realistic trading scenarios
2. **Maintains Data Quality**: Prevents corrupted signal data that could skew analysis
3. **User Guidance**: Clear error messages help users understand trading rules
4. **Consistent Logic**: Uniform validation across all signal entry methods
