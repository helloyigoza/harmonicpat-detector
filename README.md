# harmonic-detector

OHLC fiyat verisi üzerinde harmonik pattern tespiti yapan Python modülü.  
A Python module for detecting harmonic price patterns in OHLC data.

---

## Türkçe

### Ne Yapar?

Gartley, Bat, Butterfly, Crab, Cypher ve Shark harmonik formasyonlarını tespit eder. ZigZag algoritması ile pivot noktaları temizlenir, Fibonacci oranları doğrulanır ve RMSE tabanlı kalite skoru ile desenler sıralanır.

### Özellikler

- **Smart ZigZag:** Piyasa gürültüsünü filtreler, yalnızca anlamlı High/Low dönüşlerini alır
- **Vektörizasyon:** NumPy + isteğe bağlı Numba ile hızlı pivot hesaplama
- **Kalite Skoru:** Her desen için RMSE tabanlı hata skoru (düşük = daha iyi)
- **PRZ Hesaplama:** Tamamlanmamış desenler için olası dönüş bölgesi tahmini
- **Polars / Pandas:** Her iki DataFrame formatını da destekler

### Kurulum

```bash
pip install numpy polars
# İsteğe bağlı hız artışı için:
pip install numba
```

### Kullanım

```python
from harmonic_detector import calculate_harmonic_patterns

result = calculate_harmonic_patterns(
    df,                        # Polars veya Pandas DataFrame (open, high, low, close, timestamp)
    pivot_depth=5,             # Pivot pencere derinliği
    error_margin_percent=15.0, # Fibonacci oranı toleransı (%)
    enabled_patterns=["Gartley", "Bat"],  # None = tümü
    max_patterns_to_return=5
)

for pattern in result["patterns"]:
    print(pattern["name"], pattern["status"], pattern["quality_score"])
```

### Desteklenen Formasyonlar

| Formasyon  | XB Oranı     | XD Oranı    |
|------------|--------------|-------------|
| Gartley    | 0.618        | 0.786       |
| Bat        | 0.382–0.5    | 0.886       |
| Butterfly  | 0.786        | 1.27–1.618  |
| Crab       | 0.382–0.618  | 1.618       |
| Cypher     | 0.382–0.618  | 0.786       |
| Shark      | 0.382–0.618  | 0.886–1.13  |

---

## English

### What It Does

Detects Gartley, Bat, Butterfly, Crab, Cypher, and Shark harmonic chart patterns. Pivot points are cleaned with a Smart ZigZag algorithm, Fibonacci ratios are validated, and patterns are ranked by RMSE-based quality score.

### Features

- **Smart ZigZag:** Filters market noise, enforces strictly alternating High/Low pivots
- **Vectorized:** Fast pivot detection with NumPy + optional Numba JIT
- **Quality Score:** RMSE-based error score per pattern (lower = better fit)
- **PRZ Calculation:** Potential Reversal Zone estimate for incomplete patterns
- **Polars / Pandas:** Accepts both DataFrame formats

### Installation

```bash
pip install numpy polars
# Optional speedup:
pip install numba
```

### Usage

```python
from harmonic_detector import calculate_harmonic_patterns

result = calculate_harmonic_patterns(
    df,                        # Polars or Pandas DataFrame (open, high, low, close, timestamp)
    pivot_depth=5,             # Pivot window depth
    error_margin_percent=15.0, # Fibonacci ratio tolerance (%)
    enabled_patterns=["Gartley", "Bat"],  # None = all patterns
    max_patterns_to_return=5
)

for pattern in result["patterns"]:
    print(pattern["name"], pattern["status"], pattern["quality_score"])
```

### Return Format

```json
{
  "success": true,
  "patterns": [
    {
      "name": "Bullish Gartley",
      "status": "Complete",
      "confirmation_timestamp": "2024-01-15T10:30:00",
      "points": { "X": {}, "A": {}, "B": {}, "C": {}, "D": {} },
      "ratios": { "XB": 0.618, "AC": 0.712, "BD": 1.31, "XD": 0.789 },
      "quality_score": 0.0043
    }
  ],
  "error": null
}
```

### Requirements

- Python 3.10+
- numpy
- polars
- numba *(optional)*

---

## License

MIT
