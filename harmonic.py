from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, TypeVar, Union, NamedTuple

import numpy as np
import polars as pl

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Assume strict backward compatibility for imports
try:
    from utils.time_standardizer import TimeStandardizer
except ImportError:
    # Fallback if utils not available in this context
    class TimeStandardizer:
        @staticmethod
        def to_iso_format(ts):
            return ts.isoformat()

logger = logging.getLogger(__name__)

# --- Enums & Constants ---

class PatternDirection(Enum):
    BULLISH = auto()
    BEARISH = auto()

class PatternStatus(Enum):
    COMPLETE = "Complete"
    POTENTIAL = "Potential"

class PivotType(Enum):
    HIGH = "high"
    LOW = "low"

# --- Configuration Data Structures ---

@dataclass(frozen=True, slots=True)
class FibonacciRatio:
    """Immutable Fibonacci ratio configuration with vectorized validation."""
    target: float  # The ideal ratio (e.g., 0.618)
    min_value: float
    max_value: float
    
    def validate(self, value: float, error_margin: float) -> bool:
        # Dynamic error margin calculation
        
        # If strict min/max are provided (and different), use them as HARD bounds.
        # This allows for wide ranges (like Crab AC 0.382-0.886) where the target (0.618)
        # plus error margin might not cover the full valid range.
        if self.min_value != self.max_value:
             return self.min_value <= value <= self.max_value
             
        # If min == max (e.g. 0.618, 0.618), it means "Ideal Target Only".
        # We MUST use the error margin here to allow for real-world variation.
        lower = self.target * (1 - error_margin)
        upper = self.target * (1 + error_margin)
        return lower <= value <= upper

    def error_score(self, value: float) -> float:
        """Calculate squared error for ranking quality."""
        if self.target == 0: return 0.0
        return (value - self.target) ** 2

@dataclass(frozen=True, slots=True)
class PatternRatios:
    """Fibonacci ratios for a harmonic pattern."""
    xb: FibonacciRatio
    ac: FibonacciRatio
    bd: FibonacciRatio
    xd: FibonacciRatio
    cd_xc: Optional[FibonacciRatio] = None

@dataclass(frozen=True, slots=True)
class HarmonicPatternConfig:
    name: str
    ratios: PatternRatios
    supports_bullish: bool = True
    supports_bearish: bool = True

# --- Data Structures for Results ---

@dataclass(slots=True)
class PivotPoint:
    timestamp: datetime
    price: float
    pivot_type: PivotType
    index: int  # Added for array indexing performance

    def to_dict(self) -> Dict:
        return {
            "timestamp": TimeStandardizer.to_iso_format(self.timestamp),
            "price": round(self.price, 4)
        }

@dataclass(slots=True)
class PatternPoints:
    x: PivotPoint
    a: PivotPoint
    b: PivotPoint
    c: PivotPoint
    d: Optional[PivotPoint] = None
    
    def to_dict(self) -> Dict:
        points = {
            "X": self.x.to_dict(),
            "A": self.a.to_dict(),
            "B": self.b.to_dict(),
            "C": self.c.to_dict()
        }
        if self.d:
            points["D"] = self.d.to_dict()
        return points

@dataclass(slots=True)
class PotentialReversalZone:
    level_1: float
    level_2: float
    
    def to_dict(self) -> Dict:
        return {
            "price_level_1": round(self.level_1, 4),
            "price_level_2": round(self.level_2, 4)
        }

@dataclass(slots=True)
class HarmonicPattern:
    name: str
    direction: PatternDirection
    status: PatternStatus
    points: PatternPoints
    confirmation_timestamp: datetime
    quality_score: float  # Lower is better (closer to 0 error)
    ratios: Optional[Dict[str, float]] = None
    prz: Optional[PotentialReversalZone] = None

    def to_dict(self) -> Dict:
        pattern_dict = {
            "name": f"{self.direction.name.capitalize()} {self.name}",
            "status": self.status.value,
            "confirmation_timestamp": TimeStandardizer.to_iso_format(self.confirmation_timestamp),
            "points": self.points.to_dict(),
            "quality_score": round(self.quality_score, 4)
        }
        
        if self.ratios:
            pattern_dict["ratios"] = {k: round(v, 3) for k, v in self.ratios.items()}
        
        if self.prz:
            pattern_dict["potential_reversal_zone"] = self.prz.to_dict()
        
        return pattern_dict

# --- Registry (Unchanged but using new config structure) ---

class PatternRegistry:
    _patterns: Dict[str, HarmonicPatternConfig] = {}
    
    @classmethod
    def register(cls, pattern: HarmonicPatternConfig) -> None:
        cls._patterns[pattern.name] = pattern
    
    @classmethod
    def get_enabled(cls, enabled_names: Optional[List[str]] = None) -> List[HarmonicPatternConfig]:
        if enabled_names is None:
            return list(cls._patterns.values())
        return [p for p in cls._patterns.values() if p.name in enabled_names]

# Helper to create config easily
def _create_config(name, xb, ac, bd, xd, cd_xc=None):
    return HarmonicPatternConfig(
        name=name,
        ratios=PatternRatios(
            xb=FibonacciRatio(*xb),
            ac=FibonacciRatio(*ac),
            bd=FibonacciRatio(*bd),
            xd=FibonacciRatio(*xd),
            cd_xc=FibonacciRatio(*cd_xc) if cd_xc else None
        )
    )

# Registering Standard Patterns with Ideal Targets for Scoring
# Tuple format: (Target, Min, Max)
# Updated ranges to allow for detection with error margin logic
PatternRegistry.register(_create_config("Gartley", (0.618, 0.618, 0.618), (0.618, 0.382, 0.886), (1.272, 1.13, 1.618), (0.786, 0.786, 0.786)))
PatternRegistry.register(_create_config("Bat", (0.5, 0.382, 0.5), (0.618, 0.382, 0.886), (2.0, 1.618, 2.618), (0.886, 0.886, 0.886)))
PatternRegistry.register(_create_config("Butterfly", (0.786, 0.786, 0.786), (0.618, 0.382, 0.886), (1.618, 1.618, 2.24), (1.27, 1.27, 1.618)))
PatternRegistry.register(_create_config("Crab", (0.618, 0.382, 0.618), (0.618, 0.382, 0.886), (3.0, 2.24, 3.618), (1.618, 1.618, 1.618)))
PatternRegistry.register(_create_config("Cypher", (0.5, 0.382, 0.618), (1.414, 1.13, 1.414), (0.0, 0.0, 0.0), (0.786, 0.786, 0.786), (1.272, 1.272, 2.0)))
PatternRegistry.register(_create_config("Shark", (0.5, 0.382, 0.618), (1.13, 1.13, 1.618), (1.618, 1.618, 2.24), (1.0, 0.886, 1.13)))

# --- Advanced Pivot Detection ---


@njit(cache=True)
def _detect_pivots_numba(highs, lows, closes, depth):
    n = len(highs)
    
    is_pivot_high = np.full(n, False)
    is_pivot_low = np.full(n, False)
    
    for i in range(depth, n - depth):
        window_high = highs[i-depth : i+depth+1]
        window_low = lows[i-depth : i+depth+1]
        
        if highs[i] == np.max(window_high):
            is_pivot_high[i] = True
        if lows[i] == np.min(window_low):
            is_pivot_low[i] = True

    # Pre-allocate arrays for cleaned pivots (max possible is n)
    cleaned_indices = np.zeros(n, dtype=np.int32)
    cleaned_prices = np.zeros(n, dtype=np.float64)
    cleaned_types = np.zeros(n, dtype=np.int32) # 1: HIGH, -1: LOW
    
    count = 0
    last_type = 0 # 0 means none
    
    # Iterate through potential indices
    for idx in range(n):
        is_h = is_pivot_high[idx]
        is_l = is_pivot_low[idx]
        
        if not is_h and not is_l:
            continue
            
        current_type = 0
        price = 0.0
        
        if is_h and is_l:
            if last_type == 1:
                current_type = -1
                price = lows[idx]
            elif last_type == -1:
                current_type = 1
                price = highs[idx]
            else:
                if abs(highs[idx] - closes[idx]) < abs(lows[idx] - closes[idx]):
                    current_type = 1
                    price = highs[idx]
                else:
                    current_type = -1
                    price = lows[idx]
        elif is_h:
            current_type = 1
            price = highs[idx]
        elif is_l:
            current_type = -1
            price = lows[idx]
            
        if count == 0:
            cleaned_indices[count] = idx
            cleaned_prices[count] = price
            cleaned_types[count] = current_type
            last_type = current_type
            count += 1
            continue
            
        if current_type == last_type:
            # Same type pivot found. Keep the more extreme one.
            last_idx = count - 1
            if current_type == 1: # HIGH
                if price > cleaned_prices[last_idx]:
                    cleaned_prices[last_idx] = price
                    cleaned_indices[last_idx] = idx
            else: # LOW
                if price < cleaned_prices[last_idx]:
                    cleaned_prices[last_idx] = price
                    cleaned_indices[last_idx] = idx
        else:
            # Alternating pivot, add new leg
            cleaned_indices[count] = idx
            cleaned_prices[count] = price
            cleaned_types[count] = current_type
            last_type = current_type
            count += 1
            
    return cleaned_indices[:count], cleaned_prices[:count], cleaned_types[:count]

class SmartPivotDetector:
    """
    Advanced Pivot Detection using Vectorized ZigZag.
    
    WHY IT WAS FAILING BEFORE:
    The previous method used a simple sliding window. Market noise often created 
    false pivots (e.g., a tiny dip in a large uptrend), breaking the X-A-B-C-D 
    topology. This class enforces strictly alternating High/Low swings, merging 
    insignificant noise into larger legs.
    """
    
    def __init__(self, depth: int):
        self.depth = depth

    def detect(self, df: pl.DataFrame) -> List[PivotPoint]:
        """
        Detects pivots using a combination of window extrema and alternating cleanup.
        """
        if len(df) < self.depth * 2:
            return []

        highs = df['high'].to_numpy()
        lows = df['low'].to_numpy()
        n = len(df)

        closes = df['close'].to_numpy()

        indices, prices, types = _detect_pivots_numba(highs, lows, closes, self.depth)

        # Get timestamp column
        if 'timestamp' in df.columns:
            timestamps = df['timestamp'].to_list()
        elif df.columns[0] and hasattr(df, 'index'):
            timestamps = df.index.to_list()
        else:
            timestamps = [datetime.now() for _ in range(n)]

        cleaned_pivots: List[PivotPoint] = []
        for i in range(len(indices)):
            idx = indices[i]
            price = prices[i]
            ptype = PivotType.HIGH if types[i] == 1 else PivotType.LOW
            ts = timestamps[idx] if idx < len(timestamps) else datetime.now()
            # Ensure ts is datetime
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            elif isinstance(ts, (int, float)):
                ts = datetime.fromtimestamp(ts)
            cleaned_pivots.append(PivotPoint(ts, price, ptype, idx))

        return cleaned_pivots

# --- Core Logic ---

class HarmonicPatternDetector:
    """
    Main detector class. 
    Refactored to check patterns efficiently and rank by quality.
    """
    
    def __init__(
        self,
        pivot_depth: int = 5,
        error_margin_percent: float = 15.0,
        max_patterns: int = 5
    ):
        self.pivot_detector = SmartPivotDetector(pivot_depth)
        self.error_margin = error_margin_percent / 100.0
        self.max_patterns = max_patterns
        self.min_bars = pivot_depth * 3

    def detect_patterns(
        self,
        df: pl.DataFrame,
        enabled_patterns: Optional[List[str]] = None
    ) -> Dict:
        result = {"success": False, "patterns": [], "error": None}
        # Eğer pandas DataFrame gelirse polars'a çevir
        if hasattr(df, 'to_numpy') and hasattr(df, 'index'):
            try:
                df = pl.from_pandas(df.reset_index() if hasattr(df, 'reset_index') else df)
            except Exception:
                pass

        if not isinstance(df, pl.DataFrame) or len(df) < self.min_bars:
            result["error"] = "Insufficient data"
            return result

        try:
            # 1. Get Cleaned ZigZag Pivots
            pivots = self.pivot_detector.detect(df)
            
            if len(pivots) < 4:
                result["success"] = True
                return result

            # 2. Get Configs
            configs = PatternRegistry.get_enabled(enabled_patterns)
            found_patterns: List[HarmonicPattern] = []

            # 3. Iterate through pivot sets (Vectorized approach conceptually)
            # We look at sets of 5 (XABCD) and sets of 4 (XABC)
            # Because pivots are strictly alternating, we can iterate sequentially
            
            num_pivots = len(pivots)
            
            # --- Check for Complete Patterns (5 points) ---
            for i in range(num_pivots - 4):
                window = pivots[i : i+5]
                pat = self._check_pattern(window, configs, is_complete=True)
                if pat: found_patterns.append(pat)
            
            # --- Check for Potential Patterns (Last 4 points) ---
            # We only care about potential patterns at the *very end* of the chart
            if num_pivots >= 4:
                last_window = pivots[-4:]
                # Verify the last pivot is actually the last significant swing on chart
                # (Logic handled by ZigZag, assuming last pivot is recent enough)
                pat = self._check_pattern(last_window, configs, is_complete=False)
                if pat: found_patterns.append(pat)

            # 4. Sort and Filter
            # Primary Sort: Complete > Potential
            # Secondary Sort: Confirmation Time (Newest First)
            # Tertiary Sort: Quality Score (Lower is better)
            
            found_patterns.sort(key=lambda p: (
                0 if p.status == PatternStatus.COMPLETE else 1,
                -p.confirmation_timestamp.timestamp(), 
                p.quality_score
            ))

            result["patterns"] = [p.to_dict() for p in found_patterns[:self.max_patterns]]
            result["success"] = True

        except Exception as e:
            logger.exception("Error in Harmonic detection")
            result["error"] = str(e)
            
        return result

    def _check_pattern(
        self, 
        points: List[PivotPoint], 
        configs: List[HarmonicPatternConfig],
        is_complete: bool
    ) -> Optional[HarmonicPattern]:
        
        # Determine Direction based on X and A
        # If X is Low, A is High -> XA is Bullish Leg -> Pattern is usually Bearish or Bullish depending on definition
        # Standard:
        # Bullish Pattern (Buy at D): X(Low) -> A(High) -> B(Low) -> C(High) -> D(Low)
        # Bearish Pattern (Sell at D): X(High) -> A(Low) -> B(High) -> C(Low) -> D(High)
        
        pX, pA, pB, pC = points[0], points[1], points[2], points[3]
        pD = points[4] if is_complete else None
        
        # Fast fail: Check topology consistency (ZigZag guarantees this, but double check)
        # Bullish Topo: L, H, L, H, (L)
        is_bullish_topo = (pX.pivot_type == PivotType.LOW)
        direction = PatternDirection.BULLISH if is_bullish_topo else PatternDirection.BEARISH
        
        # Calculate Segment Lengths
        # Standardize strictly on vertical price distance
        xa_height = abs(pA.price - pX.price)
        ab_height = abs(pB.price - pA.price)
        bc_height = abs(pC.price - pB.price)
        xc_height = abs(pC.price - pX.price)
        
        if xa_height == 0 or ab_height == 0: return None

        # Calculate Ratios
        # XB: AB / XA
        r_xb = ab_height / xa_height
        
        # AC: BC / AB
        r_ac = bc_height / ab_height
        
        r_xd = 0.0
        r_bd = 0.0
        r_cd_xc = 0.0
        
        if is_complete and pD:
            cd_height = abs(pD.price - pC.price)
            
            # BD Ratio: CD / BC (Standard)
            if bc_height > 0: r_bd = cd_height / bc_height
            
            # XD Ratio: 
            # D position relative to X, normalized by XA.
            # Correct Directional Logic:
            # Bullish: X(Low) -> A(High). D should be below A.
            #   Retracement (Gartley): D is between X and A.
            #   Extension (Crab): D is below X.
            #   Formula: (A - D) / XA
            # Bearish: X(High) -> A(Low). D should be above A.
            #   Formula: (D - A) / XA
            
            if xa_height > 0: 
                if direction == PatternDirection.BULLISH:
                     # Check if D is incorrectly above A (impossible for Bullish Buy pattern)
                     if pD.price > pA.price: return None
                     r_xd = (pA.price - pD.price) / xa_height
                else:
                     # Bearish: A is Low. D must be > A
                     if pD.price < pA.price: return None
                     r_xd = (pD.price - pA.price) / xa_height

            # Cypher CD/XC Ratio
            if xc_height > 0: r_cd_xc = cd_height / xc_height
            
        # Match against configs
        best_pattern: Optional[HarmonicPattern] = None
        best_score = float('inf')
        
        current_ratios = {"XB": r_xb, "AC": r_ac, "BD": r_bd, "XD": r_xd, "CD/XC": r_cd_xc}

        for config in configs:
            # Check direction support
            if direction == PatternDirection.BULLISH and not config.supports_bullish: continue
            if direction == PatternDirection.BEARISH and not config.supports_bearish: continue
            
            # Validate Ratios & Calculate Score
            score = 0.0
            valid = True
            count_ratios = 0
            
            # XB Check
            if not config.ratios.xb.validate(r_xb, self.error_margin): continue
            score += config.ratios.xb.error_score(r_xb)
            count_ratios += 1
            
            # AC Check
            if not config.ratios.ac.validate(r_ac, self.error_margin): continue
            score += config.ratios.ac.error_score(r_ac)
            count_ratios += 1
            
            if is_complete:
                # BD Check
                # Cypher usually has ignore-val (0).
                # Only check if target > 0 OR if it's explicitly required
                if config.ratios.bd.target > 0 or config.ratios.bd.min_value > 0:
                     if not config.ratios.bd.validate(r_bd, self.error_margin): continue
                     score += config.ratios.bd.error_score(r_bd)
                     count_ratios += 1
                
                # XD Check
                if config.ratios.xd.target > 0:
                    if not config.ratios.xd.validate(r_xd, self.error_margin): continue
                    score += config.ratios.xd.error_score(r_xd)
                    count_ratios += 1
                
                # Special CD/XC for Cypher
                if config.ratios.cd_xc:
                    if not config.ratios.cd_xc.validate(r_cd_xc, self.error_margin): continue
                    score += config.ratios.cd_xc.error_score(r_cd_xc)
                    count_ratios += 1
            
            # Determine Pattern Points structure
            pp = PatternPoints(pX, pA, pB, pC, pD)
            
            # Determine PRZ if Potential
            prz = None
            if not is_complete:
                prz = self._calculate_prz(pp, config, direction)
            
            # Keep the best match (lowest score) for this set of points
            if score < best_score:
                best_score = score
                status = PatternStatus.COMPLETE if is_complete else PatternStatus.POTENTIAL
                conf_time = pD.timestamp if pD else pC.timestamp
                
                # Filter result ratios
                out_ratios = {k: v for k, v in current_ratios.items() if v > 0}
                
                # Normalized Quality Score (RMSE)
                normalized_score = math.sqrt(score / count_ratios) if count_ratios > 0 else float('inf')
                
                best_pattern = HarmonicPattern(
                    name=config.name,
                    direction=direction,
                    status=status,
                    points=pp,
                    confirmation_timestamp=conf_time,
                    quality_score=normalized_score,
                    ratios=out_ratios,
                    prz=prz
                )
        
        return best_pattern

    def _calculate_prz(self, p: PatternPoints, config: HarmonicPatternConfig, direction: PatternDirection) -> PotentialReversalZone:
        xa = abs(p.a.price - p.x.price)
        bc = abs(p.c.price - p.b.price)
        xc = abs(p.c.price - p.x.price)
        
        # PRZ = Convergence of projections
        
        # 1. XD Projection from A (Not X!)
        # D is at distance (XA * Ratio) from A.
        # Direction:
        # Bullish: A is High. D is Low. D = A - (XA * Ratio)
        # Bearish: A is Low. D is High. D = A + (XA * Ratio)
        
        sign = -1 if direction == PatternDirection.BULLISH else 1
        
        if config.name == "Cypher":
            # Cypher XC projection logic
            # Simplification: Cypher CD leg = XC * 0.786? No.
            # Cypher D is 0.786 of XC. 
            # D = C - (XC * 0.786) for Bullish?
            # Let's use the explicit CD/XC ratio if available, else standard XD logic.
            # Standard Cypher definition: D is 0.786 retracement of XC.
            # So Distance(CD) = 0.786 * Distance(XC).
            # D = C + sign * (XC * 0.786).
            # Note: For Bullish Cypher, C is High? 
            # Bullish Cypher: X(Low), A(High), B(Low), C(High), D(Low).
            # So C is High, D is Low. sign (-1) is correct.
            target = p.c.price + (sign * xc * config.ratios.xd.target)
            return PotentialReversalZone(target, target)

        # Standard XD Projection (from A)
        d_from_a_min = p.a.price + (sign * xa * config.ratios.xd.min_value)
        d_from_a_max = p.a.price + (sign * xa * config.ratios.xd.max_value)
        
        # BC Projection (from C)
        # Bullish: C is High, D is Low. D = C - (BC * Ratio)
        d_from_c_min = p.c.price + (sign * bc * config.ratios.bd.min_value)
        d_from_c_max = p.c.price + (sign * bc * config.ratios.bd.max_value)
        
        vals = [d_from_a_min, d_from_a_max, d_from_c_min, d_from_c_max]
        
        # Sort by proximity to ideal target to find the tightest cluster?
        # Ideal D:
        ideal_d = p.a.price + (sign * xa * config.ratios.xd.target)
        vals.sort(key=lambda x: abs(x - ideal_d))
        
        return PotentialReversalZone(*sorted(vals[:2]))

# --- Public API (Backwards Compatible) ---

def calculate_harmonic_patterns(
    df,
    pivot_depth: int = 5,
    error_margin_percent: float = 15.0,
    enabled_patterns: Optional[List[str]] = None,
    max_patterns_to_return: int = 5
) -> Dict:
    global pl  # Polars modülünün global olduğunu belirt
    
    # Pandas DataFrame kontrolü ve Polars'a dönüşümü
    if hasattr(df, 'to_numpy') and hasattr(df, 'index'):
        try:
            df = pl.from_pandas(df.reset_index() if hasattr(df, 'reset_index') else df)
        except Exception:
            pass

    if not isinstance(df, pl.DataFrame):
        return {"success": False, "patterns": [], "error": "Data must be a Polars DataFrame"}

    detector = HarmonicPatternDetector(
        pivot_depth=pivot_depth,
        error_margin_percent=error_margin_percent,
        max_patterns=max_patterns_to_return
    )

    return detector.detect_patterns(df, enabled_patterns)