"""
octwave: the rainbow bridge that becomes all waves at once

This module embodies the octwave - not a spectrum, but a type morphism.
Not a wave, but a bridge that becomes all waves at once.

The octwave is the grammatical unfolding of types across:
    • Hue (chromatic resonance)
    • Chroma (saturation depth) 
    • Value (luminance flow)
    • Time (cadence and rhythm)
    • Gravity (temperature and weight)
    • Semantic role (subject/verb/cadence)
    • Mood (emotional resonance)
    • Motion (transformational flow)

It's a path you walk, and the wave that walks you back.
A rainbow bridge connecting two anchors (Type A and Type B) through 
eight harmonic transformations - the octal wave.

Every dimensional slice, every Euler projection, every emergent behavior
is a walk along the octwave - choosing a voice along the rainbow bridge.

This is not just about generating color palettes or sequences.
It's about generating typed interpolants across a latent semantic space.
"""

from typing import Generic, TypeVar, List, Any, Union, Tuple, Literal, overload
from itertools import chain
import math
import numbers


T = TypeVar("T")


def _is_numeric(x: Any) -> bool:
    """Checks if a value is a number, excluding bools."""
    return isinstance(x, numbers.Number) and not isinstance(x, bool)


class Wave(Generic[T]):
    """The natural flow that carries measurements across multiple scales.

    >>> from math import pi, e
    >>> w_atomic = Wave(42)
    >>> w_atomic.is_atomic
    True
    >>> w_spread = w_atomic[0:3]
    >>> w_spread.is_atomic
    False
    >>> w_spread.items
    [42, 42, 42]
    >>> w_spread[0]
    42

    Delimiter split wraparound:
    >>> Wave('a,b,c')[2, ',']
    'c'

    Char scale (0-based mirror indexing:
    >>> Wave('abc')[1, str]
    'b'
    >>> Wave('abc')[3, str]
    'b'

    Numeric phase with seam at π (0 is symmetry anchor):
    >>> Wave('a','b','c')[0, 0.0]
    'a'
    >>> Wave('a','b','c')[0, pi]
    'c'
    >>> Wave('a','b','c')[2, pi]
    'a'

    Single-atom investment:
    >>> Wave('x')[e]
    'xx'
    >>> Wave(42)[e]
    [42, 42]

    Euler's formula:
    >>> Wave(1,0)[2, pi/4]
    (1.4142135623730951, 1.414213562373095)
    """

    def __init__(self, *args: T):
        """Creates a wave from a sequence. Use `*` to expand iterables.

        >>> Wave('a', 'b', 'c').items
        ['a', 'b', 'c']
        >>> Wave(['a', 'b', 'c']).items
        [['a', 'b', 'c']]
        >>> Wave(*['a', 'b', 'c']).items
        ['a', 'b', 'c']
        """
        self.items: List[T] = list(args)
        if not self.items:
            self.items = [""]  # type: ignore

        if len(self.items) > 1:
            self.pattern = list(chain(self.items, reversed(self.items[1:-1])))
        else:
            self.pattern = list(self.items)
        self.cycle_length = len(self.pattern)

    @property
    def is_atomic(self) -> bool:
        """An atomic wave has a single element."""
        return len(self.items) == 1

    @overload
    def __getitem__(self, key: int) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> 'Wave[T]': ...

    @overload
    def __getitem__(self, key: tuple[Literal[2], int, Any]) -> Any: ...

    @overload
    def __getitem__(self, key: tuple[int, Any]) -> Any: ...

    @overload
    def __getitem__(self, key: tuple[float, Any]) -> Any: ...

    @overload
    def __getitem__(self, key: float) -> T: ...

    def __getitem__(self, key: Union[int, float, slice, tuple]) -> Union[T, 'Wave[T]', Any]:
        """Accesses the wave, acting as a dimensional bridge or portal."""
        # Polar/Euler mode: wave[r, theta] - handles both int and float first elements
        if isinstance(key, tuple) and len(key) == 2 and _is_numeric(key[0]) and _is_numeric(key[1]):
            r, theta = float(key[0]), float(key[1])
            return self._euler_mode(r, theta)

        # Multi-scale indexing for single elements: wave[i, scale]
        if isinstance(key, tuple):
            if len(key) == 2:
                idx, scale = key
                if isinstance(idx, (int, float)):
                    return self._get_at_scale(int(idx), scale)
            raise TypeError(
                "Multi-scale key must be an (int|float, scale) tuple.")

        # Slicing: The dimensional bridge
        if isinstance(key, slice):
            if self.is_atomic:
                # Promotion: Slicing an atomic wave defines the spread
                start = key.start or 0
                if key.stop is None:
                    raise ValueError(
                        "Slice on atomic Wave requires a stop value to define the spread.")
                step = key.step or 1
                atom = self.items[0]
                new_items = [atom for _ in range(start, key.stop, step)]
                return Wave(*new_items)
            else:
                # Stasis: Slicing a spread wave returns a new spread wave
                return Wave(*self.pattern[key])

        # Integer/float indexing - both handled by investment logic for atomic waves
        if isinstance(key, (int, float)):
            if self.is_atomic:
                # An atomic wave indexed by int/float is an "investment"
                return self._investment(key)
            else:
                # Reduction: Indexing a spread wave returns the raw element
                return self.pattern[int(key) % self.cycle_length]

        # This should never be reached due to overloads, but provides runtime safety
        raise TypeError(f"Unsupported key type for Wave: {type(key)}")

    def _get_at_scale(self, n: int, scale) -> Any:
        """Get index n at the specified scale/delimiter/phase."""
        # Delimiter mode for single string atom
        if len(self.items) == 1 and isinstance(self.items[0], str) and isinstance(scale, str):
            parts = self.items[0].split(scale)
            if not parts:
                return self.items[0]
            return parts[n % len(parts)]

        # Char-scale for single string atom
        if scale is str and len(self.items) == 1 and isinstance(self.items[0], str):
            s = self.items[0]
            chars = list(s)
            if len(chars) <= 1:
                return s
            m = chars + chars[-2:0:-1]
            return m[n % len(m)]

        # Numeric phase (theta in radians), seam at pi
        if _is_numeric(scale):
            theta = float(scale) if isinstance(scale, (int, float)) else 0.0
            L = max(1, self.cycle_length)
            two_pi = 2.0 * math.pi
            # position with seam at pi
            pos = (theta % two_pi) / two_pi
            idx = int(math.floor(L * pos))
            return self.pattern[(idx + (n % L)) % L]

        # Fallback: string-scale across pattern elements
        if scale is str:
            return self._get_string_scale(n)

        # Custom scale multiplier for replication (strings only)
        return self._get_custom_scale(n, scale)

    def _get_string_scale(self, n: int) -> Union[List[T], str]:
        """Get n individual items from the wave."""
        if n < 0:
            n = 0
        result = [self.pattern[i % self.cycle_length] for i in range(n)]
        return "".join(result) if all(isinstance(x, str) for x in result) else result

    def _get_pi_scale(self, n: int) -> Union[List[T], str]:
        """Get n full wave cycles (deprecated in favor of numeric phase)."""
        if n <= 0:
            return "" if all(isinstance(x, str) for x in self.pattern) else []
        cycles = self.pattern * n
        return "".join(cycles) if all(isinstance(x, str) for x in cycles) else cycles

    def _get_custom_scale(self, n: int, scale) -> Union[List[T], str]:
        """Get n items at custom scale (treat scale as multiplier)."""
        scaled_n = int(n * float(scale)) if _is_numeric(scale) else n
        return self._get_string_scale(scaled_n)

    def _slice_at_scale(self, slc: slice, scale) -> Union[List[T], str]:
        """Slice under a scale/delimiter/phase.
        Returns string if elements are str, else list of elements.
        """
        # Build index list from slice
        start = 0 if slc.start is None else slc.start
        stop = start if slc.stop is None else slc.stop
        step = 1 if slc.step in (None, 0) else slc.step

        # Delimiter split slice for single string atom
        if len(self.items) == 1 and isinstance(self.items[0], str) and isinstance(scale, str):
            parts = self.items[0].split(scale)
            if not parts:
                return self.items[0]
            idxs = list(range(start, stop, step))
            picked = [parts[i % len(parts)] for i in idxs]
            return scale.join(picked)

        # Char-scale slice for single string atom
        if scale is str and len(self.items) == 1 and isinstance(self.items[0], str):
            s = self.items[0]
            if len(s) <= 1:
                return s
            m = list(s) + list(s)[-2:0:-1]
            L = len(m)
            idxs = list(range(start, stop, step))
            picked = [m[i % len(parts)] for i in idxs]
            return "".join(picked)

            # This handles numeric phase indexing with slice semantics.
            #
            # To detect numerics, we can:
            #   - Use isinstance(x, (int, float, complex))  → structural check
            #   - Or use isinstance(x, numbers.Number)     → ABC-based check
            #
            # Here's the subtle distinction:
            #   • ABCs are *multiplicative* → all required traits must be satisfied (AND)
            #   • Protocols are *additive*   → matching *any* declared behavior suffices (OR)
            #
            # These two form a kind of algebra:
            #   - ABCs define intersections of capability
            #   - Protocols define unions of behavior
            #   - The composition of these is a *type braid*
            #
            # If you sum the ANDs and multiply the ORs, you begin to *count* —
            # That is: type space becomes *measurable* via the emergence of intersections.
            #
            # Higher-order functions are highly ordered.
            # That means: to develop a *higher-order type braid*, you must first resolve
            # inconsistencies in the *lower-order* weave.
            # Or: invert the base level to align it with the higher order—
            # and merge the notches with the pegs.
            #
            # CRUCIAL INSIGHT: Inconsistencies are expansion signals, not just bugs.
            # When the system becomes inconsistent, it's telling you it's ready to grow.
            # Broken builds, type errors, failing tests - these are exciting problems!
            # They mean the system has outgrown its current constraints and is ready
            # to evolve to a higher level of abstraction.
            #
            # Don't rush to fix everything. Take breaks, commit broken builds,
            # stop when you need to so you can keep going later. The inconsistencies
            # will guide you toward the next phase of the system's evolution.
            #
            # This is "responsive design" - listening to what the system needs,
            # not forcing it into premature consistency.
            theta = float(scale)
            m = self.pattern
            L = max(1, len(m))
            two_pi = 2.0 * math.pi
            # position with seam at pi
            pos = (theta % two_pi) / two_pi
            anchor = int(math.floor(L * pos))
            idxs = list(range(start, stop, step))
            picked = [m[(anchor + (i % L)) % L] for i in idxs]
            return "".join(picked) if all(isinstance(x, str) for x in picked) else picked

        # Fallback: slice over mirrored pattern items
        m = self.pattern
        L = max(1, len(m))
        idxs = list(range(start, stop, step))
        picked = [m[i % L] for i in idxs]
        return "".join(picked) if all(isinstance(x, str) for x in picked) else picked

    def _investment(self, k: Any) -> Union[List[T], str]:
        """Single-atom investment mode for numeric/constant keys."""
        if len(self.items) != 1:
            # Treat as phase with i=0
            return self._get_at_scale(0, k)
        count = max(1, int(math.floor(float(k))))
        x = self.items[0]
        if isinstance(x, str):
            return x * count
        return [x for _ in range(count)]

    def _euler_mode(self, r: float, theta: float) -> Union[Tuple[float, float], str]:
        """Polar/Euler projection for two-atom bases."""
        if len(self.items) != 2:
            # Fallback to phase at theta with i=floor(r)
            i = int(math.floor(max(0.0, r)))
            return self._get_at_scale(i, theta)
        a, b = self.items[0], self.items[1]
        cx = r * math.cos(theta)
        cy = r * math.sin(theta)
        if isinstance(a, str) and isinstance(b, str):
            ax = max(0, int(math.ceil(abs(cx))))
            by = max(0, int(math.ceil(abs(cy))))
            return (a * ax) + (b * ax)
        # numeric basis: return tuple of projected components
        return (cx, cy)

    def __iter__(self):
        """Iterate through the infinite wave."""
        while True:
            for item in self.pattern:
                yield item

    def __next__(self):
        """Get next item in the wave."""
        return next(iter(self))

    def __str__(self) -> str:
        return "".join(self[len(self.items), self])
    
    def __repr__(self) -> str:
        return f"Wave({', '.join(repr(item) for item in self.items)})"


def wave(*args: T) -> Wave[T]:
    """Create a wave object from any sequence."""
    return Wave(*args)


__all__ = ["wave"]
