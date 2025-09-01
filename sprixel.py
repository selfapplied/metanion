"""
sprixel: infinite mirrors and adaptive signals

This module turns finite motifs into infinite, adaptive signals. It offers:
- mirror: reflect a finite motif into an endless symmetric stream
- signal: adaptively scale a motif to any length, with optional morphing
- parametric: evolve parameters across a signal via a function of position
- layout / responsive: simple responsive composition for text UI

Names favor clarity and flow; functions compose like light operators.
"""

import doctest
from itertools import chain, islice
from typing import Iterable, Callable, Optional, TypeVar, List, Any, Union, Tuple, Protocol, Generic, SupportsFloat, SupportsInt, Literal, overload
import math
from math import pi, e
import random
import hashlib
import numbers

# Import Wave from the new octwave module
from octwave import Wave, wave

T = TypeVar("T")
U = TypeVar("U")


def is_number(x: Any) -> bool:
    """Checks if a value is a number, excluding bools."""
    return isinstance(x, numbers.Number) and not isinstance(x, bool)

class Parametric[U]:
    """Protocol for functions that transform position ratios into parameter space."""

    def __call__(self, ratio: float) -> U: ...


def signal(seq: Iterable[T], morph: Optional[Callable[[T, int, int], Any]] = None) -> Callable[[int], List[Any]]:
    """
    Carrier wave that transports measurements through time.
    Takes a wave flow and creates a time-based transport system.

    The returned callable transports n measurements through time.
    If morph is provided, measurements are transformed during transport.

    Example:
        head = signal(["╔", "╦", "╗"])  # Create time transport
        head(5)  # Transport 5 measurements through time
        body = signal(["╠", "╬", "╣"])  # Another time transport
        body(7)  # Transport 7 measurements through time

        def scale(glyph: str, i: int, n: int) -> str:
            s = 1.0 + (i / max(1, n)) * 0.5
            return f"{glyph}×{s:.1f}"

        scaled = signal(["╔", "╦", "╗"], scale)
        scaled(5)
    """
    # Unpack the sequence into individual arguments for wave
    loop = wave(*seq)

    def at(n: int, *args: Any, **kw: Any) -> List[Any]:
        view = list(islice(loop, n))
        if morph:
            return [morph(x, i, n, *args, **kw) for i, x in enumerate(view)]
        return view

    return at


def parametric(base: Iterable[T], phi: Parametric[U],
               emit: Optional[Callable[[T, float, int, U], Any]] = None) -> Callable[[int], List[Union[T, Tuple[T, U]]]]:
    """
    Evolve parameters across a mirrored signal using a function of position.

    phi receives a ratio r in [0,1] and returns a parameter value in U-space.
    emit can post-process each item as emit(item, r, n, p). If emit is not
    provided, (item, p) pairs are always produced.

    Example:
        wave = parametric(["A", "B", "C"], lambda r: math.sin(2*math.pi*r))
        wave(8)  # -> [("A", s0), ("B", s1), ...]

        def paint(ch: str, r: float, n: int, p: float) -> str:
            return f"{ch}({p:.2f})"

        bright = parametric(["X", "Y", "Z"], lambda r: math.exp(2*r)-1, emit=paint)
        bright(6)
    """
    loop = wave(*base)

    def at(n: int) -> List[Any]:
        out: List[Any] = []
        for i in range(n):
            x = next(loop)
            if n <= 1:
                r = 0.0
            else:
                r = i / (n - 1)
            p = phi(r)
            if emit:
                out.append(emit(x, r, n, p))
            else:
                out.append((x, p))
        return out

    return at


def layout(elems: Iterable[str], width: int, spacing: Optional[Callable[[int, int], int]] = None, unit: int = 10) -> str:
    """
    Arranges measurements in a space.
    Takes transported measurements and arranges them spatially.

    - width: spatial width to arrange measurements within
    - unit: approximate measurement size for spacing calculations
    - spacing: function that determines gaps between measurements

    Creates spatial composition from time-transported measurements.
    """
    need = max(1, width // max(1, unit))
    row = signal(elems)(need)
    if spacing:
        gap = max(0, spacing(width, need))
        joiner = " " * gap
        return joiner.join(row)
    return "".join(row)


def responsive(elems: Iterable[str], spacing: Optional[Callable[[int, int], int]] = None, unit: int = 10) -> Callable[[int], str]:
    """
    Return a layout function width -> string, using the same rules as layout.

    Example:
        menu = responsive(["Home", "Products", "About", "Contact"],
                          spacing=lambda w, k: max(1, (w - k*10)//max(1, k-1)))
        menu(80)
    """
    def at(width: int) -> str:
        return layout(elems, width, spacing=spacing, unit=unit)

    return at


# --- Color and flourish ----------------------------------------------------

def hue(t: float) -> tuple[int, int, int]:
    """
    Map t in [0,1] around the color wheel to an RGB triplet (0-255).
    Full saturation and value for vivid output.
    """
    a = (t % 1.0) * 6.0
    k0 = int(a) % 6
    f = a - int(a)
    q = int(255 * (1 - f))
    u = int(255 * f)
    if k0 == 0:
        return (255, u, 0)
    if k0 == 1:
        return (q, 255, 0)
    if k0 == 2:
        return (0, 255, u)
    if k0 == 3:
        return (0, q, 255)
    if k0 == 4:
        return (u, 0, 255)
    return (255, 0, q)


def dye(text: str, rgb: tuple[int, int, int]) -> str:
    """Wrap text with 24-bit ANSI foreground color."""
    r, g, b = rgb
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def fromhex(code: str) -> tuple[int, int, int]:
    code = code.strip().lstrip('#')
    if len(code) == 3:
        code = ''.join([c*2 for c in code])
    r = int(code[0:2], 16)
    g = int(code[2:4], 16)
    b = int(code[4:6], 16)
    return (r, g, b)


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    ar, ag, ab = a
    br, bg, bb = b
    return (
        int(ar + (br - ar) * t),
        int(ag + (bg - ag) * t),
        int(ab + (bb - ab) * t),
    )


def ramp(stops: list[tuple[float, tuple[int, int, int]]]) -> Callable[[float], tuple[int, int, int]]:
    """
    Build a piecewise-linear gradient palette from (pos, rgb) stops.
    pos in [0,1].
    """
    pts = sorted(stops, key=lambda p: p[0])
    def at(t: float) -> tuple[int, int, int]:
        if t <= pts[0][0]:
            return pts[0][1]
        if t >= pts[-1][0]:
            return pts[-1][1]
        for (p0, c0), (p1, c1) in zip(pts, pts[1:]):
            if p0 <= t <= p1:
                if p1 == p0:
                    return c1
                u = (t - p0) / (p1 - p0)
                return blend(c0, c1, u)
        return pts[-1][1]
    return at


# Preset palettes

dusk = ramp([
    (0.00, fromhex('#1b1f3b')),
    (0.35, fromhex('#53354a')),
    (0.65, fromhex('#903749')),
    (1.00, fromhex('#e84545')),
])
dusk.__name__ = 'dusk'  # friendlier repr
setattr(dusk, 'label', 'dusk')

neon = ramp([
    (0.00, fromhex('#00ffcc')),
    (0.40, fromhex('#00ccff')),
    (0.70, fromhex('#cc00ff')),
    (1.00, fromhex('#ff00cc')),
])
neon.__name__ = 'neon'
setattr(neon, 'label', 'neon')

sea = ramp([
    (0.00, fromhex('#0b132b')),
    (0.30, fromhex('#1c2541')),
    (0.60, fromhex('#3a506b')),
    (1.00, fromhex('#5bc0be')),
])
sea.__name__ = 'sea'
setattr(sea, 'label', 'sea')

earth = ramp([
    (0.00, fromhex('#3b2f2f')),
    (0.30, fromhex('#7f4f24')),
    (0.60, fromhex('#a68a64')),
    (1.00, fromhex('#cdb4ab')),
])
earth.__name__ = 'earth'
setattr(earth, 'label', 'earth')

fire = ramp([
    (0.00, fromhex('#1a1a1a')),
    (0.25, fromhex('#8b0000')),
    (0.55, fromhex('#ff4500')),
    (0.85, fromhex('#ffd700')),
    (1.00, fromhex('#ffffe0')),
])
fire.__name__ = 'fire'
setattr(fire, 'label', 'fire')

ice = ramp([
    (0.00, fromhex('#001219')),
    (0.35, fromhex('#005f73')),
    (0.65, fromhex('#94d2bd')),
    (1.00, fromhex('#e9d8a6')),
])
ice.__name__ = 'ice'
setattr(ice, 'label', 'ice')

pastel = ramp([
    (0.00, fromhex('#f3c4fb')),
    (0.33, fromhex('#b9fbc0')),
    (0.66, fromhex('#a0c4ff')),
    (1.00, fromhex('#ffc6ff')),
])
pastel.__name__ = 'pastel'
setattr(pastel, 'label', 'pastel')

mono = ramp([
    (0.00, fromhex('#111111')),
    (1.00, fromhex('#eeeeee')),
])
mono.__name__ = 'mono'
setattr(mono, 'label', 'mono')

solar = ramp([
    (0.00, fromhex('#002b36')),
    (0.25, fromhex('#073642')),
    (0.50, fromhex('#586e75')),
    (0.75, fromhex('#93a1a1')),
    (1.00, fromhex('#eee8d5')),
])
solar.__name__ = 'solar'
setattr(solar, 'label', 'solar')


# Thermal gravity color system (alternative palette generation)
def thermal_palette(base_color: tuple[int, int, int], temp_vector: list[float], 
                    stops: Optional[list[float]] = None) -> Callable[[float], tuple[int, int, int]]:
    """
    Generate palette from thermal gravity vector.
    base_color: anchor color (usually coldest)
    temp_vector: temperature shifts [+0.4, +0.3, +0.8] for 3 transitions
    stops: optional position stops, defaults to even distribution
    
    Example thermal vectors:
    - [+0.4, +0.3, +0.8]: "hot escape from cold origins" (like dusk)
    - [+0.0, -0.1, +0.0]: "cool drift" (like pastel)
    - [+0.6, +0.8, +0.9, +1.0]: "rising inferno" (like fire)
    
    >>> # Test dusk palette analysis
    >>> dusk_pal = thermal_palette(fromhex('#1b1f3b'), [+0.4, +0.3, +0.8])
    >>> analysis = dusk_pal.thermal_analysis
    >>> analysis.semantic_description
    'Thermal Escape'
    >>> analysis.gravity_direction
    'neutral'
    """
    if stops is None:
        stops = [i / max(1, len(temp_vector)) for i in range(len(temp_vector) + 1)]
    
    # Thermal color mapping using existing palette colors for consistency
    def temp_to_color(temp: float) -> tuple[int, int, int]:
        if temp <= -0.5:  # Cold: deep blues
            t = (temp + 1.0) / 0.5  # [-1, -0.5] -> [0, 1]
            return blend(fromhex('#0b132b'), fromhex('#1c2541'), t)
        elif temp <= 0.0:  # Cool-neutral: blues to purples
            t = (temp + 0.5) / 0.5  # [-0.5, 0] -> [0, 1]
            return blend(fromhex('#1c2541'), fromhex('#53354a'), t)
        elif temp <= 0.5:  # Warm-neutral: purples to burgundies
            t = temp / 0.5  # [0, 0.5] -> [0, 1]
            return blend(fromhex('#53354a'), fromhex('#903749'), t)
        else:  # Hot: burgundies to reds
            t = (temp - 0.5) / 0.5  # [0.5, 1] -> [0, 1]
            return blend(fromhex('#903749'), fromhex('#e84545'), t)
    
    # Build color stops from thermal vector
    color_stops = [(stops[0], base_color)]
    for i, temp_shift in enumerate(temp_vector):
        color = temp_to_color(temp_shift)
        color_stops.append((stops[i + 1], color))
    
    palette_func = ramp(color_stops)
    
    # Attach thermal analysis to the palette function
    from collections import namedtuple
    ThermalAnalysis = namedtuple('ThermalAnalysis', [
        'temp_vector', 'temp_range', 'thermal_acceleration', 
        'energy_release', 'gravity_direction', 'semantic_description', 'stops'
    ])
    
    def heat_accel(temps: list[float], positions: list[float]) -> float:
        """Compute how quickly temperature changes across the palette"""
        if len(temps) < 2:
            return 0.0
        n = len(temps)
        sum_x = sum(positions[1:])
        sum_y = sum(temps)
        sum_xy = sum(x * y for x, y in zip(positions[1:], temps))
        sum_x2 = sum(x * x for x in positions[1:])
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        return (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    
    def energy_release(temps: list[float]) -> float:
        """Compute the total energy change across the thermal vector"""
        if len(temps) < 2:
            return 0.0
        return sum(abs(temps[i] - temps[i-1]) for i in range(1, len(temps)))
    
    def gravity_dir(temps: list[float]) -> str:
        """Determine the overall gravitational direction"""
        if not temps:
            return "neutral"
        avg_temp = sum(temps) / len(temps)
        if avg_temp > 0.3:
            return "heatward"
        elif avg_temp < -0.3:
            return "coldward"
        else:
            return "neutral"
    
    def describe_heat(temps: list[float]) -> str:
        """Generate a semantic description of the thermal pattern"""
        if len(temps) < 2:
            return "Single Temperature"
        if all(t > 0.3 for t in temps) and temps[-1] > temps[0]:
            return "Rising Inferno"
        elif all(t > 0.2 for t in temps) and max(temps) - min(temps) < 0.4:
            return "Sustained Warmth"
        elif all(t < -0.2 for t in temps) and temps[-1] < temps[0]:
            return "Cooling Spiral"
        elif temps[-1] - temps[0] > 1.0:
            return "Thermal Escape"
        elif abs(temps[0]) > 0.8:
            return "Gravitational Anchor"
        temp_range = max(temps) - min(temps)
        if temp_range > 1.0:
            return "Thermal Drama"
        elif temp_range > 0.5:
            return "Moderate Shift"
        else:
            return "Subtle Variation"
    
    # Create and attach the analysis
    temp_range = max(temp_vector) - min(temp_vector)
    temp_acceleration = heat_accel(temp_vector, stops)
    energy_total = energy_release(temp_vector)
    gravity_direction = gravity_dir(temp_vector)
    semantic_description = describe_heat(temp_vector)
    
    try:
        setattr(palette_func, 'thermal_analysis', ThermalAnalysis(
        temp_vector=temp_vector,
        temp_range=temp_range,
        thermal_acceleration=temp_acceleration,
            energy_release=energy_total,
        gravity_direction=gravity_direction,
        semantic_description=semantic_description,
        stops=stops
        ))
    except Exception:
        pass
    
    return palette_func


# Example: dusk could be recreated as:
# dusk_thermal = thermal_palette(fromhex('#1b1f3b'), [+0.4, +0.3, +0.8], [0.0, 0.35, 0.65, 1.0])


def splash(items: Iterable[str] | str, pal: Callable[[float], tuple[int, int, int]], tone: Optional[Callable[[float], float]] = None) -> List[str]:
    """Colorize using a specific palette function pal(r)."""
    return wash(items, tone=tone, pal=pal)


def rainbow(seq: Iterable[str], tone: Optional[Callable[[float], float]] = None) -> Callable[[int], List[str]]:
    """
    Paint a signal with a rainbow gradient across its length.

    tone, if provided, modulates hue by returning a delta added to r.
    """
    base = signal(seq)

    def shade(n: int) -> List[str]:
        out: List[str] = []
        items = base(n)
        for i, x in enumerate(items):
            if n <= 1:
                r = 0.0
            else:
                r = i / (n - 1)
            if tone:
                r = (r + tone(r)) % 1.0
            out.append(dye(str(x), hue(r)))
        return out

    return shade


def wash(items: Iterable[str], tone: Optional[Callable[[float], float]] = None, pal: Optional[Callable[[float], tuple[int, int, int]]] = None) -> List[str]:
    """
    Colorize an existing list or string with a smooth gradient.
    By default uses rainbow; pass pal to use a custom palette(r -> rgb).
    """
    if isinstance(items, str):
        buf = list(items)
    else:
        buf = list(items)
    n = len(buf)
    out: List[str] = []
    for i, x in enumerate(buf):
        if n <= 1:
            r = 0.0
        else:
            r = i / (n - 1)
        if tone:
            r = (r + tone(r)) % 1.0
        rgb = (pal(r) if pal else hue(r))
        out.append(dye(str(x), rgb))
    return out


def showcase(width: Optional[int] = None) -> str:
    """
    Return a compact multi-line demo showcasing sprixel's adaptive color flow.
    """
    try:
        if width is None:
            import shutil
            width = shutil.get_terminal_size().columns
    except Exception:
        width = 80 if width is None else width

    w = max(20, int(width))

    # Line 1: solid ribbon with rainbow wash
    ribbon = rainbow(["█"])(w)

    # Line 2: interference waves painted over diamonds
    def tone(r: float) -> float:
        # gentle interference of two sines
        a = 0.15 * math.sin(2 * math.pi * r * 3.0)
        b = 0.10 * math.sin(2 * math.pi * r * 5.0 + math.pi / 3)
        return a + b

    weave = rainbow(["◆", "◇"], tone)
    band = weave(w)

    # Line 3: responsive menu with gradient accents
    menu = responsive(["Home", "Products", "About", "Contact"],
                      spacing=lambda W, k: max(1, (W - k * 10) // max(1, k - 1)))
    menu_text = menu(w)
    menu_wash = "".join(rainbow(menu_text)(len(menu_text)))

    # Line 4: framing with mirrored corners
    frame = signal(["╔", "═", "╗"])(w)
    floor = signal(["╚", "═", "╝"])(w)
    top = "".join(rainbow(frame)(len(frame)))
    bot = "".join(rainbow(floor)(len(floor)))

    return "\n".join([
        "".join(ribbon),
        "".join(band),
        menu_wash,
        top,
        bot,
    ])


# --- Gates and responsive templates ----------------------------------------

def switch(bands: Iterable[tuple[int, T]]) -> Callable[[int], T]:
    """
    Choose a value by width using threshold bands.

    bands: iterable of (threshold, value) pairs. The pair with the highest
    threshold <= width is selected. If none are <= width, the smallest
    threshold's value is returned.
    """
    table = sorted(bands, key=lambda p: p[0])

    def pick(width: int) -> T:
        chosen: Optional[T] = None
        best = -10**12
        for th, val in table:
            if width >= th and th >= best:
                chosen = val
                best = th
        if chosen is None:
            return table[0][1]
        return chosen

    return pick


def forms(*bands: tuple[int, Callable[[int], str]]) -> Callable[[int], Callable[[int], str]]:
    """
    Build a gate of templates: returns a template function given width.
    Each band is (threshold, template), where template is width -> string.
    """
    pick = switch(bands)

    def at(width: int) -> Callable[[int], str]:
        return pick(width)

    return at


def center_text(text: str, n: int) -> str:
    if n <= 0:
        return ""
    s = text[:n]
    pad = max(0, n - len(s))
    left = pad // 2
    right = pad - left
    return (" " * left) + s + (" " * right)


def pad_text(text: str, n: int) -> str:
    s = text[:n]
    return s + (" " * max(0, n - len(s)))


def dialog(title: str, body: Iterable[str], gate: Optional[Callable[[int], Callable[[int], str]]] = None) -> Callable[[int], str]:
    """
    Build a responsive dialog. Returns width -> multi-line string.

    Provide a gate of templates via forms(...) to customize at breakpoints.
    If gate is None, a default three-form dialog is used (narrow/mid/wide).
    """
    lines = list(body)

    def ascii_box(width: int) -> str:
        inner = max(10, width - 4)
        top = "+" + ("-" * inner) + "+"
        cap = "|" + center_text(title, inner) + "|"
        mid = ["|" + pad_text(x, inner) + "|" for x in lines]
        bot = "+" + ("-" * inner) + "+"
        return "\n".join([top, cap] + mid + [bot])

    def light_box(width: int) -> str:
        inner = max(12, width - 4)
        top = "┌" + ("─" * inner) + "┐"
        cap = "│" + center_text(title, inner) + "│"
        mid = ["│" + pad_text(x, inner) + "│" for x in lines]
        bot = "└" + ("─" * inner) + "┘"
        return "\n".join([top, cap] + mid + [bot])

    def bold_box(width: int) -> str:
        inner = max(16, width - 6)
        top = "╔" + ("═" * inner) + "╗"
        cap = "║" + center_text(title, inner) + "║"
        mid = ["║" + pad_text(x, inner) + "║" for x in lines]
        bot = "╚" + ("═" * inner) + "╝"
        return "\n".join([top, cap] + mid + [bot])

    if gate is None:
        gate = forms(
            (0, ascii_box),
            (40, light_box),
            (70, bold_box),
        )

    def show(width: int) -> str:
        form = gate(width)
        art = form(width)
        return art

    return show


def wrap_lines(text: str, width: int) -> list[str]:
    if width <= 0:
        return []
    words = text.split()
    out: list[str] = []
    line: list[str] = []
    n = 0
    for w in words:
        need = len(w) + (1 if n > 0 else 0)
        if n + need > width:
            out.append(" ".join(line))
            line = [w]
            n = len(w)
        else:
            if n > 0:
                line.append(w)
                n += 1 + len(w)
            else:
                line = [w]
                n = len(w)
    if line:
        out.append(" ".join(line))
    return out


def columns(left: Iterable[str], right: Iterable[str], gap: int = 3) -> Callable[[int], str]:
    """
    Return a renderer width -> string that arranges two columns side by side
    when space permits; otherwise stacks them vertically.

    - gap: spaces between columns
    - The available width is split: left grows a little more than right.
    """
    L = list(left)
    R = list(right)

    def render(width: int) -> str:
        if width < 40:
            # Stack for narrow
            return "\n".join(L + [""] + R)

        inner = width
        g = max(1, gap)
        lw = (inner - g) * 3 // 5
        rw = inner - g - lw

        lw = max(10, lw)
        rw = max(10, rw)

        Lw = [ln for x in L for ln in wrap_lines(x, lw)]
        Rw = [ln for x in R for ln in wrap_lines(x, rw)]

        h = max(len(Lw), len(Rw))
        Lw += [""] * (h - len(Lw))
        Rw += [""] * (h - len(Rw))

        out = []
        for a, b in zip(Lw, Rw):
            out.append(pad_text(a, lw) + (" " * g) + pad_text(b, rw))
        return "\n".join(out)

    return render


# --- Soft tones and reflections --------------------------------------------

def fog(level: float) -> tuple[int, int, int]:
    """Gray tone by level in [0,1]."""
    q = max(0, min(255, int(255 * max(0.0, min(1.0, level)))))
    return (q, q, q)


def shade(text: str, level: float) -> str:
    """Color text in a muted gray level."""
    return dye(text, fog(level))


def reflect(text: Iterable[str] | str, depth: int = 8, drift: int = 1, quiet: float = 0.6) -> str:
    """
    Mirror text beneath itself with muted fading tones.

    - depth: number of reflected rows
    - drift: spaces added per reflected row for a soft perspective
    - quiet: overall dimming multiplier for the reflection (0..1)
    """
    if isinstance(text, str):
        lines = text.splitlines() or [text]
    else:
        lines = list(text)

    out: list[str] = []
    out.extend(lines)

    base = "\n".join(lines)
    # Use the concatenated base line for reflection to maintain width
    base_line = base.split("\n")[-1] if base else ""

    for i in range(1, depth + 1):
        t = 1.0 - (i / (depth + 1))
        lvl = max(0.0, min(1.0, t * quiet))
        offset = " " * (i * max(0, drift))
        out.append(offset + shade(base_line, lvl))

    return "\n".join(out)


def muffle(ch: str, clarity: float) -> str:
    """Return a softened glyph as clarity drops to 0."""
    if ch.isspace():
        return ch
    c = max(0.0, min(1.0, clarity))
    if c > 0.85:
        return ch
    if c > 0.65:
        return '~' if ch != '~' else '-'
    if c > 0.45:
        return '-'
    if c > 0.25:
        return '.'
    return ' '


def wavechar(angle: float) -> str:
    """
    Choose an ASCII glyph that approximates a line at the given angle (radians).
    Range: [-pi, pi].
    """
    a = (angle + math.pi) % (2 * math.pi) - math.pi
    aa = abs(a)
    if aa < math.pi / 12:
        return "-"
    if aa < math.pi / 6:
        return "~"
    if aa < math.pi / 3:
        return "/" if a > 0 else "\\"
    if aa < math.pi * 5 / 12:
        return "/" if a > 0 else "\\"
    return "|"


def waveset() -> list[str]:
    """A compact set of wave-like ASCII glyphs for shading water."""
    return [" ", "·", ".", "-", "~", "/", "\\", "|"]


def reflect_wave(
    text: Iterable[str] | str,
    depth: int = 8,
    drift: int = 1,
    quiet: float = 0.6,
    amp: float = 2.0,
    freq: float = 0.15,
    phase: float = 0.0,
    fade: float = 0.85,
    sway: float = 0.0,
    swayfreq: float = 0.35,
    hush: float = 0.5,
    rows: Optional[int] = None,
    with_source: bool = False,
) -> str:
    """
    Reflect with a sine-wave ripple distortion and luminance modulation.

    - amp: horizontal ripple amplitude (chars)
    - freq: spatial frequency (cycles per char)
    - phase: initial phase (radians)
    - fade: per-row amplitude decay multiplier
    - quiet: overall dimming of reflection
    """
    if isinstance(text, str):
        lines = text.splitlines() or [text]
    else:
        lines = list(text)

    # Strip ANSI color codes from geometry source
    import re
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    clean_lines = [ansi_re.sub("", ln) for ln in lines]

    out: list[str] = []
    if with_source:
        out.extend(lines)

    base = "\n".join(clean_lines)
    base_line = base.split("\n")[-1] if base else ""
    W = len(base_line)
    if W == 0:
        return "\n".join(out)

    k = 2 * math.pi * freq

    if rows is not None and rows > 0:
        # Sample approximately "rows" levels across the full depth
        idxs = sorted({max(1, min(depth, round(1 + t * (depth - 1)))) for t in [j / max(1, rows) for j in range(1, rows + 1)]})
    else:
        idxs = list(range(1, depth + 1))

    for i in idxs:
        damp = fade ** i
        y = []
        for x in range(W):
            dx = amp * damp * math.sin(k * x + phase + i * 0.6)
            src = min(W - 1, max(0, int(round(x + dx))))
            ch = base_line[src]
            lum = quiet * (1.0 - i / (depth + 1)) * (0.75 + 0.25 * math.cos(k * x + phase + i * 0.6))
            clarity = max(0.0, 1.0 - hush * (i / (depth + 1)))
            y.append(shade(muffle(ch, clarity), max(0.0, min(1.0, lum))))
        rowshift = 0
        if sway != 0.0:
            rowshift = int(round(sway * damp * math.sin(swayfreq * i + phase)))
        base_offset = i * max(0, drift)
        total_offset = max(0, base_offset + rowshift)
        offset = " " * total_offset
        out.append(offset + "".join(y))

    return "\n".join(out)


# --- Emergent genomes -------------------------------------------------------

def seed(motif: Iterable[str] | str,
         pal: Optional[Callable[[float], tuple[int, int, int]]] = None,
         gate: Optional[Callable[[int], Callable[[int], str]]] = None,
         fx: Optional[dict] = None) -> dict:
    """
    Create a simple visual genome from a motif, palette, optional gate, and effects.

    - motif: glyphs or text fragments
    - pal: r->rgb palette (defaults to rainbow hue)
    - gate: width->template chooser (e.g., forms(...))
    - fx: ripple/reflect parameters: {amp,freq,phase,fade,quiet,sway,swayfreq,hush,rows,drift,depth}
    """
    if isinstance(motif, str):
        mot = [c for c in motif] if motif.strip() else [motif]
    else:
        mot = list(motif)
    if not mot:
        mot = [" "]
    return {
        "motif": mot,
        "pal": pal or hue,
        "gate": gate,
        "fx": fx or {}
    }


def splice(a: dict, b: dict, mix: float = 0.5) -> dict:
    """
    Emergent splice of two genomes by blending palettes and concatenating motifs.

    - mix in [0,1]: 0 -> a, 1 -> b
    Gate is borrowed from the dominant parent for simplicity.
    Effects are linearly mixed where numeric.
    """
    t = max(0.0, min(1.0, mix))
    mot = list(a.get("motif", [])) + list(b.get("motif", []))
    pa = a.get("pal", hue)
    pb = b.get("pal", hue)

    def pal(r: float) -> tuple[int, int, int]:
        return blend(pa(r), pb(r), t)

    ga = a.get("gate")
    gb = b.get("gate")
    gate = gb if t >= 0.5 else ga

    fxa = dict(a.get("fx", {}))
    fxb = dict(b.get("fx", {}))
    keys = set(fxa.keys()) | set(fxb.keys())
    fx: dict = {}
    for k in keys:
        va = fxa.get(k)
        vb = fxb.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            fx[k] = (1 - t) * va + t * vb
        else:
            fx[k] = vb if t >= 0.5 else va

    return {"motif": mot, "pal": pal, "gate": gate, "fx": fx}


def bloom(g: dict) -> Callable[[int], str]:
    """
    Render a genome: width -> multi-line composition with palette and ripple.
    Uses the motif to weave a colored band, then applies a rippled reflection.
    If a gate is provided, its form(width) is appended beneath.
    """
    mot = g.get("motif", [" "])
    pal = g.get("pal", hue)
    gate = g.get("gate")
    fx = {
        "amp": 2.0, "freq": 0.12, "phase": 0.0, "fade": 0.9,
        "quiet": 0.55, "sway": 3.0, "swayfreq": 0.4, "hush": 0.6,
        "rows": 2, "drift": 1, "depth": 8,
        **(g.get("fx", {})),
    }

    band = signal(mot)

    def at(width: int) -> str:
        w = max(20, int(width))
        line = "".join(band(w))
        colored = "".join(splash(line, pal))
        art = colored
        def fnum(key: str, default: float) -> float:
            val = fx.get(key, default)
            return float(val) if isinstance(val, (int, float)) else float(default)
        def inum(key: str, default: int) -> int:
            val = fx.get(key, default)
            return int(val) if isinstance(val, (int, float)) else int(default)
        ripple = reflect_wave(colored,
                              depth=inum("depth", 8),
                              drift=inum("drift", 1),
                              quiet=fnum("quiet", 0.55),
                              amp=fnum("amp", 2.0),
                              freq=fnum("freq", 0.12),
                              phase=fnum("phase", 0.0),
                              fade=fnum("fade", 0.9),
                              sway=fnum("sway", 0.0),
                              swayfreq=fnum("swayfreq", 0.35),
                              hush=fnum("hush", 0.5),
                              rows=inum("rows", 2))
        if gate is None:
            return art + "\n" + ripple
        form = gate(w)
        return art + "\n" + ripple + "\n" + form(w)

    return at


# --- Output normalization ---------------------------------------------------

def squeeze(text: str, max_lines: int = 12, head: int = 5, tail: int = 5, keep_runs: int = 1) -> str:
    """
    Reduce vertical size:
    - collapse adjacent identical lines to at most keep_runs
    - keep head/tail lines, eliding the middle with an ellipsis if needed
    """
    lines = text.splitlines()
    if not lines:
        return text
    # collapse runs
    collapsed: list[str] = []
    last = None
    count = 0
    for ln in lines:
        if ln == last:
            count += 1
            if count <= keep_runs:
                collapsed.append(ln)
        else:
            last = ln
            count = 1
            collapsed.append(ln)
    if len(collapsed) <= max_lines:
        return "\n".join(collapsed)
    h = max(0, min(head, max_lines))
    t = max(0, min(tail, max_lines - h - 1))
    if h + t + 1 > max_lines:
        t = max_lines - h - 1
        if t < 0:
            t = 0
            h = max_lines
    return "\n".join(collapsed[:h] + ["…"] + collapsed[-t:] if t > 0 else collapsed[:h])


# --- Text fractal blending --------------------------------------------------

def lex(text: str) -> list[str]:
    """Split into words and separators, preserving spaces/punct."""
    import re
    parts = re.findall(r"[A-Za-z0-9_]+|[^A-Za-z0-9_]+", text)
    return parts if parts else [text]


def common_prefix(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


def common_suffix(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[-1 - i] == b[-1 - i]:
        i += 1
    return a[len(a) - i:] if i else ""


def blend_word(a: str, b: str, t: float) -> str:
    """Blend two words by preserving common prefix/suffix and mixing middle.
    Bias the split point to the nearest vowel/consonant boundary.
    """
    pref = common_prefix(a, b)
    suff = common_suffix(a[len(pref):] if len(pref) < len(
        a) else a, b[len(pref):] if len(pref) < len(b) else b)
    core_a = a[len(pref):len(a) - len(suff) if len(suff) else len(a)]
    core_b = b[len(pref):len(b) - len(suff) if len(suff) else len(b)]
    m = max(len(core_a), len(core_b))
    if m == 0:
        return pref + suff

    vowels = set("aeiouAEIOUyY")
    def boundaries(s: str) -> list[int]:
        idxs: list[int] = []
        if not s:
            return idxs
        for i in range(1, len(s)):
            a_v = s[i-1] in vowels
            b_v = s[i] in vowels
            if a_v != b_v:
                idxs.append(i)
        return idxs

    # collect candidate boundary indices from both cores and the longer template
    template = core_a if len(core_a) >= len(core_b) else core_b
    cand = set(boundaries(core_a)) | set(boundaries(core_b)) | set(boundaries(template))
    if not cand:
        k = int(round(t * m))
    else:
        # desired split scaled into [0, m]
        s = t * m
        k = min(cand, key=lambda i: abs(i - s))
    k = max(0, min(m, k))

    out_chars: list[str] = []
    for i in range(m):
        if i < k:
            ch = core_a[i] if i < len(core_a) else (core_b[i] if i < len(core_b) else "")
        else:
            ch = core_b[i] if i < len(core_b) else (core_a[i] if i < len(core_a) else "")
        out_chars.append(ch)
    return pref + "".join(out_chars) + suff


def phrase(a: str, b: str, t: float, levels: int = 3) -> str:
    """
    Fractal-like blend between phrases a and b. t in [0,1].
    Uses multi-frequency sine mask across token positions.
    """
    ta = lex(a)
    tb = lex(b)
    n = max(len(ta), len(tb))
    ta += [""] * (n - len(ta))
    tb += [""] * (n - len(tb))
    import math as _m
    def weight(i: int) -> float:
        x = (i + 0.5) / max(1, n)
        s = 0.0
        amp = 1.0
        for k in range(levels):
            freq = 2 ** k
            s += amp * (0.5 * (1.0 + _m.sin(2 * _m.pi * (freq * x + t))))
            amp *= 0.5
        return s / (2 - 2 ** (1 - levels)) if levels > 0 else 0.5
    out: list[str] = []
    for i in range(n):
        ai = ta[i]
        bi = tb[i]
        if ai.isalnum() and bi.isalnum():
            out.append(blend_word(ai, bi, weight(i)))
        else:
            out.append(bi if weight(i) > 0.5 else ai)
    return "".join(out)


def speak(a: str, b: str, width: int, t: float, pal: Optional[Callable[[float], tuple[int, int, int]]] = None) -> str:
    """Blend two phrases at t and color-wash them, centered to width."""
    line = phrase(a, b, t)
    centered = center_text(line, width)
    colored = "".join(wash(centered, pal=pal))
    return colored


# --- Novelty-driven palette splicing ---------------------------------------

def norm_word(w: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_]+", "", w).lower()


def chunk_words(words: List[str], max_words: int = 8) -> List[List[str]]:
    out: List[List[str]] = []
    cur: List[str] = []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def palette_for(score: float) -> Callable[[float], tuple[int, int, int]]:
    s = max(0.0, min(1.0, score))
    if s < 0.15:
        return mono
    if s < 0.35:
        return solar
    if s < 0.55:
        return earth
    if s < 0.75:
        return sea
    if s < 0.9:
        return dusk
    return neon


def novel(text: str) -> Callable[[int], str]:
    """
    Return width -> string that colors text chunks by novelty of words.
    Novelty is fraction of new (not seen before) normalized words within a chunk.
    """
    words = text.split()
    chunks = chunk_words(words, max_words=8)
    seen: set[str] = set()
    scored: List[tuple[str, float]] = []
    for ch in chunks:
        toks = [norm_word(w) for w in ch]
        toks = [t for t in toks if t]
        if not toks:
            scored.append((" ".join(ch), 0.0))
            continue
        new = sum(1 for t in toks if t not in seen)
        for t in toks:
            seen.add(t)
        score = new / max(1, len(toks))
        scored.append((" ".join(ch), score))

    def at(width: int) -> str:
        lines: List[str] = []
        for text_chunk, score in scored:
            pal = palette_for(score)
            centered = center_text(text_chunk, width)
            colored = "".join(splash(centered, pal))
            lines.append(colored)
        # Subtle unified reflection beneath
        base = "\n".join(lines)
        ripple = reflect_wave(base, depth=6, drift=0, quiet=0.4, amp=1.5, freq=0.1, fade=0.9, hush=0.7, rows=2)
        return squeeze(base + "\n" + ripple, max_lines=20, head=8, tail=10)

    return at


def wire(text: str, width: int) -> str:
    """Thin wireframe border around text block, using airy corners."""
    lines = text.splitlines()
    inner = max(2, width - 2)
    top = "╭" + ("─" * inner) + "╮"
    bot = "╰" + ("─" * inner) + "╯"
    mid = []
    for ln in lines:
        s = ln[:inner]
        s += " " * max(0, inner - len(s))
        mid.append("│" + s + "│")
    return "\n".join([top] + mid + [bot])


def rng_from_seed(text: str) -> random.Random:
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    return random.Random(int.from_bytes(h, "big"))


def fuse(text: str) -> dict:
    """
    Deterministically generate a spliced genome from seed text.
    Picks two motifs, palettes, and effect params based on the seed, then splices.
    """
    r = rng_from_seed(text)

    motif_pool: List[Iterable[str] | str] = [
        ["◆", "◇"], ["□", "◇"], ["▲", "△"], ["╔", "╦", "╗"], ["╚", "╩", "╝"],
        ["─", "┄"], ["╭", "╮"], ["╰", "╯"], ["✶", "✷"], ["·", "•"],
    ]
    pals = [dusk, neon, sea, earth, fire, ice, pastel, mono, solar]

    def pick(seq):
        return seq[r.randrange(len(seq))]

    m1 = pick(motif_pool)
    m2 = pick(motif_pool)
    p1 = pick(pals)
    p2 = pick(pals)

    # Effects ranges tuned for pleasant output
    def fx_draw() -> dict:
        return {
            "amp": r.uniform(1.5, 3.5),
            "freq": r.uniform(0.08, 0.16),
            "phase": r.uniform(0.0, 2*math.pi),
            "fade": r.uniform(0.85, 0.95),
            "quiet": r.uniform(0.45, 0.65),
            "sway": r.uniform(0.0, 4.0),
            "swayfreq": r.uniform(0.2, 0.6),
            "hush": r.uniform(0.5, 0.8),
            "rows": r.choice([2, 3]),
            "drift": r.choice([0, 1, 2]),
            "depth": r.choice([6, 8, 10]),
        }

    # Optional gate: 50% chance add a thin dialog form
    gate = None
    if r.random() < 0.5:
        title = "Signal " + text[:12]
        gate = forms((0, lambda w: dialog(title, ["seeded splice"])(w)))

    g1 = seed(m1, pal=p1, gate=None, fx=fx_draw())
    g2 = seed(m2, pal=p2, gate=gate, fx=fx_draw())
    mix = r.random()
    return splice(g1, g2, mix)


if __name__ == "__main__":
    doctest.testmod()
