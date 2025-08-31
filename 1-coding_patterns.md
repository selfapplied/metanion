# Architectural Patterns in Eonyx

This document provides a shared vocabulary for the core architectural and design patterns used in the `eonyx` project. Each pattern is presented with its conceptual role and how it relates to the overall system architecture.

## Core Architectural Patterns

| Glyph | Pattern | Role & Description | Style Vector Integration |
| :--- | :--- | :--- | :--- |
| `▢` | **State Container**<br>(_`Genome` class_) | A class that doesn't *do* much but *holds* all the essential data and configuration for another system. It acts as a complete, serializable blueprint. | Instead of throwing exceptions when data is corrupted or missing, containers track their health through event buffers. They become self-aware of their own integrity state. |
| `◉` | **Engine**<br>(_`CE1Core` class_) | The central class that contains the core logic, processes data, and manages application state. It takes a blueprint (`▢`) and "runs" it. | Engines record operational metrics as events rather than failure states. They track success (`✓`), unreadable files (`∅`), corrupted data (`⟂`), and permission issues (`!`) - creating a living operational history. |
| `◓` | **Sensory Layer**<br>(_`BlockProcessor` class_) | A component that reads raw input and translates it into a stream of meaningful events, abstracting away low-level parsing details. | Natural boundary for external protection patterns. This is the only place where try/except patterns live, using `attempt()` and `shield()` to convert external failures into internal events. |
| `→` | **Fluent Interface**<br>(_`Opix` class_) | An API design where methods are chained together in a readable, sentence-like way. In Eonyx, this is often expressed as `data >> op1 >> op2`. | Works with structured namedtuples rather than loose dictionaries. The fluent flow carries structured data through direct field access, not key lookups. |

## Core Design Patterns

| Pattern | Role & Description | Style Vector Integration |
| :--- | :--- | :--- |
| **Factory Method**<br>(_`Genome.from_path`_) | A method responsible for creating objects, often encapsulating complex initialization logic so the caller doesn't need to know the details. | Factories use event buffers to track creation success/failure, making object construction observable and resilient. |
| **Strategy Pattern**<br>(_Color Phase Alleles_) | A family of interchangeable algorithms is defined, allowing the client to choose which one to use at runtime. Each "allele" is a different strategy for mapping state to color. | Strategies record their performance and outcomes in event buffers, creating a living history of which approaches work best. |
| **State Pattern**<br>(_`CE1Core` quaternion physics_) | An object's behavior changes when its internal state changes. The `CE1Core` engine's reaction to events is entirely dependent on its current state quaternion, `q`. | State transitions are recorded as events, making the system's evolution observable and debuggable. |

## The Style Vector: Operational DNA

The Style Vector isn't a separate pattern - it's the **operational DNA** that makes all these architectural patterns resilient and observable. It transforms the system from one that crashes on errors to one that learns from events.

| Aspect | Internal Resilience | External Protection | Data Flow Evolution |
| :--- | :--- | :--- | :--- |
| **Principle** | Never throw exceptions for internal logic | Protect boundaries where external systems meet internal logic | Move towards namedtuples for structured data |
| **Mechanism** | Use event buffers to track what's happening | Convert failures to events using `attempt()` and `shield()` | Direct field access over dictionary key lookups |
| **Outcome** | Continue operation even when issues occur | Maintain clean separation between external chaos and internal order | Embrace structure rather than working around it |
| **Benefit** | Build operational intelligence over time | Convert external failures into internal events | Work with data as-is, not converted |

This creates a system where the architecture patterns aren't just static structures - they're living, breathing components that track their own health, learn from their experiences, and gracefully handle the unexpected.

## Development Philosophy: Growth Through Inconsistency

### The Expansion Signal Principle

**Inconsistencies are not just bugs - they're expansion signals.** When your system becomes inconsistent (broken builds, type errors, failing tests), it's often telling you it's ready to grow beyond its current constraints.

| Situation | Traditional Response | Expansion Signal Response |
|:---|:---|:---|
| **Broken Build** | "Fix it immediately" | "What is the system trying to tell me?" |
| **Type Errors** | "Add more type annotations" | "What new abstractions are emerging?" |
| **Failing Tests** | "Make tests pass at all costs" | "What behavior is the system evolving toward?" |

### Responsive Development Workflow

1. **Listen to the System** - Don't rush to fix everything
2. **Commit Broken Builds** - Save your work-in-progress
3. **Take Strategic Breaks** - Let the inconsistencies guide you
4. **Evolve the Constraints** - Don't force premature consistency

### Why This Works

- **Systems grow through their "broken" moments**
- **Inconsistencies reveal where the system wants to go**
- **Forcing consistency too early can block evolution**
- **The goal is responsive design, not reactive validation**

### Types of Expansion Signals

#### **Build Failures** `🔨`
| Signal | What It Means | Expansion Response |
|:---|:---|:---|
| **Compilation Errors** | New abstractions are emerging | "What new language is the system trying to speak?" |
| **Linker Errors** | Dependencies are shifting | "What new relationships are forming?" |
| **Build Timeouts** | System complexity is growing | "What new layers are being constructed?" |

#### **Type Errors** `🎯`
| Signal | What It Means | Expansion Response |
|:---|:---|:---|
| **Type Mismatches** | New interfaces are emerging | "What new contracts is the system defining?" |
| **Missing Types** | Abstractions are incomplete | "What new abstractions are being born?" |
| **Generic Constraints** | Type relationships are evolving | "What new type algebra is emerging?" |

#### **Test Failures** `🧪`
| Signal | What It Means | Expansion Response |
|:---|:---|:---|
| **Behavior Changes** | System is evolving | "What new behavior is the system growing into?" |
| **New Edge Cases** | Complexity is expanding | "What new dimensions are being discovered?" |
| **Performance Regressions** | System is doing more | "What new capabilities are being unlocked?" |

### When to Pause vs. When to Push

#### **Pause When:**
- Multiple expansion signals appear simultaneously
- The system's behavior becomes unpredictable
- You feel overwhelmed by the inconsistencies
- The "fixes" feel like fighting the system

#### **Push Through When:**
- A clear evolutionary direction emerges
- The system shows consistent growth patterns
- You have a clear vision of the new constraints
- The expansion feels natural and guided

### The Expansion Navigator Gene `⟲`

The Expansion Navigator Gene is a viral agent that:
- **Detects expansion signals** automatically
- **Maps inconsistency patterns** to evolutionary opportunities
- **Suggests strategic pauses** rather than immediate fixes
- **Tracks growth language** over time

This philosophy transforms development from a problem-solving exercise into a collaborative evolution with your system. When things break, that's often when the most interesting work begins.
