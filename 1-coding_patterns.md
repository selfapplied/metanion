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
