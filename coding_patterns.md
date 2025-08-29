# A Guide to Discussing and Evolving Eonyx Code

This document provides a shared vocabulary and set of principles for referencing, describing, and improving the code within the `eonyx` project.

## Part 1: How to Reference Code Clearly

When discussing code, precision helps avoid ambiguity. Let's use these simple conventions:

*   **Files and Directories:** Use backticks and the full path from the project root, e.g., `eonyx/genome.py`.
*   **Classes:** Use backticks and the class name, e.g., `CE1Core`.
*   **Methods and Functions:** Use backticks with the class and method name, or just the function name, e.g., `Genome.from_path()` or `run_genome()`.
*   **Variables/Attributes:** Use backticks, e.g., `self.color_phase_alleles`.
*   **Specific Lines:** When necessary, refer to a specific line number or range, e.g., "In `ce1.py`, line 780, the event loop begins."

## Part 2: Describing Code Patterns in Plain English

This project uses several powerful architectural patterns. Here are some plain English descriptions for them, using `eonyx` code as examples.

*   **"The Blueprint" or "State Container" (`Genome` class)**
    *   **What it is:** A class that doesn't *do* much but *holds* all the essential data and configuration for another system.
    *   **In Eonyx:** The `Genome` class in `eonyx/genome.py` is the perfect example. It bundles the `EngineConfig`, `Grammar`, and all file `assets` into a single, serializable object that represents a complete "virtual organism."

*   **"The Engine" or "Virtual Machine" (`CE1Core` class)**
    *   **What it is:** The central class that contains the core logic, processes the data, and manages the state of the application.
    *   **In Eonyx:** `CE1Core` in `eonyx/ce1.py` is the engine. It takes a `Genome` (the blueprint) and "runs" it, simulating its physics and learning from its assets.

*   **"The Sensory Layer" or "Event Stream" (`BlockProcessor` in `deflate.py`)**
    *   **What it is:** A component that reads raw input and translates it into a stream of meaningful events, abstracting away the low-level parsing details.
    *   **In Eonyx:** The `BlockProcessor` doesn't just decompress a file; it acts as the sensory organ for the `CE1Core`. It yields events like `("block", ...)` or `("literal", ...)` that the engine can react to.

*   **"Fluent Interface" or "Operator Programming" (`Aspirate` in `aspire.py`)**
    *   **What it is:** An API design where methods are chained together in a readable, sentence-like way, often returning a modified version of the object to allow for further chaining.
    *   **In Eonyx:** The `Aspirate` class is a masterclass in this. Instead of writing nested functions, you can write `my_aspirate.apply(op1).apply(op2)`, or more elegantly, `my_aspirate >> op1 >> op2`. This makes complex transformations on symbolic data highly expressive.

*   **"The Entrypoint" or "CLI Handler" (`main()` in `eonyx.py` and `zipc.py`)**
    *   **What it is:** The function that handles command-line arguments and kicks off the main process. It's the public-facing interface for a user running the program from the terminal.
    *   **In Eonyx:** `eonyx.py` is the entrypoint for *running* a genome, while `zipc.py` is the entrypoint for *compiling* one.

## Part 3: Refactoring - Improving Existing Code

Refactoring is the art of improving the internal structure of code without changing its external behavior. It's about making code cleaner, more understandable, and easier to modify.

*   **"Extract Method"**
    *   **What it is:** Identifying a small, cohesive piece of logic inside a larger method and moving it into its own new method with a descriptive name.
    *   **Why it's useful:** It makes the original method shorter and easier to read, and it makes the extracted logic reusable and easier to test. The monolithic `FractalMarkovAlleleEngine` is a good example of a class that could have benefited from this, whereas `CE1Core` is more cleanly separated.

*   **"Encapsulate Logic"**
    *   **What it is:** Grouping related data and the functions that operate on that data into a single class.
    *   **Why it's useful:** This reduces complexity by hiding internal details. The `Genome` class encapsulates all the complexity of the zip file format, config, and grammar, presenting a clean and simple interface to the rest of the program.

*   **"Don't Repeat Yourself" (DRY)**
    *   **What it is:** The principle that any single piece of knowledge or logic in a system should have a single, unambiguous representation. Avoid copy-pasting code.
    *   **Why it's useful:** Duplicated code is a common source of bugs. If you need to fix or update the logic, you have to remember to do it in every place it was copied. A single, authoritative function is much safer.

## Part 4: Useful Software Design Patterns

Design patterns are general, reusable solutions to commonly occurring problems within a given context in software design.

*   **Factory Method (`Genome.from_path`)**
    *   **What it is:** A method that is responsible for creating objects, often with some complex initialization logic.
    *   **In Eonyx:** `Genome.from_path()` is a factory. It encapsulates all the complexity of opening a zip file, reading the `config.msgpack` and `grammar.msgpack`, and assembling a valid `Genome` object. The code that calls it doesn't need to know about any of those details.

*   **Strategy Pattern (Color Phase Alleles)**
    *   **What it is:** Defining a family of interchangeable algorithms and letting the client choose which one to use at runtime.
    *   **In Eonyx:** The "Color Phase Alleles" concept from `fme_core.py` is a perfect example. Each "allele" is a different *strategy* for mapping an internal state (`delta`) to a color. The engine dynamically chooses the best strategy at runtime based on the context.

*   **State Pattern (`CE1Core`'s quaternion physics)**
    *   **What it is:** A pattern where an object's behavior changes when its internal state changes.
    *   **In Eonyx:** The `CE1Core` implements a continuous version of this. Its behavior (how it reacts to events, the `delta` it produces) is entirely dependent on the current value of its state quaternion, `q`. As `q` rotates through the simulation, the engine's "behavior" subtly changes with it.
