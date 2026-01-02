"""Workflow management for Nd/Dy separation paper.

Defines computational tasks with dependencies and manages execution order.
Caches intermediate results for reproducibility and efficiency.
"""

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import jax.numpy as jnp
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

# Output directories
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CACHE_DIR = RESULTS_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

TABLES_DIR = Path(__file__).parent.parent / "manuscript" / "tables"
TABLES_DIR.mkdir(exist_ok=True)

FIGURES_DIR = Path(__file__).parent.parent / "manuscript" / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# =============================================================================
# Task Definition
# =============================================================================

@dataclass
class Task:
    """A computational task in the workflow.

    Attributes:
        name: Unique task identifier
        func: Callable that performs the computation
        dependencies: List of task names this depends on
        description: Human-readable description
        cache: Whether to cache results
    """
    name: str
    func: Callable
    dependencies: list[str] = field(default_factory=list)
    description: str = ""
    cache: bool = True

    def __hash__(self):
        return hash(self.name)


@dataclass
class TaskResult:
    """Result of a task execution.

    Attributes:
        task_name: Name of the task
        data: The computed result
        elapsed_time: Computation time in seconds
        timestamp: When the task was run
        from_cache: Whether result was loaded from cache
    """
    task_name: str
    data: Any
    elapsed_time: float
    timestamp: str
    from_cache: bool = False


# =============================================================================
# Workflow Runner
# =============================================================================

class Workflow:
    """Manages task execution with dependency resolution and caching.

    Example:
        >>> wf = Workflow()
        >>> wf.add_task(Task("base_case", run_base_case))
        >>> wf.add_task(Task("sensitivity", run_sensitivity, dependencies=["base_case"]))
        >>> results = wf.run_all()
    """

    def __init__(self, use_cache: bool = True, verbose: bool = True):
        """Initialize workflow.

        Args:
            use_cache: Whether to use cached results
            verbose: Whether to print progress
        """
        self.tasks: dict[str, Task] = {}
        self.results: dict[str, TaskResult] = {}
        self.use_cache = use_cache
        self.verbose = verbose

    def add_task(self, task: Task):
        """Add a task to the workflow."""
        self.tasks[task.name] = task

    def _get_cache_path(self, task_name: str) -> Path:
        """Get cache file path for a task."""
        return CACHE_DIR / f"{task_name}.pkl"

    def _load_from_cache(self, task_name: str) -> Optional[TaskResult]:
        """Load task result from cache if available."""
        cache_path = self._get_cache_path(task_name)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                    result.from_cache = True
                    return result
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Cache load failed for {task_name}: {e}")
        return None

    def _save_to_cache(self, result: TaskResult):
        """Save task result to cache."""
        cache_path = self._get_cache_path(result.task_name)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Cache save failed for {result.task_name}: {e}")

    def _topological_sort(self) -> list[str]:
        """Sort tasks by dependencies (Kahn's algorithm)."""
        # Build in-degree map
        in_degree = {name: 0 for name in self.tasks}
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep in self.tasks:
                    in_degree[task.name] += 1

        # Start with tasks that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Sort for deterministic order
            queue.sort()
            current = queue.pop(0)
            result.append(current)

            # Reduce in-degree for dependent tasks
            for task in self.tasks.values():
                if current in task.dependencies:
                    in_degree[task.name] -= 1
                    if in_degree[task.name] == 0:
                        queue.append(task.name)

        if len(result) != len(self.tasks):
            raise ValueError("Circular dependency detected in workflow")

        return result

    def run_task(self, task_name: str, force: bool = False) -> TaskResult:
        """Run a single task.

        Args:
            task_name: Name of task to run
            force: Force re-computation even if cached

        Returns:
            TaskResult with computed data
        """
        task = self.tasks[task_name]

        # Check cache
        if self.use_cache and task.cache and not force:
            cached = self._load_from_cache(task_name)
            if cached is not None:
                if self.verbose:
                    print(f"  [cached] {task_name}")
                self.results[task_name] = cached
                return cached

        # Ensure dependencies are run
        dep_results = {}
        for dep in task.dependencies:
            if dep not in self.results:
                self.run_task(dep)
            dep_results[dep] = self.results[dep].data

        # Run the task
        if self.verbose:
            print(f"  [running] {task_name}...")

        start_time = time.time()
        try:
            data = task.func(dep_results)
        except Exception as e:
            print(f"  [ERROR] {task_name}: {e}")
            raise
        elapsed = time.time() - start_time

        result = TaskResult(
            task_name=task_name,
            data=data,
            elapsed_time=elapsed,
            timestamp=datetime.now().isoformat(),
            from_cache=False
        )

        # Save to cache
        if task.cache:
            self._save_to_cache(result)

        self.results[task_name] = result

        if self.verbose:
            print(f"  [done] {task_name} ({elapsed:.2f}s)")

        return result

    def run_all(self, force: bool = False) -> dict[str, TaskResult]:
        """Run all tasks in dependency order.

        Args:
            force: Force re-computation of all tasks

        Returns:
            Dictionary of task_name -> TaskResult
        """
        order = self._topological_sort()

        if self.verbose:
            print(f"\nWorkflow: {len(order)} tasks")
            print("=" * 50)

        for task_name in order:
            self.run_task(task_name, force=force)

        if self.verbose:
            print("=" * 50)
            total_time = sum(r.elapsed_time for r in self.results.values() if not r.from_cache)
            cached_count = sum(1 for r in self.results.values() if r.from_cache)
            print(f"Completed: {len(self.results)} tasks ({cached_count} cached)")
            print(f"Total compute time: {total_time:.2f}s")

        return self.results

    def clear_cache(self):
        """Clear all cached results."""
        for path in CACHE_DIR.glob("*.pkl"):
            path.unlink()
        if self.verbose:
            print("Cache cleared")

    def get_result(self, task_name: str) -> Any:
        """Get the data from a task result."""
        if task_name in self.results:
            return self.results[task_name].data
        raise KeyError(f"Task '{task_name}' has not been run")


# =============================================================================
# Result Serialization Helpers
# =============================================================================

def to_serializable(obj: Any) -> Any:
    """Convert JAX/numpy arrays to serializable format."""
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    elif hasattr(obj, '_asdict'):  # NamedTuple
        return {k: to_serializable(v) for k, v in obj._asdict().items()}
    elif hasattr(obj, '__dataclass_fields__'):  # dataclass
        return {k: to_serializable(getattr(obj, k)) for k in obj.__dataclass_fields__}
    else:
        return obj


def save_results_json(results: dict, filename: str):
    """Save results to JSON file."""
    output_path = RESULTS_DIR / filename
    with open(output_path, 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print(f"Results saved to: {output_path}")


def load_results_json(filename: str) -> dict:
    """Load results from JSON file."""
    input_path = RESULTS_DIR / filename
    with open(input_path, 'r') as f:
        return json.load(f)


# =============================================================================
# Summary Report Generator
# =============================================================================

def generate_summary_report(workflow: Workflow) -> str:
    """Generate a summary report of workflow execution."""
    lines = [
        "# Workflow Execution Summary",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Task Results",
        "",
    ]

    for task_name, result in workflow.results.items():
        task = workflow.tasks[task_name]
        status = "cached" if result.from_cache else f"{result.elapsed_time:.2f}s"
        lines.append(f"- **{task_name}**: {status}")
        if task.description:
            lines.append(f"  - {task.description}")
        if task.dependencies:
            lines.append(f"  - Depends on: {', '.join(task.dependencies)}")

    lines.extend([
        "",
        "## Output Files",
        "",
    ])

    # List generated files
    for table in sorted(TABLES_DIR.glob("*.csv")):
        lines.append(f"- Table: `{table.name}`")
    for fig in sorted(FIGURES_DIR.glob("*.png")):
        lines.append(f"- Figure: `{fig.name}`")

    return "\n".join(lines)
