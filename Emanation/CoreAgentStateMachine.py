from typing import Optional, Generic, TypeVar, Dict, Any, List, Callable, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from queue import Queue
import logging

C = TypeVar('C')  # Context type
S = TypeVar('S')  # State identifier type


@dataclass
class StateFlags:
    """Tracks state processing flags"""
    flags: Set[str] = field(default_factory=set)

    def set(self, flag: str) -> None:
        self.flags.add(flag)

    def clear(self) -> None:
        self.flags.clear()

    def has(self, flag: str) -> bool:
        return flag in self.flags


@dataclass
class ProcessContext(Generic[C]):
    """Wraps user context with processing metadata"""
    user_context: C
    output_queue: Queue
    flags: StateFlags = field(default_factory=StateFlags)

    def clear_flags(self) -> None:
        self.flags.clear()


class State(ABC, Generic[C]):
    """Base state implementation"""

    @abstractmethod
    def process(self, context: ProcessContext[C]) -> None:
        """Process the state and set appropriate flags"""
        pass


@dataclass
class Transition(Generic[S, C]):
    """Transition definition"""
    target: S
    condition: Callable[[ProcessContext[C]], bool]


class StateMachine(Generic[S, C]):
    """Core state machine implementation"""

    def __init__(self):
        self.states: Dict[S, State[C]] = {}
        self.transitions: Dict[S, List[Transition[S, C]]] = {}
        self._start_state: Optional[S] = None
        self.max_iterations = 99

        # Basic logger setup
        self.logger = logging.getLogger("StateMachine")

    def register_state(self, state_id: S, state: State[C]) -> 'StateMachine[S, C]':
        """Register a new state"""
        self.states[state_id] = state
        self.transitions[state_id] = []
        return self

    def add_transition(self,
                       from_state: S,
                       to_state: S,
                       condition: Callable[[ProcessContext[C]], bool]) -> 'StateMachine[S, C]':
        """Add a transition with a flag-based condition"""
        if from_state not in self.transitions:
            self.transitions[from_state] = []
        self.transitions[from_state].append(Transition(to_state, condition))
        return self

    def set_start_state(self, state_id: S) -> 'StateMachine[S, C]':
        """Set the starting state"""
        if state_id not in self.states:
            raise ValueError(f"State {state_id} not registered")
        self._start_state = state_id
        return self

    def process(self, context: C, output_queue: Queue) -> S:
        """Main processing loop"""
        if not self._start_state:
            raise ValueError("Start state not set")

        wrapped_context = ProcessContext(
            user_context=context,
            output_queue=output_queue,  # Pass through the queue
            flags=StateFlags()
        )

        current_state_id = self._start_state
        iterations = 0

        try:
            while iterations < self.max_iterations:
                self.logger.info(f"📎 Current State: {current_state_id}")

                # Process current state
                wrapped_context.clear_flags()
                current_state = self.states[current_state_id]

                try:
                    current_state.process(wrapped_context)
                except Exception as e:
                    self.logger.error(f"❌ State processing error: {str(e)}")
                    wrapped_context.flags.set("ERROR")
                    # Look for error transition or break
                    next_state_id = self._find_error_transition(current_state_id)
                    if next_state_id:
                        current_state_id = next_state_id
                        continue
                    break

                # Check transitions in order
                next_state_id = None
                for transition in self.transitions[current_state_id]:
                    if transition.condition(wrapped_context):
                        next_state_id = transition.target
                        break

                if next_state_id is None:
                    break

                current_state_id = next_state_id
                iterations += 1

            if iterations >= self.max_iterations:
                self.logger.error(f"❌ Maximum iterations ({self.max_iterations}) reached")
                wrapped_context.flags.set("MAX_ITERATIONS_ERROR")

        except Exception as e:
            self.logger.error(f"❌ Processing error: {str(e)}")
            wrapped_context.flags.set("ERROR")

        return current_state_id

    def _find_error_transition(self, current_state: S) -> Optional[S]:
        """Find error transition if it exists"""
        for transition in self.transitions[current_state]:
            if "error" in str(transition.target).lower():
                return transition.target
        return None


