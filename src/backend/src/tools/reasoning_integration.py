"""Integration between MCP Server and Reasoning Provider.

This module provides functions for integrating the MCP Server with the Reasoning
Provider. It extends the MCP Server to use the Reasoning Provider for planning,
analyzing, and evaluating tasks.

Usage:
    from src.tools.reasoning_integration import ReasoningClient

    # Initialize the Reasoning Client
    reasoning_client = ReasoningClient()

    # Use it in the MCP Server
    plan = reasoning_client.plan("Implement a new feature")
"""

from __future__ import annotations

import os
import json
import requests
from typing import Dict, List, Any, Optional, Union

# ---------------------------------------------------------------------------
# Reasoning Provider client
# ---------------------------------------------------------------------------

# Default Reasoning Provider URL (can be overridden with environment variable)
REASONING_PROVIDER_URL = os.environ.get("REASONING_PROVIDER_URL", "http://localhost:3210")

class ReasoningClient:
    """Client for interacting with the Reasoning Provider."""

    def __init__(self, base_url: str = REASONING_PROVIDER_URL):
        """Initialize the Reasoning client.

        Args:
            base_url: The base URL of the Reasoning Provider.
        """
        self.base_url = base_url

    def health(self) -> Dict[str, Any]:
        """Check if the Reasoning Provider is healthy.

        Returns:
            A dictionary with the health status.
        """
        try:
            response = requests.get(f"{self.base_url}/reasoning/health")
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "details": {"message": str(e)}}

    def plan(self, task: str, context: Optional[str] = None, 
             constraints: Optional[List[str]] = None, max_steps: int = 10) -> Dict[str, Any]:
        """Generate a plan for executing a task.

        Args:
            task: The task to plan for.
            context: Additional context for the task.
            constraints: Constraints to consider.
            max_steps: Maximum number of steps in the plan.

        Returns:
            A dictionary containing the plan, reasoning, and estimated time.
        """
        data = {
            "task": task,
            "context": context,
            "constraints": constraints,
            "max_steps": max_steps
        }
        try:
            response = requests.post(f"{self.base_url}/reasoning/plan", json=data)
            return response.json()
        except Exception as e:
            # Fallback to a simple template-based approach
            plan = [
                "1. Understand the requirements and constraints",
                "2. Research existing solutions and approaches",
                "3. Design a solution architecture",
                "4. Break down the implementation into manageable tasks",
                "5. Implement core functionality first",
                "6. Add additional features incrementally",
                "7. Write tests to verify functionality",
                "8. Refactor and optimize the code",
                "9. Document the solution",
                "10. Review and finalize"
            ]
            
            reasoning = (
                "This plan follows a standard software development lifecycle. "
                "It starts with understanding the problem, then moves to design, "
                "implementation, testing, and finally documentation and review. "
                "This approach ensures that the solution is well-thought-out, "
                "properly implemented, and thoroughly tested."
            )
            
            return {
                "plan": plan[:max_steps],  # Limit to max_steps
                "reasoning": reasoning,
                "estimated_time": "Estimated time: Depends on the complexity of the task."
            }

    def analyze(self, content: str, content_type: str, 
                focus: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze code or text to identify issues or improvements.

        Args:
            content: The code or text to analyze.
            content_type: Type of content (e.g., 'code', 'text', 'requirements').
            focus: Specific aspects to focus on.

        Returns:
            A dictionary containing the analysis, issues, and suggestions.
        """
        data = {
            "content": content,
            "content_type": content_type,
            "focus": focus
        }
        try:
            response = requests.post(f"{self.base_url}/reasoning/analyze", json=data)
            return response.json()
        except Exception as e:
            # Fallback to a simple template-based approach
            if content_type.lower() == "code":
                analysis = "This appears to be code. A proper analysis would examine code quality, structure, and potential bugs."
                issues = [
                    {"description": "Without a more sophisticated analysis tool, specific issues cannot be identified."},
                    {"description": "Consider running a linter or static analysis tool on this code."}
                ]
                suggestions = [
                    {"description": "Ensure proper error handling is implemented."},
                    {"description": "Add comprehensive unit tests."},
                    {"description": "Document the code thoroughly."}
                ]
            elif content_type.lower() == "text":
                analysis = "This appears to be text content. A proper analysis would examine clarity, structure, and effectiveness."
                issues = [
                    {"description": "Without a more sophisticated analysis tool, specific issues cannot be identified."}
                ]
                suggestions = [
                    {"description": "Ensure the text is clear and concise."},
                    {"description": "Structure the content with appropriate headings and paragraphs."},
                    {"description": "Review for grammar and spelling errors."}
                ]
            elif content_type.lower() == "requirements":
                analysis = "This appears to be requirements documentation. A proper analysis would examine completeness, clarity, and testability."
                issues = [
                    {"description": "Without a more sophisticated analysis tool, specific issues cannot be identified."}
                ]
                suggestions = [
                    {"description": "Ensure requirements are specific, measurable, achievable, relevant, and time-bound (SMART)."},
                    {"description": "Include acceptance criteria for each requirement."},
                    {"description": "Prioritize requirements based on business value and implementation complexity."}
                ]
            else:
                analysis = f"This appears to be {content_type} content. A proper analysis would require domain-specific knowledge."
                issues = [
                    {"description": "Without a more sophisticated analysis tool, specific issues cannot be identified."}
                ]
                suggestions = [
                    {"description": "Review the content with domain experts."},
                    {"description": "Ensure the content meets its intended purpose."}
                ]
                
            return {
                "analysis": analysis,
                "issues": issues,
                "suggestions": suggestions
            }

    def evaluate(self, solution: str, requirements: str, 
                 criteria: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate a solution against requirements.

        Args:
            solution: The solution to evaluate.
            requirements: The requirements to evaluate against.
            criteria: Specific evaluation criteria.

        Returns:
            A dictionary containing the score, evaluation, strengths, weaknesses, and recommendations.
        """
        data = {
            "solution": solution,
            "requirements": requirements,
            "criteria": criteria
        }
        try:
            response = requests.post(f"{self.base_url}/reasoning/evaluate", json=data)
            return response.json()
        except Exception as e:
            # Fallback to a simple template-based approach
            evaluation = "Without a more sophisticated evaluation tool, a detailed analysis cannot be provided. Consider reviewing the solution against the requirements manually."
            strengths = ["The solution appears to address the problem domain."]
            weaknesses = ["Without a more sophisticated evaluation tool, specific weaknesses cannot be identified."]
            recommendations = [
                "Ensure the solution fully addresses all requirements.",
                "Test the solution thoroughly against the requirements.",
                "Get feedback from stakeholders or domain experts."
            ]
            
            return {
                "score": 0.5,  # Neutral score
                "evaluation": evaluation,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "recommendations": recommendations
            }

# ---------------------------------------------------------------------------
# MCP Server integration
# ---------------------------------------------------------------------------

class ReasoningProvider:
    """Provider for reasoning capabilities in the MCP Server."""

    def __init__(self, reasoning_url: str = REASONING_PROVIDER_URL):
        """Initialize the Reasoning Provider.

        Args:
            reasoning_url: The URL of the Reasoning Provider.
        """
        self.client = ReasoningClient(reasoning_url)
        
        # Check if the Reasoning Provider is healthy
        health = self.client.health()
        if health["status"] != "healthy":
            print(f"Warning: Reasoning Provider is not healthy: {health}")
            print("Reasoning capabilities will use fallback mechanisms")
            self._healthy = False
        else:
            self._healthy = True
            print("Reasoning Provider is healthy")

    def generate_plan(self, task: str, context: Optional[str] = None, 
                     constraints: Optional[List[str]] = None, max_steps: int = 10) -> Dict[str, Any]:
        """Generate a plan for executing a task.

        Args:
            task: The task to plan for.
            context: Additional context for the task.
            constraints: Constraints to consider.
            max_steps: Maximum number of steps in the plan.

        Returns:
            A dictionary containing the plan, reasoning, and estimated time.
        """
        return self.client.plan(task, context, constraints, max_steps)

    def analyze_content(self, content: str, content_type: str, 
                       focus: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze code or text to identify issues or improvements.

        Args:
            content: The code or text to analyze.
            content_type: Type of content (e.g., 'code', 'text', 'requirements').
            focus: Specific aspects to focus on.

        Returns:
            A dictionary containing the analysis, issues, and suggestions.
        """
        return self.client.analyze(content, content_type, focus)

    def evaluate_solution(self, solution: str, requirements: str, 
                         criteria: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate a solution against requirements.

        Args:
            solution: The solution to evaluate.
            requirements: The requirements to evaluate against.
            criteria: Specific evaluation criteria.

        Returns:
            A dictionary containing the score, evaluation, strengths, weaknesses, and recommendations.
        """
        return self.client.evaluate(solution, requirements, criteria)

    def is_healthy(self) -> bool:
        """Check if the Reasoning Provider is healthy.

        Returns:
            True if the Reasoning Provider is healthy, False otherwise.
        """
        return self._healthy