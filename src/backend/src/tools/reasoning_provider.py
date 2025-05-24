"""Reasoning Provider for the AGI system.

This module provides a FastAPI server that exposes endpoints for reasoning
and planning execution in the role of a developer.

Run with:

    uvicorn reasoning_provider:app --host 127.0.0.1 --port 3210

Environment variable `REASONING_PROVIDER_URL` should then be set to
```
export REASONING_PROVIDER_URL=http://127.0.0.1:3210
```

The server exposes the following endpoints:

1. POST /reasoning/plan - Generate a plan for executing a task.
2. POST /reasoning/analyze - Analyze code or text to identify issues or improvements.
3. POST /reasoning/evaluate - Evaluate a solution against requirements.
4. GET /reasoning/health - Check if the reasoning provider is healthy.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Any, Union

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import the GPTLikeModel for reasoning
try:
    from src.neural.gpt import GPTLikeModel
    from src.tools.neural_integration import GPTLikeModelProvider
    HAVE_GPT_MODEL = True
except ImportError:
    print("Warning: GPTLikeModel not found. Reasoning will use a simpler approach.")
    HAVE_GPT_MODEL = False

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Reasoning Provider", version="0.1.0")

# ---------------------------------------------------------------------------
# Reasoning model
# ---------------------------------------------------------------------------

# Initialize the GPTLikeModel for reasoning if available
if HAVE_GPT_MODEL:
    try:
        # Use environment variables for model configuration
        MODEL_PATH = os.environ.get("REASONING_MODEL_PATH", None)
        
        # Initialize the model provider
        reasoning_model = GPTLikeModelProvider(
            model_path=MODEL_PATH,
            # Use larger dimensions for reasoning tasks
            embed_dim=int(os.environ.get("REASONING_EMBED_DIM", "256")),
            hidden_dim=int(os.environ.get("REASONING_HIDDEN_DIM", "512")),
            # Use attention for better reasoning capabilities
            use_attention=True,
        )
        print("Using GPTLikeModel for reasoning")
    except Exception as e:
        print(f"Error initializing GPTLikeModel: {e}")
        print("Falling back to simpler reasoning approach")
        reasoning_model = None
else:
    reasoning_model = None

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PlanRequest(BaseModel):
    task: str = Field(..., description="The task to plan for")
    context: Optional[str] = Field(default=None, description="Additional context for the task")
    constraints: Optional[List[str]] = Field(default=None, description="Constraints to consider")
    max_steps: int = Field(default=10, ge=1, le=50, description="Maximum number of steps in the plan")

class PlanResponse(BaseModel):
    plan: List[str] = Field(..., description="The steps of the plan")
    reasoning: str = Field(..., description="Reasoning behind the plan")
    estimated_time: Optional[str] = Field(default=None, description="Estimated time to complete the task")

class AnalyzeRequest(BaseModel):
    content: str = Field(..., description="The code or text to analyze")
    content_type: str = Field(..., description="Type of content (e.g., 'code', 'text', 'requirements')")
    focus: Optional[List[str]] = Field(default=None, description="Specific aspects to focus on")

class AnalyzeResponse(BaseModel):
    analysis: str = Field(..., description="Analysis of the content")
    issues: List[Dict[str, Any]] = Field(..., description="Identified issues")
    suggestions: List[Dict[str, Any]] = Field(..., description="Improvement suggestions")

class EvaluateRequest(BaseModel):
    solution: str = Field(..., description="The solution to evaluate")
    requirements: str = Field(..., description="The requirements to evaluate against")
    criteria: Optional[List[str]] = Field(default=None, description="Specific evaluation criteria")

class EvaluateResponse(BaseModel):
    score: float = Field(..., ge=0, le=1, description="Overall score (0-1)")
    evaluation: str = Field(..., description="Detailed evaluation")
    strengths: List[str] = Field(..., description="Strengths of the solution")
    weaknesses: List[str] = Field(..., description="Weaknesses of the solution")
    recommendations: List[str] = Field(..., description="Recommendations for improvement")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def generate_plan(task: str, context: Optional[str], constraints: Optional[List[str]], max_steps: int) -> Dict[str, Any]:
    """Generate a plan for executing a task.
    
    Args:
        task: The task to plan for.
        context: Additional context for the task.
        constraints: Constraints to consider.
        max_steps: Maximum number of steps in the plan.
        
    Returns:
        A dictionary containing the plan, reasoning, and estimated time.
    """
    if HAVE_GPT_MODEL and reasoning_model is not None:
        # Construct the prompt for the GPTLikeModel
        prompt = f"Task: {task}\n"
        
        if context:
            prompt += f"Context: {context}\n"
            
        if constraints:
            prompt += "Constraints:\n"
            for i, constraint in enumerate(constraints, 1):
                prompt += f"{i}. {constraint}\n"
                
        prompt += f"\nCreate a detailed plan with up to {max_steps} steps to accomplish this task. For each step, provide a clear action. After the plan, explain your reasoning and provide an estimated time to complete the task."
        
        # Generate the response
        response = reasoning_model.generate(
            prompt=prompt,
            max_new_tokens=512,  # Longer response for detailed planning
            temperature=0.7,     # Some creativity but not too random
            use_history=False    # Don't include in conversation history
        )
        
        # Parse the response to extract plan, reasoning, and estimated time
        plan_lines = []
        reasoning_lines = []
        estimated_time = None
        
        # Simple parsing logic - can be improved
        sections = response.split("\n\n")
        for section in sections:
            if section.lower().startswith("plan:") or section.lower().startswith("steps:"):
                # Extract plan steps
                lines = section.split("\n")[1:]  # Skip the "Plan:" header
                for line in lines:
                    if line.strip():
                        # Remove step numbers if present
                        if line.strip()[0].isdigit() and ". " in line:
                            line = line.split(". ", 1)[1]
                        plan_lines.append(line.strip())
            elif section.lower().startswith("reasoning:") or "reason" in section.lower():
                # Extract reasoning
                if ":" in section:
                    reasoning_lines = section.split(":", 1)[1].strip().split("\n")
                else:
                    reasoning_lines = section.strip().split("\n")
            elif "time" in section.lower() or "duration" in section.lower() or "estimate" in section.lower():
                # Extract estimated time
                estimated_time = section.strip()
                
        # If we couldn't parse the sections properly, use a simpler approach
        if not plan_lines:
            # Just split by numbered lines
            for line in response.split("\n"):
                line = line.strip()
                if line and line[0].isdigit() and ". " in line:
                    plan_lines.append(line.split(". ", 1)[1])
                    
        if not reasoning_lines and "reason" in response.lower():
            # Try to extract reasoning from the full response
            start_idx = response.lower().find("reason")
            if start_idx != -1:
                end_idx = response.lower().find("time", start_idx)
                if end_idx == -1:
                    end_idx = len(response)
                reasoning_text = response[start_idx:end_idx].strip()
                reasoning_lines = reasoning_text.split("\n")
                
        if not estimated_time and "time" in response.lower():
            # Try to extract estimated time from the full response
            start_idx = response.lower().find("time")
            if start_idx != -1:
                estimated_time = response[start_idx:].strip()
                
        # Ensure we have at least something for each section
        if not plan_lines:
            plan_lines = ["1. Analyze the task requirements", 
                         "2. Design a solution approach",
                         "3. Implement the solution",
                         "4. Test the implementation",
                         "5. Review and refine"]
            
        if not reasoning_lines:
            reasoning_lines = ["This is a standard approach to software development tasks."]
            
        if not estimated_time:
            estimated_time = "Estimated time: Depends on the complexity of the task."
            
        return {
            "plan": plan_lines[:max_steps],  # Limit to max_steps
            "reasoning": "\n".join(reasoning_lines),
            "estimated_time": estimated_time
        }
    else:
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

def analyze_content(content: str, content_type: str, focus: Optional[List[str]]) -> Dict[str, Any]:
    """Analyze code or text to identify issues or improvements.
    
    Args:
        content: The code or text to analyze.
        content_type: Type of content (e.g., 'code', 'text', 'requirements').
        focus: Specific aspects to focus on.
        
    Returns:
        A dictionary containing the analysis, issues, and suggestions.
    """
    if HAVE_GPT_MODEL and reasoning_model is not None:
        # Construct the prompt for the GPTLikeModel
        prompt = f"Content type: {content_type}\n\n"
        prompt += f"Content to analyze:\n{content}\n\n"
        
        if focus:
            prompt += "Focus on the following aspects:\n"
            for aspect in focus:
                prompt += f"- {aspect}\n"
                
        prompt += "\nProvide a detailed analysis of the content, identifying issues and suggesting improvements. Format your response with clear sections for Analysis, Issues, and Suggestions."
        
        # Generate the response
        response = reasoning_model.generate(
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.7,
            use_history=False
        )
        
        # Parse the response to extract analysis, issues, and suggestions
        analysis = ""
        issues = []
        suggestions = []
        
        # Simple parsing logic
        current_section = None
        section_content = []
        
        for line in response.split("\n"):
            line = line.strip()
            
            if line.lower().startswith("analysis:"):
                if current_section and section_content:
                    if current_section == "analysis":
                        analysis = "\n".join(section_content)
                    elif current_section == "issues":
                        issues = [{"description": item} for item in section_content]
                    elif current_section == "suggestions":
                        suggestions = [{"description": item} for item in section_content]
                
                current_section = "analysis"
                section_content = []
            elif line.lower().startswith("issues:"):
                if current_section and section_content:
                    if current_section == "analysis":
                        analysis = "\n".join(section_content)
                    elif current_section == "issues":
                        issues = [{"description": item} for item in section_content]
                    elif current_section == "suggestions":
                        suggestions = [{"description": item} for item in section_content]
                
                current_section = "issues"
                section_content = []
            elif line.lower().startswith("suggestions:"):
                if current_section and section_content:
                    if current_section == "analysis":
                        analysis = "\n".join(section_content)
                    elif current_section == "issues":
                        issues = [{"description": item} for item in section_content]
                    elif current_section == "suggestions":
                        suggestions = [{"description": item} for item in section_content]
                
                current_section = "suggestions"
                section_content = []
            elif line and current_section:
                # Check if it's a list item
                if line.startswith("- ") or line.startswith("* "):
                    section_content.append(line[2:])
                elif line[0].isdigit() and ". " in line:
                    section_content.append(line.split(". ", 1)[1])
                else:
                    section_content.append(line)
        
        # Process the last section
        if current_section and section_content:
            if current_section == "analysis":
                analysis = "\n".join(section_content)
            elif current_section == "issues":
                issues = [{"description": item} for item in section_content]
            elif current_section == "suggestions":
                suggestions = [{"description": item} for item in section_content]
        
        # If we couldn't parse the sections properly, use the whole response as analysis
        if not analysis:
            analysis = response
            
        # Ensure we have at least empty lists for issues and suggestions
        if not issues:
            issues = []
            
        if not suggestions:
            suggestions = []
            
        return {
            "analysis": analysis,
            "issues": issues,
            "suggestions": suggestions
        }
    else:
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

def evaluate_solution(solution: str, requirements: str, criteria: Optional[List[str]]) -> Dict[str, Any]:
    """Evaluate a solution against requirements.
    
    Args:
        solution: The solution to evaluate.
        requirements: The requirements to evaluate against.
        criteria: Specific evaluation criteria.
        
    Returns:
        A dictionary containing the score, evaluation, strengths, weaknesses, and recommendations.
    """
    if HAVE_GPT_MODEL and reasoning_model is not None:
        # Construct the prompt for the GPTLikeModel
        prompt = f"Requirements:\n{requirements}\n\n"
        prompt += f"Solution:\n{solution}\n\n"
        
        if criteria:
            prompt += "Evaluation criteria:\n"
            for criterion in criteria:
                prompt += f"- {criterion}\n"
                
        prompt += "\nEvaluate how well the solution meets the requirements. Provide a detailed evaluation, list strengths and weaknesses, and offer recommendations for improvement. Also provide an overall score from 0 to 1, where 0 means the solution doesn't meet any requirements and 1 means it perfectly meets all requirements."
        
        # Generate the response
        response = reasoning_model.generate(
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.7,
            use_history=False
        )
        
        # Parse the response to extract evaluation, strengths, weaknesses, recommendations, and score
        evaluation = ""
        strengths = []
        weaknesses = []
        recommendations = []
        score = 0.5  # Default score
        
        # Simple parsing logic
        current_section = None
        section_content = []
        
        for line in response.split("\n"):
            line = line.strip()
            
            if line.lower().startswith("evaluation:"):
                if current_section and section_content:
                    if current_section == "evaluation":
                        evaluation = "\n".join(section_content)
                    elif current_section == "strengths":
                        strengths = section_content
                    elif current_section == "weaknesses":
                        weaknesses = section_content
                    elif current_section == "recommendations":
                        recommendations = section_content
                
                current_section = "evaluation"
                section_content = []
            elif line.lower().startswith("strengths:"):
                if current_section and section_content:
                    if current_section == "evaluation":
                        evaluation = "\n".join(section_content)
                    elif current_section == "strengths":
                        strengths = section_content
                    elif current_section == "weaknesses":
                        weaknesses = section_content
                    elif current_section == "recommendations":
                        recommendations = section_content
                
                current_section = "strengths"
                section_content = []
            elif line.lower().startswith("weaknesses:"):
                if current_section and section_content:
                    if current_section == "evaluation":
                        evaluation = "\n".join(section_content)
                    elif current_section == "strengths":
                        strengths = section_content
                    elif current_section == "weaknesses":
                        weaknesses = section_content
                    elif current_section == "recommendations":
                        recommendations = section_content
                
                current_section = "weaknesses"
                section_content = []
            elif line.lower().startswith("recommendations:"):
                if current_section and section_content:
                    if current_section == "evaluation":
                        evaluation = "\n".join(section_content)
                    elif current_section == "strengths":
                        strengths = section_content
                    elif current_section == "weaknesses":
                        weaknesses = section_content
                    elif current_section == "recommendations":
                        recommendations = section_content
                
                current_section = "recommendations"
                section_content = []
            elif line.lower().startswith("score:"):
                # Extract score
                try:
                    score_text = line.split(":", 1)[1].strip()
                    # Handle different formats (e.g., "0.8", "8/10", "80%")
                    if "/" in score_text:
                        num, denom = score_text.split("/", 1)
                        score = float(num) / float(denom)
                    elif "%" in score_text:
                        score = float(score_text.replace("%", "")) / 100
                    else:
                        score = float(score_text)
                        
                    # Ensure score is between 0 and 1
                    score = max(0, min(1, score))
                except (ValueError, IndexError):
                    # If we can't parse the score, use the default
                    score = 0.5
            elif line and current_section:
                # Check if it's a list item
                if line.startswith("- ") or line.startswith("* "):
                    section_content.append(line[2:])
                elif line[0].isdigit() and ". " in line:
                    section_content.append(line.split(". ", 1)[1])
                else:
                    section_content.append(line)
        
        # Process the last section
        if current_section and section_content:
            if current_section == "evaluation":
                evaluation = "\n".join(section_content)
            elif current_section == "strengths":
                strengths = section_content
            elif current_section == "weaknesses":
                weaknesses = section_content
            elif current_section == "recommendations":
                recommendations = section_content
        
        # If we couldn't parse the sections properly, use the whole response as evaluation
        if not evaluation:
            evaluation = response
            
        # Ensure we have at least empty lists for strengths, weaknesses, and recommendations
        if not strengths:
            strengths = []
            
        if not weaknesses:
            weaknesses = []
            
        if not recommendations:
            recommendations = []
            
        return {
            "score": score,
            "evaluation": evaluation,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations
        }
    else:
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
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reasoning/plan", response_model=PlanResponse)
def plan(req: PlanRequest):
    """Generate a plan for executing a task."""
    try:
        result = generate_plan(req.task, req.context, req.constraints, req.max_steps)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating plan: {str(e)}")

@app.post("/reasoning/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """Analyze code or text to identify issues or improvements."""
    try:
        result = analyze_content(req.content, req.content_type, req.focus)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing content: {str(e)}")

@app.post("/reasoning/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    """Evaluate a solution against requirements."""
    try:
        result = evaluate_solution(req.solution, req.requirements, req.criteria)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating solution: {str(e)}")

@app.get("/reasoning/health", response_model=HealthResponse)
def health_check():
    """Check if the reasoning provider is healthy."""
    try:
        if HAVE_GPT_MODEL and reasoning_model is not None:
            return {"status": "healthy", "details": {"model": "GPTLikeModel"}}
        else:
            return {"status": "healthy", "details": {"model": "fallback"}}
    except Exception as e:
        return {"status": "unhealthy", "details": {"message": str(e)}}