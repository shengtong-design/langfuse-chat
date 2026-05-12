import contextlib
import io
from typing import Any, Dict

from .base import BaseCrew


class FitnessTrainingCrew(BaseCrew):
    @property
    def crew_name(self) -> str:
        return "crewai.fitness_training"

    def run(self, inputs: Dict[str, Any], obs: Any) -> Dict[str, Any]:
        from crewai import Agent, Crew, Task

        goals = inputs["goals"]
        fitness_level = inputs["fitness_level"]
        equipment = inputs["equipment"]
        time_per_week = inputs["time_per_week"]
        limitations = inputs.get("limitations", "None specified")

        fitness_analyst = Agent(
            role="Fitness Analyst",
            goal="Analyze user fitness goals, current fitness level, and preferences to create a comprehensive fitness profile",
            backstory=(
                "You are an experienced fitness analyst with expertise in understanding "
                "different body types, fitness levels, and training goals. You have helped "
                "thousands of clients achieve their fitness objectives through careful "
                "analysis and personalized recommendations."
            ),
            verbose=True,
            allow_delegation=False,
        )
        workout_designer = Agent(
            role="Workout Program Designer",
            goal="Design effective and safe workout programs tailored to individual needs and goals",
            backstory=(
                "You are a certified personal trainer and workout program designer with "
                "over 15 years of experience. You specialize in creating progressive "
                "training programs that maximize results while minimizing injury risk. "
                "You understand periodization, exercise selection, and program structure."
            ),
            verbose=True,
            allow_delegation=False,
        )
        nutrition_advisor = Agent(
            role="Nutrition Advisor",
            goal="Provide nutrition guidance that supports the fitness goals and training program",
            backstory=(
                "You are a certified nutritionist specializing in sports nutrition. "
                "You understand how to optimize nutrition for different training goals "
                "including muscle building, fat loss, and athletic performance."
            ),
            verbose=True,
            allow_delegation=False,
        )

        task_analysis = Task(
            description=(
                f"Analyze the user's fitness profile based on the following information:\n"
                f"- User Goals: {goals}\n"
                f"- Current Fitness Level: {fitness_level}\n"
                f"- Available Equipment: {equipment}\n"
                f"- Time Available: {time_per_week} hours per week\n"
                f"- Any Limitations: {limitations}\n\n"
                "Create a comprehensive fitness profile that includes:\n"
                "1. Assessment of current fitness state\n"
                "2. Realistic goal timeline\n"
                "3. Key areas to focus on\n"
                "4. Potential challenges and how to overcome them"
            ),
            expected_output=(
                "A detailed fitness profile analysis in markdown format including "
                "current state assessment, goal analysis, focus areas, and recommendations."
            ),
            agent=fitness_analyst,
        )
        task_workout = Task(
            description=(
                "Based on the fitness analysis, design a complete workout program that includes:\n"
                "1. Weekly workout schedule\n"
                "2. Specific exercises for each session\n"
                "3. Sets, reps, and rest periods\n"
                "4. Progression plan for the next 4-8 weeks\n"
                "5. Warm-up and cool-down routines\n\n"
                f"Consider the user's available time of {time_per_week} hours per week "
                f"and equipment: {equipment}"
            ),
            expected_output=(
                "A complete workout program in markdown format with weekly schedule, "
                "detailed exercise descriptions, and progression guidelines."
            ),
            agent=workout_designer,
        )
        task_nutrition = Task(
            description=(
                "Create nutrition recommendations that support the workout program and goals:\n"
                "1. Daily caloric needs estimate\n"
                "2. Macronutrient recommendations\n"
                "3. Meal timing suggestions around workouts\n"
                "4. Hydration guidelines\n"
                "5. Supplement recommendations (if appropriate)\n\n"
                f"User's goal: {goals}"
            ),
            expected_output=(
                "Nutrition guidelines in markdown format including calorie and macro "
                "recommendations, meal timing, and practical eating suggestions."
            ),
            agent=nutrition_advisor,
        )

        crew = Crew(
            agents=[fitness_analyst, workout_designer, nutrition_advisor],
            tasks=[task_analysis, task_workout, task_nutrition],
            verbose=True,
        )

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        with obs.span(
            "crewai.fitness_training",
            "chain",
            input_data=inputs,
            metadata={"framework": "crewai", "crew": self.crew_name},
        ) as root:
            with obs.span(
                "crew.kickoff",
                "span",
                input_data=inputs,
            ) as kickoff:
                try:
                    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                        result = crew.kickoff()

                    # Combine all three task outputs into one structured plan.
                    # Guard against unexpected CrewOutput shape.
                    tasks_out = result.tasks_output
                    if tasks_out and len(tasks_out) >= 3:
                        combined = (
                            "## Fitness Profile Analysis\n\n" + tasks_out[0].raw + "\n\n"
                            "## Workout Program\n\n" + tasks_out[1].raw + "\n\n"
                            "## Nutrition Plan\n\n" + tasks_out[2].raw
                        )
                    else:
                        combined = str(result)

                    kickoff.update(output={"result": combined, "stdout": stdout_buf.getvalue(), "stderr": stderr_buf.getvalue()})
                    root.update(output={"result": combined})
                    return {"result": combined, "stdout": stdout_buf.getvalue(), "stderr": stderr_buf.getvalue()}
                except Exception as e:
                    kickoff.update(output={"error": repr(e), "stdout": stdout_buf.getvalue(), "stderr": stderr_buf.getvalue()}, level="ERROR")
                    root.update(output={"error": repr(e)}, level="ERROR")
                    raise
