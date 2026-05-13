from .fitness_crew import FitnessCrew
from .research_crew import ResearchCrew

# To add a new crew: create crews/<name>_crew.py, subclass BaseCrew, register here.
CREWS = {
    "researcher": ResearchCrew,
    "fitness_training": FitnessCrew,
}
