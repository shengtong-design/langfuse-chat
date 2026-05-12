from .fitness_training import FitnessTrainingCrew
from .researcher import ResearcherCrew

# To add a new crew: create crews/<name>.py, subclass BaseCrew, add it here.
CREWS = {
    "researcher": ResearcherCrew,
    "fitness_training": FitnessTrainingCrew,
}
