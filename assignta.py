import pandas as pd
import numpy as np
import random as rnd
from evo import Evo
from profiler import profile, Profiler

# Load data
sections = pd.read_csv('sections.csv')
tas = pd.read_csv('tas.csv')
ta_prefs = tas.iloc[:, 3:].replace({'U': 0, 'W': 1, 'P': 2}).values
time_slots = sections['daytime'].values
conflict_matrix = np.array([[1 if t1 == t2 else 0 for t2 in time_slots] for t1 in time_slots])

# Objective Functions
@profile
def overallocation(solution):
    ta_assignments = solution.sum(axis=1)
    max_assigned = tas['max_assigned'].values
    penalties = np.maximum(ta_assignments - max_assigned, 0)
    return penalties.sum()

@profile
def conflicts(solution):
    assignments_per_ta = solution.dot(conflict_matrix)
    has_conflict = np.any(assignments_per_ta > 1, axis=1)
    return has_conflict.sum()

@profile
def undersupport(solution):
    assigned_tas = solution.sum(axis=0)
    min_ta = sections['min_ta'].values
    penalties = np.maximum(min_ta - assigned_tas, 0)
    return penalties.sum()

@profile
def unavailable(solution):
    return np.sum(solution * (ta_prefs == 0))

@profile
def unpreferred(solution):
    return np.sum(solution * (ta_prefs == 1))

# Agents
@profile
def random_solution(_):
    return np.random.randint(0, 2, size=(tas.shape[0], sections.shape[0]))

@profile
def reduce_overallocation(solutions):
    sol = solutions[0].copy()
    ta_assigns = sol.sum(axis=1)
    over = np.where(ta_assigns > tas['max_assigned'].values)[0]
    if over.size > 0:
        ta = rnd.choice(over)
        assigned = np.where(sol[ta] == 1)[0]
        if assigned.size > 0:
            sol[ta, rnd.choice(assigned)] = 0
    return sol

@profile
def fix_undersupport(solutions):
    sol = solutions[0].copy()
    assigned = sol.sum(axis=0)
    under = np.where(assigned < sections['min_ta'].values)[0]
    if under.size > 0:
        sec = rnd.choice(under)
        available_tas = np.where(ta_prefs[:, sec] > 0)[0]
        if available_tas.size > 0:
            sol[rnd.choice(available_tas), sec] = 1
    return sol

@profile
def swap_assignments(solutions):
    sol = solutions[0].copy()
    ta1, ta2 = rnd.sample(range(tas.shape[0]), 2)
    sec1, sec2 = rnd.sample(range(sections.shape[0]), 2)
    sol[ta1, sec1], sol[ta2, sec2] = sol[ta2, sec2], sol[ta1, sec1]
    return sol

# Main execution
def main():
    evo = Evo()
    evo.add_objective('overallocation', overallocation)
    evo.add_objective('conflicts', conflicts)
    evo.add_objective('undersupport', undersupport)
    evo.add_objective('unavailable', unavailable)
    evo.add_objective('unpreferred', unpreferred)

    evo.add_agent('random', random_solution)
    evo.add_agent('reduce_over', reduce_overallocation)
    evo.add_agent('fix_under', fix_undersupport)
    evo.add_agent('swap', swap_assignments)

    for _ in range(10):
        evo.add_solution(random_solution([]))

    evo.evolve(time_limit=300)

    summary = evo.summarize('KELH')
    with open('KELH_summary.csv', 'w') as f:
        f.write(summary)

    # Generate profiling report
    with open('KELH_profile.txt', 'w') as f:
        f.write("Function              Calls     TotSec   Sec/Call\n")
        for name, num in Profiler.calls.items():
            sec = Profiler.time[name]
            f.write(f'{name:20s} {num:6d} {sec:10.6f} {sec / num:10.6f}\n')

if __name__ == "__main__":
    main()
    