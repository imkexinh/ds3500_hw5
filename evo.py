import random as rnd
import copy
from functools import reduce
import time
from profiler import profile


class Evo:
    def __init__(self):
        self.pop = {}  # population: evaluation --> solution
        self.fitness = {}  # objectives: name --> function
        self.agents = {}  # agents: name --> (operator, k)

    def add_objective(self, name, f):
        self.fitness[name] = f

    def add_agent(self, name, op, k=1):
        self.agents[name] = (op, k)

    def get_random_solutions(self, k=1):
        if len(self.pop) == 0:
            return []
        solutions = tuple(self.pop.values())
        return [copy.deepcopy(rnd.choice(solutions)) for _ in range(k)]

    def add_solution(self, sol):
        eval = tuple([(name, f(sol)) for name, f in self.fitness.items()])
        self.pop[eval] = sol

    def run_agent(self, name):
        op, k = self.agents[name]
        picks = self.get_random_solutions(k)
        new_solution = op(picks)
        self.add_solution(new_solution)

    @staticmethod
    def _dominates(p, q):
        pscores = [score for _, score in p]
        qscores = [score for _, score in q]
        score_diffs = list(map(lambda x, y: y - x, pscores, qscores))
        min_diff = min(score_diffs)
        max_diff = max(score_diffs)
        return min_diff >= 0.0 and max_diff > 0.0

    @staticmethod
    def _reduce_nds(S, p):
        return S - {q for q in S if Evo._dominates(p, q)}

    def remove_dominated(self):
        nds = reduce(Evo._reduce_nds, self.pop.keys(), self.pop.keys())
        self.pop = {k: self.pop[k] for k in nds}

    @profile
    def evolve(self, dom=100, status=1000, time_limit=300):
        agent_names = list(self.agents.keys())
        start_time = time.time()
        i = 0
        while time.time() - start_time < time_limit:
            pick = rnd.choice(agent_names)
            self.run_agent(pick)
            if i % dom == 0:
                self.remove_dominated()
            if i % status == 0:
                self.remove_dominated()
                print(f"Iteration: {i}, Population size: {len(self.pop)}")
            i += 1
        self.remove_dominated()

    def summarize(self, groupname):
        # Output non-dominated solutions in CSV format
        header = "groupname,overallocation,conflicts,undersupport,unavailable,unpreferred"
        rows = [header]
        for eval in self.pop.keys():
            scores = dict(eval)
            row = f"{groupname},{scores['overallocation']},{scores['conflicts']},{scores['undersupport']},{scores['unavailable']},{scores['unpreferred']}"
            rows.append(row)
        return "\n".join(rows)

    def __str__(self):
        rslt = ""
        for eval, sol in self.pop.items():
            rslt += str(dict(eval)) + ":\t" + str(sol) + "\n"
        return rslt