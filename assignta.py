"""
assignta.py - TA Assignment to Lab Sections using Evolutionary Computing

This module implements the objectives and agents for solving the TA-to-lab assignment problem
using evolutionary computing.
"""

import numpy as np
import pandas as pd
import random
import copy
import time
from evo import Evo


class AssignTA:
    def __init__(self, sections_file='sections.csv', tas_file='tas.csv'):
        """Initialize the AssignTA problem with data from sections and TAs files."""
        # Load data
        self.sections = pd.read_csv(sections_file)
        self.tas = pd.read_csv(tas_file)

        # Convert preference columns to numeric for easier processing
        # U = Unavailable (0), W = Willing (1), P = Preferred (2)
        self.pref_map = {'U': 0, 'W': 1, 'P': 2}

        # Create a preferences matrix
        num_tas = len(self.tas)
        num_sections = len(self.sections)
        self.preferences = np.zeros((num_tas, num_sections))

        for ta_idx in range(num_tas):
            for section_idx in range(num_sections):
                # Get the preference value from the corresponding column
                pref_val = self.tas.iloc[ta_idx][str(section_idx)]
                self.preferences[ta_idx, section_idx] = self.pref_map.get(pref_val, 0)

        # Initialize the evolutionary computing framework
        self.evo = Evo()

        # Register objectives
        self.evo.add_objective("overallocation", self.overallocation)
        self.evo.add_objective("conflicts", self.conflicts)
        self.evo.add_objective("undersupport", self.undersupport)
        self.evo.add_objective("unavailable", self.unavailable)
        self.evo.add_objective("unpreferred", self.unpreferred)

        # Register agents
        self.evo.add_agent("random_solution", self.random_solution)
        self.evo.add_agent("swap_assignments", self.swap_assignments, 1)
        self.evo.add_agent("fix_overallocation", self.fix_overallocation, 1)
        self.evo.add_agent("fix_undersupport", self.fix_undersupport, 1)
        self.evo.add_agent("fix_unavailable", self.fix_unavailable, 1)

    def random_solution(self, _):
        """Generate a random solution for the TA assignment problem."""
        num_tas = len(self.tas)
        num_sections = len(self.sections)

        # Initialize an empty assignment matrix
        # 1 means TA is assigned to section, 0 means not assigned
        solution = np.zeros((num_tas, num_sections), dtype=int)

        # For each TA, assign to random sections up to their max_assigned
        for ta_idx in range(num_tas):
            max_assigned = self.tas.iloc[ta_idx]['max_assigned']
            # Get sections where the TA is at least willing (preference > 0)
            available_sections = [i for i in range(num_sections)
                                  if self.preferences[ta_idx, i] > 0]

            # If there are available sections, assign the TA to some of them
            if available_sections:
                # Determine how many sections to assign (up to max_assigned)
                num_to_assign = min(max_assigned, len(available_sections))
                sections_to_assign = random.sample(available_sections, num_to_assign)

                # Assign the TA to those sections
                for section_idx in sections_to_assign:
                    solution[ta_idx, section_idx] = 1

        return solution

    def swap_assignments(self, solutions):
        """Swap TA assignments between sections."""
        solution = solutions[0].copy()
        num_tas = len(self.tas)
        num_sections = len(self.sections)

        # Pick two random TAs
        ta1, ta2 = random.sample(range(num_tas), 2)

        # Find sections where they're assigned
        ta1_sections = [i for i in range(num_sections) if solution[ta1, i] == 1]
        ta2_sections = [i for i in range(num_sections) if solution[ta2, i] == 1]

        if not ta1_sections or not ta2_sections:
            return solution  # No swap possible

        # Pick one section from each TA
        section1 = random.choice(ta1_sections)
        section2 = random.choice(ta2_sections)

        # Check if swap is possible based on preferences
        if (self.preferences[ta1, section2] > 0 and
                self.preferences[ta2, section1] > 0):
            # Swap the assignments
            solution[ta1, section1] = 0
            solution[ta1, section2] = 1
            solution[ta2, section1] = 1
            solution[ta2, section2] = 0

        return solution

    def fix_overallocation(self, solutions):
        """Fix overallocation of TAs by removing some assignments."""
        solution = solutions[0].copy()
        num_tas = len(self.tas)
        num_sections = len(self.sections)

        # Find TAs who are overallocated
        overallocated_tas = []
        for ta_idx in range(num_tas):
            assigned_count = np.sum(solution[ta_idx])
            max_assigned = self.tas.iloc[ta_idx]['max_assigned']
            if assigned_count > max_assigned:
                overallocated_tas.append((ta_idx, assigned_count - max_assigned))

        if not overallocated_tas:
            return solution  # No overallocation to fix

        # Pick a random overallocated TA and remove random assignments
        ta_idx, excess = random.choice(overallocated_tas)
        assigned_sections = [i for i in range(num_sections) if solution[ta_idx, i] == 1]

        # Remove assignments from least preferred sections first
        section_prefs = [(i, self.preferences[ta_idx, i]) for i in assigned_sections]
        section_prefs.sort(key=lambda x: x[1])  # Sort by preference value

        # Remove 'excess' assignments
        for i in range(min(excess, len(section_prefs))):
            section_idx = section_prefs[i][0]
            solution[ta_idx, section_idx] = 0

        return solution

    def fix_undersupport(self, solutions):
        """Fix undersupported sections by adding more TAs."""
        solution = solutions[0].copy()
        num_tas = len(self.tas)
        num_sections = len(self.sections)

        # Find undersupported sections
        undersupported = []
        for section_idx in range(num_sections):
            assigned_count = np.sum(solution[:, section_idx])
            min_required = self.sections.iloc[section_idx]['min_ta']
            if assigned_count < min_required:
                undersupported.append((section_idx, min_required - assigned_count))

        if not undersupported:
            return solution  # No undersupported sections

        # Pick a random undersupported section
        section_idx, shortage = random.choice(undersupported)

        # Find TAs who could be assigned to this section
        potential_tas = []
        for ta_idx in range(num_tas):
            # Check if TA is not already assigned to this section
            if solution[ta_idx, section_idx] == 0:
                # Check if TA has availability and preference
                if self.preferences[ta_idx, section_idx] > 0:
                    # Check if TA has room for more assignments
                    assigned_count = np.sum(solution[ta_idx])
                    max_assigned = self.tas.iloc[ta_idx]['max_assigned']
                    if assigned_count < max_assigned:
                        potential_tas.append((ta_idx, self.preferences[ta_idx, section_idx]))

        # Sort potential TAs by preference (highest first)
        potential_tas.sort(key=lambda x: x[1], reverse=True)

        # Assign TAs to the section
        for i in range(min(shortage, len(potential_tas))):
            ta_idx = potential_tas[i][0]
            solution[ta_idx, section_idx] = 1

        return solution

    def fix_unavailable(self, solutions):
        """Fix assignments where TAs are unavailable."""
        solution = solutions[0].copy()
        num_tas = len(self.tas)
        num_sections = len(self.sections)

        # Find unavailable assignments
        unavailable = []
        for ta_idx in range(num_tas):
            for section_idx in range(num_sections):
                if (solution[ta_idx, section_idx] == 1 and
                        self.preferences[ta_idx, section_idx] == 0):  # Unavailable
                    unavailable.append((ta_idx, section_idx))

        if not unavailable:
            return solution  # No unavailable assignments

        # Pick a random unavailable assignment
        ta_idx, section_idx = random.choice(unavailable)

        # Remove the assignment
        solution[ta_idx, section_idx] = 0

        # Find a replacement TA if possible
        potential_tas = []
        for new_ta_idx in range(num_tas):
            if new_ta_idx != ta_idx and solution[new_ta_idx, section_idx] == 0:
                # Check if new TA is available and has preference
                if self.preferences[new_ta_idx, section_idx] > 0:
                    # Check if new TA has room for more assignments
                    assigned_count = np.sum(solution[new_ta_idx])
                    max_assigned = self.tas.iloc[new_ta_idx]['max_assigned']
                    if assigned_count < max_assigned:
                        potential_tas.append((new_ta_idx, self.preferences[new_ta_idx, section_idx]))

        # Sort potential TAs by preference (highest first)
        potential_tas.sort(key=lambda x: x[1], reverse=True)

        # Assign a new TA if possible
        if potential_tas:
            new_ta_idx = potential_tas[0][0]
            solution[new_ta_idx, section_idx] = 1

        return solution

    # Objective functions
    def overallocation(self, solution):
        """
        Objective 1: Minimize overallocation of TAs.
        Calculates penalty for assigning more labs than a TA's max_assigned.
        """
        num_tas = len(self.tas)
        penalty = 0

        for ta_idx in range(num_tas):
            assigned_count = np.sum(solution[ta_idx])
            max_assigned = self.tas.iloc[ta_idx]['max_assigned']
            if assigned_count > max_assigned:
                penalty += assigned_count - max_assigned

        return penalty

    def conflicts(self, solution):
        """
        Objective 2: Minimize time conflicts.
        Counts TAs with at least one time conflict.
        """
        num_tas = len(self.tas)
        num_sections = len(self.sections)
        unique_times = self.sections['daytime'].unique()

        # Count TAs with time conflicts
        tas_with_conflicts = 0

        for ta_idx in range(num_tas):
            has_conflict = False

            # Check each unique time
            for time in unique_times:
                # Find sections at this time
                time_sections = [i for i in range(num_sections)
                                 if self.sections.iloc[i]['daytime'] == time]

                # Count assignments at this time
                assignments_at_time = sum(solution[ta_idx, i] for i in time_sections)

                if assignments_at_time > 1:
                    has_conflict = True
                    break

            if has_conflict:
                tas_with_conflicts += 1

        return tas_with_conflicts

    def undersupport(self, solution):
        """
        Objective 3: Minimize Under-Support.
        Calculates penalty for not assigning enough TAs to sections.
        """
        num_sections = len(self.sections)
        penalty = 0

        for section_idx in range(num_sections):
            assigned_count = np.sum(solution[:, section_idx])
            min_required = self.sections.iloc[section_idx]['min_ta']
            if assigned_count < min_required:
                penalty += min_required - assigned_count

        return penalty

    def unavailable(self, solution):
        """
        Objective 4: Minimize assignments where TAs are unavailable.
        """
        num_tas = len(self.tas)
        num_sections = len(self.sections)
        count = 0

        for ta_idx in range(num_tas):
            for section_idx in range(num_sections):
                if (solution[ta_idx, section_idx] == 1 and
                        self.preferences[ta_idx, section_idx] == 0):  # Unavailable
                    count += 1

        return count

    def unpreferred(self, solution):
        """
        Objective 5: Minimize assignments where TAs are willing but don't prefer.
        """
        num_tas = len(self.tas)
        num_sections = len(self.sections)
        count = 0

        for ta_idx in range(num_tas):
            for section_idx in range(num_sections):
                if (solution[ta_idx, section_idx] == 1 and
                        self.preferences[ta_idx, section_idx] == 1):  # Willing but not preferred
                    count += 1

        return count

    def run_optimization(self, time_limit=300, iterations=1000, dom_freq=100, status_freq=1000):
        """Run the optimization process for the specified time limit."""
        # Generate some initial solutions
        for _ in range(10):
            self.evo.add_solution(self.random_solution([]))

        # Run the evolutionary process
        start_time = time.time()
        self.evo.evolve(n=iterations, dom=dom_freq, status=status_freq, time_limit=time_limit)

        return self.summarize()

    def summarize(self, group_name="mygroup"):
        """
        Summarize the non-dominated solutions in the required format.
        Returns a dataframe and also saves to a CSV.
        """
        summary_data = []

        for eval_key, solution in self.evo.pop.items():
            # Convert evaluation key to dictionary
            eval_dict = dict(eval_key)

            # Create a row with group name and objectives
            row = {
                'groupname': group_name,
                'overallocation': eval_dict['overallocation'],
                'conflicts': eval_dict['conflicts'],
                'undersupport': eval_dict['undersupport'],
                'unavailable': eval_dict['unavailable'],
                'unpreferred': eval_dict['unpreferred']
            }

            summary_data.append(row)

        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)

        # Save to CSV
        csv_name = f"{group_name}_summary.csv"
        summary_df.to_csv(csv_name, index=False)

        return summary_df


# Main execution
if __name__ == "__main__":
    assigner = AssignTA()
    results = assigner.run_optimization(time_limit=300)
    print(results)