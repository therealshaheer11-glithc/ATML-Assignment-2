"""CPU entrypoint for the complete saved-results audit, including length analysis.

Added after the experiments. Produces length_strata.csv, word_limits.csv and
 generation_ceiling.csv together with common metrics. For balanced training use
 task1_dpo.train --run-name length_balanced; this command never trains.
"""
from task1_dpo.analyze_results import main

if __name__ == "__main__":
    main()
