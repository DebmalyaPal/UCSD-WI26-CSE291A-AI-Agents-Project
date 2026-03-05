# CREW Submodule

This folder contains the CREW repository (Apache 2.0) used for wildfire simulation.
Do not edit directly unless you are modifying CREW code.
See main README.md for instructions on running agents and simulation.

## Modifications to CREW

- Added a Dockerfile in `external/CREW/crew-algorithms/crew_algorithms/wildfire_alg/` for containerized agent execution.
- Updated Docker Compose file in `external/CREW/crew-dojo/Nakama/` for containerized agent execution.
- All other code remains unmodified.
- Original CREW repository: https://github.com/generalroboticslab/CREW
- Licensed under Apache 2.0.