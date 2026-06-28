starter code for a 3-hour practicum (lab) following a lecture on multi-objective decision making under uncertainty, from the perspective of Bayesian optimization. the lab will have students working in groups. 

the lab targets undergraduate as well as graduate students, so should be easy to follow and well documented.

i'm working in the ubuntu partition of my HP Omen, which has a rtx 3070. but students will use the cpu, so unless instructed use the cpu version of the packages and test the code on the cpu only.

all modules should be tested and tests written in `tests` with intuitive numerical cases.

the high-level initial outline, which may change later on, is in `docs/01_outline.md`. this outline file should be updated with every commit, as the plan evolves.

do not refer to the docs anywhere in the student-facing code or jupyter notebooks. it will only confuse them.

`docs` contains my design document in markdown files. i'll later put my lecture slides here as well.
`scripts` contains scripts (relevant to just me) for coming up with the starter code and data.
`notebooks` contains jupyter notebooks the students will work through.
`tests` contains tests and its directory structure should mirror the repo directory structure, e.g., (`tests/scripts/test_parse_data.py`)
