# Run each formulation on the same params
python -m ilp sweep --model srg_exact --params src/ilp/srg_params.csv --timeout 60 -o srg_exact.json --lex lex_leader
python -m ilp sweep --model srg_quadratic --params src/ilp/srg_params.csv --timeout 60 -o srg_quadratic.json --lex lex_leader
python -m ilp sweep --model srg_relaxed --params src/ilp/srg_params.csv --timeout 60 -o srg_relaxed.json --lex lex_leader

# Plot all three overlaid
python -m ilp plot srg_exact.json srg_quadratic.json srg_relaxed.json --timeout 60
