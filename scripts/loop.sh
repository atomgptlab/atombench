#!/bin/bash
set -euo pipefail
for dir in */
	do cd "$dir"
		if [ ! -f "AI-AtomGen-prop-dft_3d-test-rmse.csv" ]; then
			cd ..
			continue
		fi
		rm -rf distribution*.pdf metrics.json
		python ../../scripts/plot_error_distribution.py #| tail -n 30 > metrics.txt
		cd ..
done
python ../scripts/json_to_csv.py
