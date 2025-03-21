##!/bin/bash
set -x

RUNS = 10
for i in {1..10}
	do
		python3 load_data.py
	done
